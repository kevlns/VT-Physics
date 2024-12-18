#include "UGNSCudaApi.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "Core/Math/helper_math_cu11.6.h"

namespace VT_Physics::UGNS {

#define THREAD_CHECK()  \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= d_config->totalParticleNum) \
        return;

#define dCONFIG(name) \
    d_config->name

#define dPARAMS(name, ind) \
    d_params->name[ind]

    /**
 * @brief math helper, floor [float3] to [int3]
 *
 * @param[in] v 3d-vector of float
 *
 * @return    3d-vector of int
 */
    __device__ inline int3
    floor_to_int3(const float3 &v) {
        return {static_cast<int>(floor(v.x)),
                static_cast<int>(floor(v.y)),
                static_cast<int>(floor(v.z))};
    }

/**
 * @brief compute cell pos by particle pos
 *
 * @param[in] pos particle pos
 * @param[in] sceneLB left-bottom of the scene
 * @param[in] cellLength cell length
 *
 * @return    cell pos, 3d-vector of int
 */
    __device__ inline int3
    getCellPos(float3 &pos, float3 &sceneLB, float cellLength) {
        int3 cellPos = floor_to_int3((pos - sceneLB) / cellLength);
        return cellPos;
    }

/**
 * @brief compute cell id by cell pos
 *
 * @param[in] cellPos cell pos
 * @param[in] gridSize size of background grid
 *
 * @return    cell id, uint32_t
 */
    __device__ inline unsigned
    getCellId(const int3 &cellPos, const uint3 &gridSize) {
        unsigned cellId =
                cellPos.z * (gridSize.y * gridSize.x) + cellPos.y * gridSize.x + cellPos.x;
        return cellId;
    }

/**
 * @brief check if cur cell is available
 *
 * @param[in] cellPos cell pos
 * @param[in] gridSize size of background grid
 *
 * @return    true if available, false otherwise
 */
    __device__ inline bool
    cellIsAvailable(const int3 &cellPos, const uint3 &gridSize) {
        int3 cellStartPos = {0, 0, 0};
        int3 cellEndPos = {static_cast<int>(gridSize.x), static_cast<int>(gridSize.y),
                           static_cast<int>(gridSize.z)};

        return (cellPos.x >= cellStartPos.x && cellPos.y >= cellStartPos.y && cellPos.z >= cellStartPos.z &&
                cellPos.x <= cellEndPos.x && cellPos.y <= cellEndPos.y && cellPos.z <= cellEndPos.z);
    }

/**
 * @brief check if cur cell is activated
 *
 * @param[in] cellId cell id
 * @param[in] cellStart device pointer of the cell start array
 *
 * @return    true if cell is not empty, false otherwise
 */
    __device__ inline bool
    cellIsActivated(const unsigned cellId, const unsigned *cellStart) {
        return (cellStart[cellId] != UINT_MAX);
    }

    extern __host__ void
    resetDevPtr(UniformGirdNeighborSearcherConfig &h_config,
                UniformGirdNeighborSearcherParams &h_params,
                unsigned gridCellNum) {
        static size_t size1 = gridCellNum;
        static size_t size2 = h_config.totalParticleNum;
        static size_t size3 = h_config.totalParticleNum * h_config.maxNeighborNum;

        cudaMemset(h_params.cellStart_cuData, UINT_MAX, size1 * sizeof(uint32_t));
        cudaMemset(h_params.cellEnd_cuData, UINT_MAX, size1 * sizeof(uint32_t));
        cudaMemset(h_params.neighborNum_cuData, 0, size2 * sizeof(uint32_t));
        cudaMemset(h_params.neighbors_cuData, UINT_MAX, size3 * sizeof(uint32_t));
    }

    extern __global__ void
    calcParticleHashValue(UniformGirdNeighborSearcherConfig *d_config,
                          UniformGirdNeighborSearcherParams *d_params,
                          uint3 gridSize,
                          float3 *pos) {
        THREAD_CHECK()

        auto cellPos = getCellPos(pos[i], dCONFIG(simSpaceLB), dCONFIG(gridCellSize));
        if (cellIsAvailable(cellPos, gridSize)) {
            uint32_t cellId = getCellId(cellPos, gridSize);
            dPARAMS(cellIndices_cuData, i) = cellId;
            dPARAMS(particleIndices_cuData, i) = i;
            dPARAMS(particleIsAlive_cuData, i) = true;
        } else {
            dPARAMS(particleIsAlive_cuData, i) = false;
        }

    }

    extern __host__ void
    sortByHashValue(UniformGirdNeighborSearcherConfig &h_config, UniformGirdNeighborSearcherParams &h_params) {
        thrust::device_ptr<uint32_t> keys_dev_ptr(h_params.cellIndices_cuData);
        thrust::device_ptr<uint32_t> values_dev_ptr_pInd(h_params.particleIndices_cuData);
        thrust::device_ptr<bool> values_dev_ptr_pAlive(h_params.particleIsAlive_cuData);

        // use thrust::sort_by_key to order by key
        thrust::sort_by_key(keys_dev_ptr,
                            keys_dev_ptr + h_config.totalParticleNum,
                            values_dev_ptr_pInd);

        thrust::sort_by_key(keys_dev_ptr,
                            keys_dev_ptr + h_config.totalParticleNum,
                            values_dev_ptr_pAlive);
    }

    extern __global__ void
    findCellRange(UniformGirdNeighborSearcherConfig *d_config,
                  UniformGirdNeighborSearcherParams *d_params) {
        THREAD_CHECK()

        uint32_t curCellId = dPARAMS(cellIndices_cuData, i);
        if (i == 0)
            dPARAMS(cellStart_cuData, curCellId) = 0;
        else {
            unsigned pre_i = i - 1;
            uint32_t preCellId = d_params->cellIndices_cuData[pre_i];
            if (curCellId != preCellId) {
                dPARAMS(cellStart_cuData, curCellId) = i;
                dPARAMS(cellEnd_cuData, preCellId) = pre_i;

                if (dPARAMS(cellIndices_cuData, curCellId) == UINT_MAX)
                    dCONFIG(aliveParticleNum) = i;
            }

            if (i == dCONFIG(totalParticleNum) - 1)
                dPARAMS(cellEnd_cuData, curCellId) = i;
        }
    }

    extern __global__ void
    findNeighbors(UniformGirdNeighborSearcherConfig *d_config,
                  UniformGirdNeighborSearcherParams *d_params,
                  uint3 gridSize,
                  float3 *pos) {
        THREAD_CHECK()

        auto p_i = dPARAMS(particleIndices_cuData, i);
        if (!dPARAMS(particleIsAlive_cuData, p_i))
            return;

        auto pos_i = pos[p_i];
        auto pn_index = p_i * dCONFIG(maxNeighborNum);
        dPARAMS(neighborNum_cuData, p_i) = 1;
        dPARAMS(neighbors_cuData, pn_index) = p_i;
        int3 curCellPos = getCellPos(pos[p_i], dCONFIG(simSpaceLB), dCONFIG(gridCellSize));
        for (int t = 0; t < 27; ++t) {
            auto offset = dPARAMS(cellOffsets_cuData, t);
            int3 cellPos = curCellPos + offset;
            auto cellId = getCellId(cellPos, gridSize);
            if (cellIsAvailable(cellPos, gridSize) && cellIsActivated(cellId, d_params->cellStart_cuData)) {
                for (unsigned j = dPARAMS(cellStart_cuData, cellId); j <= dPARAMS(cellEnd_cuData, cellId); ++j) {
                    auto p_j = dPARAMS(particleIndices_cuData, j);
                    if (p_j == p_i || !dPARAMS(particleIsAlive_cuData, p_j))
                        continue;
                    auto pos_j = pos[p_j];
                    if (length((pos_i - pos_j)) > 0 && length((pos_i - pos_j)) <= dCONFIG(gridCellSize)) {
                        if (dPARAMS(neighborNum_cuData, p_i) < dCONFIG(maxNeighborNum)) {
                            auto ind_offset = dPARAMS(neighborNum_cuData, p_i)++;
                            dPARAMS(neighbors_cuData, pn_index + ind_offset) = p_j;
                        }
                    }
                }
            }
        }
    }

    extern __host__ void
    cu_update(UniformGirdNeighborSearcherConfig &h_config,
              UniformGirdNeighborSearcherParams &h_params,
              UniformGirdNeighborSearcherConfig *d_config,
              UniformGirdNeighborSearcherParams *d_params,
              uint3 gridSize,
              unsigned gridCellNum,
              float3 *pos) {
        const auto &block_num = h_config.cuKernelBlockNum;
        const auto &thread_num = h_config.cuKernelThreadNum;

        resetDevPtr(h_config, h_params, gridCellNum);

        calcParticleHashValue<<<block_num, thread_num>>>(d_config, d_params, gridSize, pos);

        sortByHashValue(h_config, h_params);

        findCellRange<<<block_num, thread_num>>>(d_config, d_params);

        findNeighbors<<<block_num, thread_num>>>(d_config, d_params, gridSize, pos);

        cudaGetLastError();
    }

}