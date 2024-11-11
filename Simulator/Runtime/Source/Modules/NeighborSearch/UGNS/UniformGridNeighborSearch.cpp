#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"

#include <sstream>
#include <iomanip>

#include "UGNSCudaApi.cuh"
#include "Logger/Logger.hpp"

namespace VT_Physics::UGNS {

    void UniformGirdNeighborSearcherParams::malloc(const unsigned &total_pNum,
                                                   const unsigned &max_nNum,
                                                   const unsigned &cellNum) {
        // reset
        freeMemory();

        // init neighbor search
        auto size1 = total_pNum * sizeof(unsigned);
        auto size1_1 = total_pNum * sizeof(bool);
        auto size2 = cellNum * sizeof(unsigned);
        auto size3 = total_pNum * max_nNum * sizeof(unsigned);
        auto size4 = 27 * sizeof(int3);
        cudaMalloc((void **) &particleIndices_cuData, size1);
        cudaMalloc((void **) &cellIndices_cuData, size1);
        cudaMalloc((void **) &particleHandle_cuData, size1);
        cudaMalloc((void **) &particleIsAlive_cuData, size1_1);
        cudaMalloc((void **) &neighborNum_cuData, size1);
        cudaMalloc((void **) &cellStart_cuData, size2);
        cudaMalloc((void **) &cellEnd_cuData, size2);
        cudaMalloc((void **) &neighbors_cuData, size3);
        cudaMalloc((void **) &cellOffsets_cuData, size4);

        std::vector<int3> offsets = {
                {-1, -1, -1},
                {-1, -1, 0},
                {-1, -1, 1},
                {-1, 0,  -1},
                {-1, 0,  0},
                {-1, 0,  1},
                {-1, 1,  -1},
                {-1, 1,  0},
                {-1, 1,  1},
                {0,  -1, -1},
                {0,  -1, 0},
                {0,  -1, 1},
                {0,  0,  -1},
                {0,  0,  0},
                {0,  0,  1},
                {0,  1,  -1},
                {0,  1,  0},
                {0,  1,  1},
                {1,  -1, -1},
                {1,  -1, 0},
                {1,  -1, 1},
                {1,  0,  -1},
                {1,  0,  0},
                {1,  0,  1},
                {1,  1,  -1},
                {1,  1,  0},
                {1,  1,  1}};
        cudaMemcpy(cellOffsets_cuData, offsets.data(), size4, cudaMemcpyHostToDevice);

        isInit = true;
        memUsed = static_cast<double>((size1 * 4 + size1_1 + size2 * 2 + size3 + size4) >> 20);
    }

    void UniformGirdNeighborSearcherParams::freeMemory() {
        if (isInit) {
            cudaFree(cellOffsets_cuData);
            cudaFree(particleIndices_cuData);
            cudaFree(cellIndices_cuData);
            cudaFree(particleHandle_cuData);
            cudaFree(cellStart_cuData);
            cudaFree(cellEnd_cuData);
            cudaFree(neighborNum_cuData);
            cudaFree(neighbors_cuData);
            cudaFree(particleIsAlive_cuData);

            if (cudaGetLastError() == cudaSuccess) {
                isInit = false;
                LOG_INFO("UniformGridNeighborSearcher Destroyed.");
            }

        }
    }

    double UniformGirdNeighborSearcherParams::getMemoryCount() const {
        return memUsed;
    }

    void UniformGirdNeighborSearcher::setConfig(UniformGirdNeighborSearcherConfig config) {
        m_config_hostData = config;
        m_config_hostData.aliveParticleNum = m_config_hostData.totalParticleNum;
        m_gridSize = {
                static_cast<uint32_t>(std::ceil(m_config_hostData.simSpaceSize.x / m_config_hostData.gridCellSize)),
                static_cast<uint32_t>(std::ceil(m_config_hostData.simSpaceSize.y / m_config_hostData.gridCellSize)),
                static_cast<uint32_t>(std::ceil(m_config_hostData.simSpaceSize.z / m_config_hostData.gridCellSize))};
        m_gridCellNum = m_gridSize.x * m_gridSize.y * m_gridSize.z;
        if (m_config_cuData)
            cudaFree(m_config_cuData);

        cudaMalloc((void **) &m_config_cuData, sizeof(UniformGirdNeighborSearcherConfig));
        cudaMemcpy(m_config_cuData, &m_config_hostData, sizeof(UniformGirdNeighborSearcherConfig),
                   cudaMemcpyHostToDevice);
    }

    bool UniformGirdNeighborSearcher::malloc() {
        if (m_params_cuData)
            cudaFree(m_params_cuData);

        m_params_hostData.malloc(m_config_hostData.totalParticleNum,
                                 m_config_hostData.maxNeighborNum,
                                 m_gridCellNum);
        cudaMalloc((void **) &m_params_cuData, sizeof(UniformGirdNeighborSearcherParams));
        cudaMemcpy(m_params_cuData,
                   &m_params_hostData,
                   sizeof(UniformGirdNeighborSearcherParams),
                   cudaMemcpyHostToDevice);

        bool *alive = new bool[m_config_hostData.totalParticleNum];
        memset(alive, true, m_config_hostData.totalParticleNum);
        cudaMemcpy(m_params_hostData.particleIsAlive_cuData,
                   alive,
                   m_config_hostData.totalParticleNum * sizeof(bool),
                   cudaMemcpyHostToDevice);
        delete[] alive;

        if (cudaGetLastError() != cudaSuccess) {
            LOG_INFO("UniformGridNeighborSearcher fails to initialize.");
            return false;
        }
        return true;
    }

    void UniformGirdNeighborSearcher::update(float3 *pos_cuData) {
        cu_update(m_config_hostData,
                  m_params_hostData,
                  m_config_cuData,
                  m_params_cuData,
                  m_gridSize,
                  m_gridCellNum,
                  pos_cuData);
    }

    void UniformGirdNeighborSearcher::dump() const {
        auto cellNum = m_gridCellNum;
        auto particleNum = m_config_hostData.totalParticleNum;
        auto *c_cellStart = new uint32_t[cellNum];
        auto *c_cellEnd = new uint32_t[cellNum];
        cudaMemcpy(c_cellStart, m_params_hostData.cellStart_cuData, cellNum * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(c_cellEnd, m_params_hostData.cellEnd_cuData, cellNum * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t cnt = 0;
        for (unsigned i = 0; i < cellNum; ++i) {
            if (c_cellStart[i] != UINT_MAX) {
                cnt++;
            }
        }
        delete[] c_cellStart;
        delete[] c_cellEnd;

        LOG_INFO("Dump UniformGridNeighborSearcher Info:\n");
        std::stringstream ss;
        ss << '\n' << std::setw(35) << "Allocated Memory: " << std::setw(20) <<
           std::to_string(m_params_hostData.getMemoryCount()) + "MB" << "\n"
           << std::setw(35) << "Particle Num: " << std::setw(20) << m_config_hostData.totalParticleNum << "\n"
           << std::setw(35) << "Cell Num: " << std::setw(20) << cellNum << "\n"
           << std::setw(35) << "Grid Size: " << std::setw(20)
           << std::to_string(m_gridSize.x) + " * " + std::to_string(m_gridSize.y)
              + " * " + std::to_string(m_gridSize.z) << "\n"
           << std::setw(35) << "Average PartNum Per Cell: " << std::setw(20) << (particleNum / cnt) << "\n"
           << std::setw(35) << "Active Cell num: " << std::setw(20) << cnt << "\n";
        LOG_WARNING(ss.str());
    }

    void UniformGirdNeighborSearcher::freeMemory() {
        if (m_config_cuData)
            cudaFree(m_config_cuData);

        m_params_hostData.freeMemory();
        if (m_params_cuData)
            cudaFree(m_params_cuData);

        cudaGetLastError();
    }

    json UniformGirdNeighborSearcher::getUGNSConfigTemplate() {
        return JsonHandler::loadUGNSConfigTemplateJson();
    }

    bool UniformGirdNeighborSearcher::setConfig(json config) {
        for (const auto &key: UGNSConfigRequiredKeys) {
            if (!config.contains(key)) {
                LOG_ERROR("UniformGirdNeighborSearcher Config Missing required key: " + key);
                return false;
            }
        }

        m_config_hostData.simSpaceLB = {
                config["simSpaceLB"].get<std::vector<float>>()[0],
                config["simSpaceLB"].get<std::vector<float>>()[1],
                config["simSpaceLB"].get<std::vector<float>>()[2]
        };
        m_config_hostData.simSpaceSize = {
                config["simSpaceSize"].get<std::vector<float>>()[0],
                config["simSpaceSize"].get<std::vector<float>>()[1],
                config["simSpaceSize"].get<std::vector<float>>()[2]
        };
        m_config_hostData.totalParticleNum = config["totalParticleNum"].get<unsigned>();
        m_config_hostData.aliveParticleNum = m_config_hostData.totalParticleNum;
        m_config_hostData.maxNeighborNum = config["maxNeighborNum"].get<unsigned>();
        m_config_hostData.cuKernelBlockNum = config["cuKernelBlockNum"].get<unsigned>();
        m_config_hostData.cuKernelThreadNum = config["cuKernelThreadNum"].get<unsigned>();
        m_config_hostData.gridCellSize = config["gridCellSize"].get<float>();
        m_gridSize = {
                static_cast<uint32_t>(std::ceil(m_config_hostData.simSpaceSize.x / m_config_hostData.gridCellSize)),
                static_cast<uint32_t>(std::ceil(m_config_hostData.simSpaceSize.y / m_config_hostData.gridCellSize)),
                static_cast<uint32_t>(std::ceil(m_config_hostData.simSpaceSize.z / m_config_hostData.gridCellSize))};
        m_gridCellNum = m_gridSize.x * m_gridSize.y * m_gridSize.z;

        if (m_config_cuData)
            cudaFree(m_config_cuData);

        cudaMalloc((void **) &m_config_cuData, sizeof(UniformGirdNeighborSearcherConfig));
        cudaMemcpy(m_config_cuData, &m_config_hostData, sizeof(UniformGirdNeighborSearcherConfig),
                   cudaMemcpyHostToDevice);

        return true;
    }

    bool UniformGirdNeighborSearcher::setConfigByFile(std::string config_file) {
        auto config = JsonHandler::loadJson(config_file);
        return setConfig(config);
    }

}