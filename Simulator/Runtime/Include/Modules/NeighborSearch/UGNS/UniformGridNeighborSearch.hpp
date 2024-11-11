/**
 * @brief Todo
 * @date 2024/11/7
 */

#ifndef VT_PHYSICS_UNIFORMGRIDNEIGHBORSEARCH_HPP
#define VT_PHYSICS_UNIFORMGRIDNEIGHBORSEARCH_HPP

#include <vector>

#include "Core/Math/helper_math_cu11.6.h"
#include "JSON/JSONHandler.hpp"

namespace VT_Physics::UGNS {

    class UniformGirdNeighborSearcher;

    inline const std::vector<std::string> UGNSConfigRequiredKeys = {
            "simSpaceLB",
            "simSpaceSize",
            "totalParticleNum",
            "maxNeighborNum",
            "cuKernelBlockNum",
            "cuKernelThreadNum",
            "gridCellSize"
    };

    struct UniformGirdNeighborSearcherConfig {
        float3 simSpaceLB;
        float3 simSpaceSize;
        unsigned totalParticleNum{0};
        unsigned aliveParticleNum{0};
        unsigned maxNeighborNum{0};
        unsigned cuKernelBlockNum{100};
        unsigned cuKernelThreadNum{8};
        float gridCellSize{0}; // uniform on all axis
    };

    struct UniformGirdNeighborSearcherParams {
        friend class UniformGirdNeighborSearcher;

    public:
        int3 *cellOffsets_cuData{nullptr};         // cell offsets: 3D[27 cells]; 2D[9 cells]
        unsigned *particleIndices_cuData{nullptr};  // particle index after sorted
        unsigned *cellIndices_cuData{nullptr};      // cell index after sorted
        unsigned *cellStart_cuData{nullptr};        // each cell start index
        unsigned *cellEnd_cuData{nullptr};          // each cell end index
        unsigned *neighborNum_cuData{nullptr};      // neighbor num of each particle
        unsigned *neighbors_cuData{nullptr};        // neighbor index of each particle
        unsigned *particleHandle_cuData{nullptr};   // the macro-object ID of each particle
        bool *particleIsAlive_cuData{nullptr};      // particle alive flag

    private:
        bool isInit{false};
        double memUsed{0};

    private:
        void malloc(const unsigned &total_pNum, const unsigned &max_nNum, const unsigned &cellNum);

        void freeMemory();

        double getMemoryCount() const;
    };

    class UniformGirdNeighborSearcher {
    public:
        void setConfig(UniformGirdNeighborSearcherConfig config);

        json getUGNSConfigTemplate();

        bool setConfig(json config);

        bool setConfigByFile(std::string config_file);

        void update(float3 *pos_cuData);

        void dump() const;

        bool malloc();

        void freeMemory();

    public:
        UniformGirdNeighborSearcherConfig m_config_hostData;
        UniformGirdNeighborSearcherConfig *m_config_cuData{nullptr};
        UniformGirdNeighborSearcherParams m_params_hostData;
        UniformGirdNeighborSearcherParams *m_params_cuData{nullptr};

    private:
        uint3 m_gridSize;
        unsigned m_gridCellNum;
    };

}

#endif //VT_PHYSICS_UNIFORMGRIDNEIGHBORSEARCH_HPP
