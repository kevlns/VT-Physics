/**
 * @brief Todo
 * @date 2024/11/7
 */

#ifndef VT_PHYSICS_UGNSCUDAAPI_CUH
#define VT_PHYSICS_UGNSCUDAAPI_CUH

#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"

namespace VT_Physics::UGNS {

    extern __host__ void
    cu_update(UniformGirdNeighborSearcherConfig &h_config,
              UniformGirdNeighborSearcherParams &h_params,
              UniformGirdNeighborSearcherConfig *d_config,
              UniformGirdNeighborSearcherParams *d_params,
              uint3 gridSize,
              unsigned gridCellNum,
              float3 *pos);

}

#endif //VT_PHYSICS_UGNSCUDAAPI_CUH
