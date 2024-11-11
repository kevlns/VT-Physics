/**
 * @brief Todo
 * @date 2024/11/7
 */

#ifndef VT_PHYSICS_PBFCUDAAPI_CUH
#define VT_PHYSICS_PBFCUDAAPI_CUH

#include "Solvers/DFSPH/DFSPHrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"
#include "Core/Math/helper_math_cu11.6.h"

namespace VT_Physics::dfsph {

    __host__ void
    init_data(Data *h_data,
              Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    prepare_dfsph(Data *h_data,
                  Data *d_data,
                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    sph_precompute(Data *h_data,
                   Data *d_data,
                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    vfsph_div(Data *h_data,
              Data *d_data,
              const std::vector<int> &obj_start_index,
              const std::vector<int> &obj_end_index,
              const std::vector<int> &obj_mats,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams,
              bool &crash);

    __host__ void
    apply_pressure_acc(Data *h_data,
                       Data *d_data,
                       UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    dfsph_gravity_vis_surface(Data *h_data,
                              Data *d_data,
                              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                              UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    vfsph_incomp(Data *h_data,
                 Data *d_data,
                 const std::vector<int> &obj_start_index,
                 const std::vector<int> &obj_end_index,
                 const std::vector<int> &obj_mats,
                 UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams,
                 bool &crash);

    __host__ void
    update_pos(Data *h_data,
               Data *d_data,
               UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

}

#endif //VT_PHYSICS_PBFCUDAAPI_CUH
