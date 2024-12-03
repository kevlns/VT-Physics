/**
 * @brief Todo
 * @date 2024/11/7
 */

#ifndef VT_PHYSICS_MCTCUDAAPI_CUH
#define VT_PHYSICS_MCTCUDAAPI_CUH

#include "Solvers/MCT/MCTrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"
#include "Core/Math/helper_math_cu11.6.h"

namespace VT_Physics::mct {

    __host__ void
    init_data(Data *h_data,
              Data *d_data);

    __host__ void
    prepare_mct(Data *h_data,
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
                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    mct_gravity_surface(Data *h_data,
                        Data *d_data,
                        UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    mpct(Data *h_data,
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
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    phase_transfer(Data *h_data,
                   Data *d_data,
                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams,
                   bool &crash);

    __host__ void
    update_mass_and_vel(Data *h_data,
                        Data *d_data,
                        UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    apply_porous_medium(Data *h_data,
                        Data *d_data,
                        UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    apply_adhesion_force(Data *h_data,
                         Data *d_data,
                         UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    digging(Data *h_data,
            Data *d_data,
            UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
            UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    update_color(Data *h_data,
                 Data *d_data,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    stir_fan(Data *h_data,
             Data *d_data,
             UGNS::UniformGirdNeighborSearcherParams *d_nsParams);
}

#endif //VT_PHYSICS_MCTCUDAAPI_CUH
