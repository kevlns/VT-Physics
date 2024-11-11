/**
 * @brief Todo
 * @date 2024/11/7
 */

#ifndef VT_PHYSICS_PBFCUDAAPI_CUH
#define VT_PHYSICS_PBFCUDAAPI_CUH

#include "Solvers/PBF/PBFrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"
#include "Core/Math/helper_math_cu11.6.h"

namespace VT_Physics::pbf {

    __host__ void
    init_data(Data *h_data,
              Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    compute_sph_density_and_error(Data *h_data,
                                  Data *d_data,
                                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    update_lamb(Data *h_data,
                Data *d_data,
                UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    compute_dx(Data *h_data,
               Data *d_data,
               UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    apply_ext_force(Data *h_data,
                    Data *d_data,
                    UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                    UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    apply_dx(Data *h_data,
             Data *d_data,
             UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
             UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    post_correct(Data *h_data,
                 Data *d_data,
                 UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

}

#endif //VT_PHYSICS_PBFCUDAAPI_CUH
