#include "MCTCudaApi.cuh"

#include "Modules/SPH/KernelFunctions.cuh"
#include "DataAnalysis/CommonFunc.hpp"
#include "Core/Math/DataStructTransfer.hpp"

namespace VT_Physics::mct {

#define CHECK_THREAD() \
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= d_data->particle_num)                     \
        return;                                         \
    auto p_i = d_nsParams->particleIndices_cuData[i];

#define DIGGING_CHECK_THREAD() \
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= d_data->digging_particle_num)                     \
        return;                                         \
    auto p_i = d_nsParams->particleIndices_cuData[i];

#define FOR_EACH_NEIGHBOR_Pj() \
       auto neib_ind = p_i * d_nsConfig->maxNeighborNum;                        \
       for (unsigned int p_j = d_nsParams->neighbors_cuData[neib_ind], t = 0;   \
            p_j != UINT_MAX && t < d_nsConfig->maxNeighborNum;                  \
            ++t, p_j = d_nsParams->neighbors_cuData[neib_ind + t])

#define FOR_EACH_NEIGHBOR_Pj_2pass() \
       for (unsigned int p_j = d_nsParams->neighbors_cuData[neib_ind], t = 0;   \
            p_j != UINT_MAX && t < d_nsConfig->maxNeighborNum;                  \
            ++t, p_j = d_nsParams->neighbors_cuData[neib_ind + t])

#define CONST_VALUE(name) \
        d_data->name

#define DATA_VALUE(name, index) \
        d_data->name[index]

#define CUBIC_KERNEL_VALUE() \
        sph::cubic_value(pos_i - pos_j, d_data->h)

#define CUBIC_KERNEL_GRAD() \
        sph::cubic_gradient(pos_i - pos_j, d_data->h)

#define SURFACE_TENSION_VALUE() \
        sph::surface_tension_C(length(pos_i - pos_j), d_data->h)

#define DATA_VALUE_PHASE(name, index, phase_index) \
        d_data->name[(index) * d_data->phase_num + (phase_index)]

#define CONST_VALUE_PHASE(name, phase_index) \
        d_data->name[phase_index]

#define FOR_EACH_PHASE_k() \
        for(int k = 0; k < d_data->phase_num; ++k)

}

/**
 * cuda impl
 */
namespace VT_Physics::mct {

    __global__ void
    init_data_cuda(Data *d_data) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= d_data->particle_num)
            return;

        DATA_VALUE(volume, i) = CONST_VALUE(fPart_rest_volume);
        DATA_VALUE(kappa_div, i) = 0;
        DATA_VALUE(acc, i) *= 0;
        DATA_VALUE(vel_adv, i) = DATA_VALUE(vel, i);
        DATA_VALUE(rest_density, i) = 0;
        DATA_VALUE(flag_negative_vol_frac, i) = 0;
        DATA_VALUE(rest_volume_rate, i) = CONST_VALUE(fPart_rest_volume);
        DATA_VALUE(is_in_porous, i) = false;
        DATA_VALUE(pressure_pore, i) = CONST_VALUE(rest_pressure_pore);
        DATA_VALUE(rigid_raw_volume, i) = CONST_VALUE(fPart_rest_volume);

        DATA_VALUE(porosity, i) = 0;
        if (DATA_VALUE(mat, i) == EPM_POROUS)
            DATA_VALUE(porosity, i) = CONST_VALUE(porous_porosity);

        FOR_EACH_PHASE_k() {
            if (i == 0) {
                CONST_VALUE_PHASE(phase_rest_vis, k) *= 1000;
            }

            DATA_VALUE_PHASE(Q, i, k) = Mat33f::eye();
            DATA_VALUE_PHASE(vol_frac_in, i, k) = 0;
            DATA_VALUE_PHASE(vol_frac_out, i, k) = 0;
            DATA_VALUE_PHASE(vel_phase, i, k) = DATA_VALUE(vel, i);
            DATA_VALUE_PHASE(acc_phase, i, k) *= 0;
            DATA_VALUE_PHASE(vel_drift_phase, i, k) *= 0;
            DATA_VALUE(rest_density, i) += DATA_VALUE_PHASE(vol_frac, i, k) * CONST_VALUE_PHASE(phase_rest_density, k);
        }
    }

    __global__ void
    update_rest_density_and_mass_cuda(Data *d_data,
                                      UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                      UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        DATA_VALUE(rest_density, p_i) *= 0;
        FOR_EACH_PHASE_k() {
            DATA_VALUE(rest_density, p_i) += DATA_VALUE_PHASE(vol_frac, p_i, k) *
                                             CONST_VALUE_PHASE(phase_rest_density, k);
        }

        __syncthreads();
        if (DATA_VALUE(mat, p_i) != EPM_FLUID) {
            FOR_EACH_NEIGHBOR_Pj() {
                if (DATA_VALUE(mat, p_j) == EPM_FLUID &&
                    DATA_VALUE(rest_density, p_i) < DATA_VALUE(rest_density, p_j)) {
                    DATA_VALUE(rest_density, p_i) = DATA_VALUE(rest_density, p_j);
                }
            }
        }

        DATA_VALUE(mass, p_i) = DATA_VALUE(rest_density, p_i) * DATA_VALUE(volume, p_i);
    }

    __global__ void
    compute_compression_ratio_cuda(Data *d_data,
                                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        DATA_VALUE(compression_ratio, p_i) = 1;
        if (DATA_VALUE(mat, p_i) != EPM_FLUID || DATA_VALUE(is_in_porous, p_i))
            return;

        DATA_VALUE(compression_ratio, p_i) *= 0;

        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
//            if (DATA_VALUE(mat, p_j) == Emitter_Particle)
//                continue;

            auto pos_j = DATA_VALUE(pos, p_j);

            DATA_VALUE(compression_ratio, p_i) += DATA_VALUE(volume, p_j) * CUBIC_KERNEL_VALUE();
        }
    }

    __global__ void
    compute_df_beta_cuda(Data *d_data,
                         UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        DATA_VALUE(df_alpha_1, p_i) *= 0;
        DATA_VALUE(df_alpha_2, p_i) = 1e-6;

        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
//            if (p_j == p_i || DATA_VALUE(mat, p_j) == Emitter_Particle)
//                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            // applied to all dynamic objects
            if (DATA_VALUE(mat, p_i) == EPM_FLUID)
                DATA_VALUE(df_alpha_1, p_i) += DATA_VALUE(volume, p_j) * CUBIC_KERNEL_GRAD();

            // applied to all dynamic objects
            if (DATA_VALUE(mat, p_j) == EPM_FLUID)
                DATA_VALUE(df_alpha_2, p_i) += dot(wGrad, wGrad) * DATA_VALUE(volume, p_j) * DATA_VALUE(volume, p_j)
                                               / DATA_VALUE(mass, p_j);
        }

        DATA_VALUE(df_alpha, p_i) =
                dot(DATA_VALUE(df_alpha_1, p_i), DATA_VALUE(df_alpha_1, p_i)) / DATA_VALUE(mass, p_i)
                + DATA_VALUE(df_alpha_2, p_i);

        if (DATA_VALUE(df_alpha, p_i) < 1e-6)
            DATA_VALUE(df_alpha, p_i) = 1e-6;
    }

    __global__ void
    compute_delta_compression_ratio_cuda(Data *d_data,
                                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(delta_compression_ratio, p_i) = DATA_VALUE(compression_ratio, p_i) - 1.f;
    }

    __global__ void
    update_delta_compression_ratio_from_vel_adv_cuda(Data *d_data,
                                                     UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                                     UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        auto vel_adv_i = DATA_VALUE(vel_adv, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
//            if (p_j == p_i || DATA_VALUE(mat, p_j) == Emitter_Particle)
//                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto vel_adv_j = DATA_VALUE(vel_adv, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            DATA_VALUE(delta_compression_ratio, p_i) += dot(wGrad, vel_adv_i - vel_adv_j) *
                                                        DATA_VALUE(volume, p_j) * CONST_VALUE(dt);
        }

        if (DATA_VALUE(delta_compression_ratio, p_i) < 0)
            DATA_VALUE(delta_compression_ratio, p_i) = 0;
    }

    __global__ void
    compute_kappa_div_from_delta_compression_ratio_cuda(Data *d_data,
                                                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        DATA_VALUE(kappa_div, p_i) *= 0;

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(kappa_div, p_i) = DATA_VALUE(delta_compression_ratio, p_i) / DATA_VALUE(df_alpha, p_i) *
                                     CONST_VALUE(inv_dt2) / DATA_VALUE(volume, p_i);
        DATA_VALUE(df_alpha_2, p_i) += DATA_VALUE(kappa_div, p_i);
    }

    __global__ void
    vf_update_vel_adv_from_kappa_div_cuda(Data *d_data,
                                          UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                          UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
//            if (p_j == p_i || DATA_VALUE(mat, p_j) == Emitter_Particle)
//                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            DATA_VALUE(vel_adv, p_i) -= CONST_VALUE(dt) * DATA_VALUE(volume, p_i) * DATA_VALUE(volume, p_j) /
                                        DATA_VALUE(mass, p_i) *
                                        (DATA_VALUE(kappa_div, p_i) + DATA_VALUE(kappa_div, p_j)) * wGrad;
        }
    }

    __global__ void
    get_acc_pressure_cuda(Data *d_data,
                          UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        DATA_VALUE(acc, p_i) = (DATA_VALUE(vel_adv, p_i) - DATA_VALUE(vel, p_i)) * CONST_VALUE(inv_dt);
    }

    __global__ void
    clear_phase_acc_cuda(Data *d_data,
                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(acc_phase, p_i, k) *= 0;
        }
    }

    __global__ void
    add_phase_acc_gravity_cuda(Data *d_data,
                               UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(acc_phase, p_i, k) += CONST_VALUE(gravity);
        }
    }

    __global__ void
    compute_surface_normal_cuda(Data *d_data,
                                UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        float3 normal = {0, 0, 0};
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != DATA_VALUE(mat, p_i))
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);

            normal += CONST_VALUE(h) * DATA_VALUE(mass, p_j) / DATA_VALUE(rest_density, p_j) *
                      CUBIC_KERNEL_GRAD();
        }

        DATA_VALUE(surface_normal, p_i) = normal;
    }

    __global__ void
    add_phase_acc_surface_tension_cuda(Data *d_data,
                                       UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        float3 acc = {0, 0, 0};
        float gamma = CONST_VALUE(surface_tension_coefficient);
        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != DATA_VALUE(mat, p_i) || p_j == p_i)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto k =
                    2 * DATA_VALUE(rest_density, p_i) / (DATA_VALUE(rest_density, p_i) + DATA_VALUE(rest_density, p_j));

            auto acc_1 = -gamma * DATA_VALUE(mass, p_i) * DATA_VALUE(mass, p_j) *
                         SURFACE_TENSION_VALUE() * (pos_i - pos_j) /
                         length(pos_i - pos_j);
            auto acc_2 = -gamma * DATA_VALUE(mass, p_i) *
                         (DATA_VALUE(surface_normal, p_i) - DATA_VALUE(surface_normal, p_j));

            acc += k * (acc_1 + acc_2);
        }

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(acc_phase, p_i, k) += acc;
        }
    }

    __global__ void
    phase_acc_2_phase_vel_cuda(Data *d_data,
                               UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(vel_phase, p_i, k) += DATA_VALUE_PHASE(acc_phase, p_i, k) * CONST_VALUE(dt);
        }
    }

    __global__ void
    update_vel_from_phase_vel_cuda(Data *d_data,
                                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(vel, p_i) *= 0;
        FOR_EACH_PHASE_k() {
            DATA_VALUE(vel, p_i) += DATA_VALUE_PHASE(vel_phase, p_i, k) * DATA_VALUE_PHASE(vol_frac, p_i, k);
        }

        DATA_VALUE(vel_adv, p_i) = DATA_VALUE(vel, p_i);

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(vel_drift_phase, p_i, k) =
                    DATA_VALUE_PHASE(vel_phase, p_i, k) - DATA_VALUE(vel, p_i);
        }
    }

    __global__ void
    distribute_acc_pressure_2_phase_cuda(Data *d_data,
                                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(acc_phase, p_i, k) +=
                    DATA_VALUE(acc, p_i) * (CONST_VALUE(Cd) + (1 - CONST_VALUE(Cd)) *
                                                              (DATA_VALUE(rest_density, p_i) /
                                                               CONST_VALUE_PHASE(phase_rest_density, k)));
        }
    }

    __global__ void
    compute_kappa_incomp_from_delta_compression_ratio_cuda(Data *d_data,
                                                           UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(kappa_incomp, p_i) = DATA_VALUE(delta_compression_ratio, p_i) / DATA_VALUE(df_alpha, p_i) *
                                        CONST_VALUE(inv_dt2) / DATA_VALUE(volume, p_i);
        DATA_VALUE(df_alpha_2, p_i) += DATA_VALUE(kappa_incomp, p_i);
    }

    __global__ void
    vf_update_vel_adv_from_kappa_incomp_cuda(Data *d_data,
                                             UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                             UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
//            if (p_j == p_i || DATA_VALUE(mat, p_j) == Emitter_Particle)
//                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            DATA_VALUE(vel_adv, p_i) -= CONST_VALUE(dt) * DATA_VALUE(volume, p_i) * DATA_VALUE(volume, p_j) /
                                        DATA_VALUE(mass, p_i) *
                                        (DATA_VALUE(kappa_incomp, p_i) + DATA_VALUE(kappa_incomp, p_j)) * wGrad;
        }
    }

    __global__ void
    update_pos_cuda(Data *d_data,
                    UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(pos, p_i) += DATA_VALUE(vel, p_i) * CONST_VALUE(dt);
        DATA_VALUE(pos_adv, p_i) = DATA_VALUE(pos, p_i);
    }

    __global__ void
    regularize_vol_frac_cuda(Data *d_data,
                             UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        float frac_sum = 0;
        FOR_EACH_PHASE_k() {
            frac_sum += DATA_VALUE_PHASE(vol_frac, p_i, k);
        }

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(vol_frac, p_i, k) /= frac_sum;
        }
    }

    __global__ void
    update_color_cuda(Data *d_data,
                      UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(color, p_i) *= 0;
        FOR_EACH_PHASE_k() {
            DATA_VALUE(color, p_i) += DATA_VALUE_PHASE(vol_frac, p_i, k) * CONST_VALUE_PHASE(phase_rest_color, k);
        }
    }

    __global__ void
    clear_vol_frac_tmp_cuda(Data *d_data,
                            UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(vol_frac_in, p_i, k) = 0;
            DATA_VALUE_PHASE(vol_frac_out, p_i, k) = 0;
        }
    }

    __global__ void
    update_phase_change_from_drift_cuda(Data *d_data,
                                        UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        if (DATA_VALUE(flag_negative_vol_frac, p_i) != 0)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (p_j == p_i || DATA_VALUE(mat, p_j) != DATA_VALUE(mat, p_i) ||
                DATA_VALUE(flag_negative_vol_frac, p_j) != 0)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            FOR_EACH_PHASE_k() {
                float vol_frac_change =
                        -CONST_VALUE(dt) * DATA_VALUE(volume, p_j) * dot(DATA_VALUE_PHASE(vol_frac, p_i, k) *
                                                                         DATA_VALUE_PHASE(vel_drift_phase, p_i, k) +
                                                                         DATA_VALUE_PHASE(vol_frac, p_j, k) *
                                                                         DATA_VALUE_PHASE(vel_drift_phase, p_j, k),
                                                                         wGrad);

                if (vol_frac_change < 0)
                    DATA_VALUE_PHASE(vol_frac_out, p_i, k) += vol_frac_change;
                else
                    DATA_VALUE_PHASE(vol_frac_in, p_i, k) += vol_frac_change;
            }
        }
    }

    __global__ void
    update_phase_change_from_diffuse_cuda(Data *d_data,
                                          UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                          UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        if (DATA_VALUE(flag_negative_vol_frac, p_i) != 0)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (p_j == p_i || DATA_VALUE(mat, p_j) != DATA_VALUE(mat, p_i) ||
                DATA_VALUE(flag_negative_vol_frac, p_j) != 0)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto x_ij = pos_i - pos_j;
            auto wGrad = CUBIC_KERNEL_GRAD();
            auto factor = dot(wGrad, x_ij) / dot(x_ij, x_ij);

            FOR_EACH_PHASE_k() {
                float vol_frac_ij = DATA_VALUE_PHASE(vol_frac, p_i, k) - DATA_VALUE_PHASE(vol_frac, p_j, k);
                float vol_frac_change = CONST_VALUE(dt) * CONST_VALUE(Cf) * vol_frac_ij * DATA_VALUE(volume, p_j) *
                                        factor;

                if (vol_frac_change < 0)
                    DATA_VALUE_PHASE(vol_frac_out, p_i, k) += vol_frac_change;
                else
                    DATA_VALUE_PHASE(vol_frac_in, p_i, k) += vol_frac_change;
            }
        }
    }

    __device__ float g_all_positive;

    __global__ void
    check_negative_cuda(Data *d_data,
                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (p_i == 0)
            g_all_positive = 1;
        __syncthreads();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        if (DATA_VALUE(flag_negative_vol_frac, p_i) != 0)
            return;

        FOR_EACH_PHASE_k() {
            auto vol_frac_tmp = DATA_VALUE_PHASE(vol_frac, p_i, k) + DATA_VALUE_PHASE(vol_frac_out, p_i, k)
                                + DATA_VALUE_PHASE(vol_frac_in, p_i, k);
            if (vol_frac_tmp < 0) {
                DATA_VALUE(flag_negative_vol_frac, p_i) = 1;
                atomicAdd(&g_all_positive, -1);
            }
        }
    }

    __global__ void
    update_phase_change_cuda(Data *d_data,
                             UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(vol_frac, p_i, k) += DATA_VALUE_PHASE(vol_frac_out, p_i, k)
                                                  + DATA_VALUE_PHASE(vol_frac_in, p_i, k);
        }
    }

    __global__ void
    release_unused_drift_vel_cuda(Data *d_data,
                                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        if (DATA_VALUE(flag_negative_vol_frac, p_i) != 0) {
            FOR_EACH_PHASE_k() {
                DATA_VALUE_PHASE(vel_phase, p_i, k) = DATA_VALUE(vel, p_i);
            }
        }
    }

    __global__ void
    release_negative_cuda(Data *d_data,
                          UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(flag_negative_vol_frac, p_i) = 0;
    }

    __global__ void
    compute_vel_grad_cuda(Data *d_data,
                          UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                          UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        auto vel_i = DATA_VALUE(vel, p_i);
        Mat33f vGrad_sum;

        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != EPM_FLUID)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto vel_j = DATA_VALUE(vel, p_j);
            auto vel_ji = make_vec3f(vel_j - vel_i);
            auto wGrad = make_vec3f(CUBIC_KERNEL_GRAD());
            auto volume_j = DATA_VALUE(mass, p_j) / DATA_VALUE(rest_density, p_j);

            vGrad_sum += volume_j * vel_ji * wGrad;
        }

        DATA_VALUE(vel_grad, p_i) = vGrad_sum;
    }

    __global__ void
    update_ct_cuda(Data *d_data,
                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        float b = 10;

        // compute mix stress
        DATA_VALUE(stress, p_i) *= 0;
        FOR_EACH_PHASE_k() {
            auto dQ = (DATA_VALUE_PHASE(Q, p_i, k) * DATA_VALUE(vel_grad, p_i) +
                       DATA_VALUE(vel_grad, p_i).transpose() * DATA_VALUE_PHASE(Q, p_i, k) -
                       (1 / CONST_VALUE_PHASE(relaxation_time, k)) * (DATA_VALUE_PHASE(Q, p_i, k) - Mat33f::eye()) *
                       (1 - DATA_VALUE_PHASE(Q, p_i, k).trace() / b) -
                       CONST_VALUE_PHASE(thinning_factor, k) * (DATA_VALUE_PHASE(Q, p_i, k) - Mat33f::eye()) *
                       DATA_VALUE_PHASE(Q, p_i, k))
                      * CONST_VALUE(dt);
            DATA_VALUE_PHASE(Q, p_i, k) += dQ;
            DATA_VALUE(stress, p_i) +=
                    DATA_VALUE_PHASE(vol_frac, p_i, k) * CONST_VALUE_PHASE(phase_rest_vis, k) *
                    (DATA_VALUE_PHASE(Q, p_i, k) - Mat33f::eye());
        }

        DATA_VALUE(stress, p_i) += 0.01 * (DATA_VALUE(vel_grad, p_i) + DATA_VALUE(vel_grad, p_i).transpose());
    }

    __global__ void
    add_phase_pct_acc_cuda(Data *d_data,
                           UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                           UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        auto dens_i = DATA_VALUE(rest_density, p_i);
        float3 acc_mix = {0, 0, 0};

        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != EPM_FLUID)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();
            auto dens_j = DATA_VALUE(rest_density, p_j);

            acc_mix += DATA_VALUE(rest_density, p_i) *
                       (DATA_VALUE(stress, p_i) / powf(dens_i, 2) +
                        DATA_VALUE(stress, p_j) / powf(dens_j, 2)) * wGrad;
        }

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(acc_phase, p_i, k) +=
                    acc_mix * (CONST_VALUE(Cd) + (1 - CONST_VALUE(Cd)) *
                                                 (DATA_VALUE(rest_density, p_i) /
                                                  CONST_VALUE_PHASE(phase_rest_density, k)));
        }

        FOR_EACH_NEIGHBOR_Pj_2pass() {
            if (DATA_VALUE(mat, p_j) != EPM_FLUID)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);

            FOR_EACH_PHASE_k() {
                auto v_k_mj = DATA_VALUE(vel, p_j) - DATA_VALUE_PHASE(vel_phase, p_i, k);
                DATA_VALUE_PHASE(acc_phase, p_i, k) += (CONST_VALUE(intermodel_impact_factor) * v_k_mj *
                                                        (1 - DATA_VALUE_PHASE(vol_frac, p_j, k)) *
                                                        CONST_VALUE(fPart_rest_volume) * CUBIC_KERNEL_VALUE()) /
                                                       CONST_VALUE(dt);
            }
        }
    }

    __global__ void
    check_fPart_state_cuda(Data *d_data,
                           UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                           UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(is_in_porous, p_i) = false;
        auto pos_i = DATA_VALUE(pos, p_i);
        float state_value = 0;
        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);

            if (DATA_VALUE(mat, p_j) == EPM_FLUID)
                state_value -= CUBIC_KERNEL_VALUE();

            if (DATA_VALUE(mat, p_j) == EPM_POROUS)
                state_value += (1 - DATA_VALUE(saturation, p_j)) * CUBIC_KERNEL_VALUE();
        }

        if (state_value >= 0) {
            DATA_VALUE(is_in_porous, p_i) = true;
        }
    }

    __global__ void
    add_capillary_acc_cuda(Data *d_data,
                           UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                           UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);

        DATA_VALUE(acc, p_i) *= 0;
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != EPM_POROUS)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);

            FOR_EACH_PHASE_k() {
                DATA_VALUE(acc, p_i) +=
                        DATA_VALUE_PHASE(vol_frac, p_i, k) * CONST_VALUE_PHASE(phase_porous_capillarity_strength, k) *
                        CONST_VALUE_PHASE(phase_rest_density, k) *
                        (1 - DATA_VALUE(saturation, p_j)) * DATA_VALUE(volume, p_j) *
                        CUBIC_KERNEL_GRAD() / DATA_VALUE(fPart_smoothed_vis, p_i);
            }
        }
    }

    __global__ void
    update_fPart_smoothed_vis_cuda(Data *d_data,
                                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(fPart_smoothed_vis, p_i) *= 0;
        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != EPM_FLUID)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);

            FOR_EACH_PHASE_k() {
                DATA_VALUE(fPart_smoothed_vis, p_i) +=
                        DATA_VALUE_PHASE(vol_frac, p_j, k) * CONST_VALUE_PHASE(phase_rest_vis, k) *
                        CUBIC_KERNEL_VALUE();
            }
        }
    }

    __global__ void
    add_pressure_pore_acc_cuda(Data *d_data,
                               UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                               UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID || !DATA_VALUE(is_in_porous, p_i))
            return;

        DATA_VALUE(acc, p_i) *= 0;
        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != EPM_POROUS)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            FOR_EACH_PHASE_k() {
                DATA_VALUE(acc, p_i) -= Mat33f::eye() /
                                        DATA_VALUE(fPart_smoothed_vis, p_i) * DATA_VALUE_PHASE(vol_frac, p_i, k) *
                                        DATA_VALUE(rest_density, p_i) * CONST_VALUE(fPart_rest_volume) *
                                        (1 - DATA_VALUE(porosity, p_j)) * DATA_VALUE(pressure_pore, p_j) *
                                        CUBIC_KERNEL_GRAD();
            }
        }
    }

    __global__ void
    distribute_acc_pore_2_phase_cuda(Data *d_data,
                                     UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(acc_phase, p_i, k) +=
                    CONST_VALUE_PHASE(phase_porous_permeability, k) * (DATA_VALUE(acc, p_i) * (CONST_VALUE(Cd) +
                                                                                               (1 - CONST_VALUE(Cd)) *
                                                                                               (DATA_VALUE(rest_density,
                                                                                                           p_i) /
                                                                                                CONST_VALUE_PHASE(
                                                                                                        phase_rest_density,
                                                                                                        k))));
        }
    }

    __global__ void
    update_porous_saturation_cuda(Data *d_data,
                                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_POROUS)
            return;

        FOR_EACH_PHASE_k() {
            DATA_VALUE_PHASE(saturation_phase, p_i, k) = 0;
        }
        DATA_VALUE(saturation, p_i) = 0;
        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != EPM_FLUID || !DATA_VALUE(is_in_porous, p_j))
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);

            FOR_EACH_PHASE_k() {
                DATA_VALUE_PHASE(saturation_phase, p_i, k) +=
                        CONST_VALUE(fPart_rest_volume) * DATA_VALUE_PHASE(vol_frac, p_j, k) * CUBIC_KERNEL_VALUE();
            }
        }

        FOR_EACH_PHASE_k() {
            DATA_VALUE(saturation, p_i) += DATA_VALUE_PHASE(saturation_phase, p_i, k);
        }

        DATA_VALUE(saturation, p_i) /= DATA_VALUE(rigid_raw_volume, p_i);
        if (DATA_VALUE(saturation, p_i) > 1.f)
            DATA_VALUE(saturation, p_i) = 1.f;

        DATA_VALUE(pressure_pore, p_i) = CONST_VALUE(rest_pressure_pore) * DATA_VALUE(volume, p_i);
    }

    __global__ void
    update_rigid_volume_cuda(Data *d_data,
                             UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                             UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) == EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        DATA_VALUE(rigid_raw_volume, p_i) = 0;
        float delta = 0;
        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);

            delta += CUBIC_KERNEL_VALUE();

            if (DATA_VALUE(mat, p_i) == EPM_POROUS && DATA_VALUE(mat, p_j) == DATA_VALUE(mat, p_i))
                DATA_VALUE(rigid_raw_volume, p_i) += CONST_VALUE(fPart_rest_volume) * CUBIC_KERNEL_VALUE();

            if (DATA_VALUE(mat, p_i) != EPM_POROUS && DATA_VALUE(mat, p_j) == EPM_FLUID)
                delta -= CUBIC_KERNEL_VALUE();
        }

        DATA_VALUE(rest_volume_rate, p_i) = delta;
        DATA_VALUE(volume, p_i) = 1 / delta * (1 - DATA_VALUE(porosity, p_i)) * (1 - DATA_VALUE(saturation, p_i));
    }

    __global__ void
    add_phase_adhesion_acc_cuda(Data *d_data,
                                UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(acc, p_i) *= 0;
        auto pos_i = DATA_VALUE(pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) == EPM_FLUID)
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);

            auto v_ji = DATA_VALUE(vel, p_j) - DATA_VALUE(vel, p_i);
            DATA_VALUE(acc, p_i) += 0.01 * (v_ji * CONST_VALUE(fPart_rest_volume) * CUBIC_KERNEL_VALUE()) /
                                    CONST_VALUE(dt);
        }
    }

    __global__ void
    digging_cuda(Data *d_data,
                 UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        DIGGING_CHECK_THREAD();

        if (DATA_VALUE(digging_mat, p_i) != EPM_POROUS || DATA_VALUE(digging_pPart_alive, p_i) == 0)
            return;

        auto pos_i = DATA_VALUE(digging_pos, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(digging_mat, p_j) != EPM_FLUID)
                continue;

            auto pos_j = DATA_VALUE(digging_pos, p_j);

            if (DATA_VALUE(digging_fPart_miner_flag, p_j) == 1 &&
                length(pos_i - pos_j) < CONST_VALUE(hr_particle_radius)) {
                DATA_VALUE(digging_pPart_alive, p_i) = 0;
                break;
            }
        }
    }

}

/**
 * host invoke impl
 */

namespace VT_Physics::mct {
    __host__ void
    init_data(Data *h_data,
              Data *d_data) {
        init_data_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data);
    }

    __host__ void
    prepare_mct(Data *h_data,
                Data *d_data,
                UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {

        update_rest_density_and_mass_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);
    }

    __host__ void
    sph_precompute(Data *h_data,
                   Data *d_data,
                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        check_fPart_state_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        update_rigid_volume_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        update_porous_saturation_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        update_fPart_smoothed_vis_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        compute_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        compute_df_beta_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);
    }

    __host__ void
    vfsph_div(Data *h_data,
              Data *d_data,
              const std::vector<int> &obj_start_index,
              const std::vector<int> &obj_end_index,
              const std::vector<int> &obj_mats,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams,
              bool &crash) {
        int iter = 0;
        while (true) {
            iter++;

            // compute_delta_compression_ratio()
            compute_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsParams);

            // update_delta_compression_ratio_from_vel_adv()
            update_delta_compression_ratio_from_vel_adv_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsConfig, d_nsParams);

            // compute compressible_ratio
            float compressible_ratio = 0;
            int fluid_obj_num = 0;
            for (int ind = 0; ind < obj_mats.size(); ++ind) {
                if (obj_mats[ind] != EPM_FLUID)
                    continue;
                compressible_ratio += cal_mean(h_data->delta_compression_ratio,
                                               h_data->particle_num,
                                               obj_end_index[ind] - obj_start_index[ind],
                                               obj_start_index[ind]);
                fluid_obj_num++;
            }
            compressible_ratio /= static_cast<float>(fluid_obj_num);


            // compute_kappa_div_from_delta_compression_ratio()
            compute_kappa_div_from_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsParams);

            // vf_update_vel_adv_from_kappa_div()
            vf_update_vel_adv_from_kappa_div_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsConfig, d_nsParams);

            // check compressible_ratio
            if (compressible_ratio < h_data->div_free_threshold || iter > 100)
                break;
        }

        if (iter > 100) {
            std::cerr << "MCT: div-free iteration exceeds 100 times, crash!\n";
            crash = true;
        }
    }

    __host__ void
    apply_pressure_acc(Data *h_data,
                       Data *d_data,
                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        get_acc_pressure_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        clear_phase_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        distribute_acc_pressure_2_phase_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        phase_acc_2_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        update_vel_from_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    mct_gravity_surface(Data *h_data,
                        Data *d_data,
                        UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        clear_phase_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        add_phase_acc_gravity_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        compute_surface_normal_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        add_phase_acc_surface_tension_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        phase_acc_2_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        update_vel_from_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    mpct(Data *h_data,
         Data *d_data,
         UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        clear_phase_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        // compute_vel_grad
        compute_vel_grad_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        // update_ct
        update_ct_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        // add_phase_pct_acc
        add_phase_pct_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        phase_acc_2_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        update_vel_from_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    vfsph_incomp(Data *h_data,
                 Data *d_data,
                 const std::vector<int> &obj_start_index,
                 const std::vector<int> &obj_end_index,
                 const std::vector<int> &obj_mats,
                 UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams,
                 bool &crash) {
        int iter = 0;
        while (true) {
            iter++;

            compute_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsParams);

            update_delta_compression_ratio_from_vel_adv_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsConfig, d_nsParams);

            // compute compressible_ratio
            float compressible_ratio = 0;
            int fluid_obj_num = 0;
            for (int ind = 0; ind < obj_mats.size(); ++ind) {
                if (obj_mats[ind] != EPM_FLUID)
                    continue;
                compressible_ratio += cal_mean(h_data->delta_compression_ratio,
                                               h_data->particle_num,
                                               obj_end_index[ind] - obj_start_index[ind],
                                               obj_start_index[ind]);
                fluid_obj_num++;
            }
            compressible_ratio /= static_cast<float>(fluid_obj_num);

            compute_kappa_incomp_from_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsParams);

            vf_update_vel_adv_from_kappa_incomp_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsConfig, d_nsParams);

            if (compressible_ratio < h_data->incomp_threshold || iter > 100)
                break;
        }

        if (iter > 100) {
            std::cerr << "MCT: incomp-iter exceeds 100 times, crash!\n";
            crash = true;
        }
    }

    __host__ void
    update_pos(Data *h_data,
               Data *d_data,
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        update_pos_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    phase_transfer(Data *h_data,
                   Data *d_data,
                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams,
                   bool &crash) {
        // clear_val_frac_tmp()
        clear_vol_frac_tmp_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        // update_phase_change_from_drift()
        update_phase_change_from_drift_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        // update_phase_change_from_diffuse()
        update_phase_change_from_diffuse_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        // while: check_negative(), update_phase_change_from_drift(), update_phase_change_from_diffuse()
        float all_positive = 0;
        int iter = 1;
        while (true) {
            // check
            check_negative_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsParams);
            cudaMemcpyFromSymbol(&all_positive, g_all_positive, sizeof(float), 0, cudaMemcpyDeviceToHost);
            if (all_positive == 1 || iter > 100)
                break;

            // clear_val_frac_tmp()
            clear_vol_frac_tmp_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsParams);

            // update_phase_change_from_drift()
            update_phase_change_from_drift_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsConfig, d_nsParams);

            // update_phase_change_from_diffuse()
            update_phase_change_from_diffuse_cuda<<<h_data->block_num, h_data->thread_num>>>(
                    d_data, d_nsConfig, d_nsParams);

            iter++;
        }

        if (iter > 100) {
            std::cerr << "MCT: phaseTrans-iter exceeds 100 times, crash!\n";
            crash = true;
        }

        // update_phase_change()
        update_phase_change_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        // release_unused_drift_vel()
        release_unused_drift_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        // release_negative()
        release_negative_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        // regularize_val_frac()
        regularize_vol_frac_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        // update_rest_density_and_mass()
        update_rest_density_and_mass_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        // update_vel_from_phase_vel()
        update_vel_from_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    update_mass_and_vel(Data *h_data,
                        Data *d_data,
                        UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        regularize_vol_frac_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        update_rest_density_and_mass_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        update_vel_from_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    apply_porous_medium(Data *h_data,
                        Data *d_data,
                        UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                        UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        clear_phase_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        add_capillary_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        distribute_acc_pressure_2_phase_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        add_pressure_pore_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        distribute_acc_pore_2_phase_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        phase_acc_2_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        update_vel_from_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    apply_adhesion_force(Data *h_data,
                         Data *d_data,
                         UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        clear_phase_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        add_phase_adhesion_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);

        distribute_acc_pressure_2_phase_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        phase_acc_2_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);

        update_vel_from_phase_vel_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }

    __host__ void
    digging(Data *h_data,
            Data *d_data,
            UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
            UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        digging_cuda<<<h_data->digging_block_num, h_data->thread_num>>>(
                d_data, d_nsConfig, d_nsParams);
    }

    __host__ void
    update_color(Data *h_data,
                 Data *d_data,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        update_color_cuda<<<h_data->block_num, h_data->thread_num>>>(
                d_data, d_nsParams);
    }
}