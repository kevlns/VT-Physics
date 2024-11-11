#include "DFSPHCudaApi.cuh"

#include "Modules/SPH/KernelFunctions.cuh"
#include "DataAnalysis/CommonFunc.hpp"

namespace VT_Physics::dfsph {

#define CHECK_THREAD() \
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= d_data->particle_num)                     \
        return;                                         \
    auto p_i = d_nsParams->particleIndices_cuData[i];

#define FOR_EACH_NEIGHBOR_Pj() \
       auto neib_ind = p_i * d_nsConfig->maxNeighborNum;                        \
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

}

namespace VT_Physics::dfsph { // cuda kernels
    __global__ void
    init_cuda(Data *d_data) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= d_data->particle_num)
            return;

        DATA_VALUE(volume, i) = CONST_VALUE(fPart_rest_volume);
        DATA_VALUE(kappa_div, i) = 0;
        DATA_VALUE(acc, i) *= 0;
        DATA_VALUE(vel_adv, i) = DATA_VALUE(vel, i);
        DATA_VALUE(vis, i) = CONST_VALUE(fPart_rest_viscosity);
        DATA_VALUE(rest_density, i) = CONST_VALUE(fPart_rest_density);
    }

    __global__ void
    update_mass_cuda(Data *d_data,
                     UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(mass, p_i) = DATA_VALUE(rest_density, p_i) * DATA_VALUE(volume, p_i);
    }

    __global__ void
    compute_rigid_volume_cuda(Data *d_data,
                              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_BOUNDARY)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        float delta = 0;
        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);

            if (DATA_VALUE(mat, p_j) == DATA_VALUE(mat, p_i))
                delta += CUBIC_KERNEL_VALUE();
        }

        DATA_VALUE(volume, p_i) = 1.f / delta;
        DATA_VALUE(rest_density, p_i) = DATA_VALUE(volume, p_i) * CONST_VALUE(bPart_rest_density);
    }

    __global__ void
    compute_compression_ratio_cuda(Data *d_data,
                                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        DATA_VALUE(compression_ratio, p_i) *= 0;

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

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
    clear_acc_cuda(Data *d_data,
                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        DATA_VALUE(acc, p_i) *= 0;
    }

    __global__ void
    add_acc_pressure_cuda(Data *d_data,
                          UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(acc, p_i) += (DATA_VALUE(vel_adv, p_i) - DATA_VALUE(vel, p_i)) * CONST_VALUE(inv_dt);
    }

    __global__ void
    acc_2_vel(Data *d_data,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(vel, p_i) += DATA_VALUE(acc, p_i) * CONST_VALUE(dt);
        DATA_VALUE(vel_adv, p_i) = DATA_VALUE(vel, p_i);
    }

    __global__ void
    add_acc_gravity_cuda(Data *d_data,
                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(acc, p_i) += CONST_VALUE(gravity);
    }

    __global__ void
    add_acc_explicit_lap_vis_cuda(Data *d_data,
                                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        // applied to all dynamic objects
        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        float3 acc = {0, 0, 0};
        float h2_001 = 0.001f * pow(CONST_VALUE(h), 2);
        auto pos_i = DATA_VALUE(pos, p_i);
        auto vel_i = DATA_VALUE(vel, p_i);
        FOR_EACH_NEIGHBOR_Pj() {
            if (DATA_VALUE(mat, p_j) != DATA_VALUE(mat, p_i))
                continue;

            auto pos_j = DATA_VALUE(pos, p_j);
            auto x_ij = pos_i - pos_j;
            auto vel_j = DATA_VALUE(vel, p_j);
            auto v_ij = vel_i - vel_j;

            auto vis = (DATA_VALUE(vis, p_i) + DATA_VALUE(vis, p_j)) / 2;

            auto pi = vis * DATA_VALUE(mass, p_j) / DATA_VALUE(rest_density, p_j) * dot(v_ij, x_ij) /
                      (dot(x_ij, x_ij) + h2_001);

            acc += 10 * pi * CUBIC_KERNEL_GRAD();
        }

        DATA_VALUE(acc, p_i) += acc;
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
    add_acc_surface_tension_cuda(Data *d_data,
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
                         SURFACE_TENSION_VALUE() * (pos_i - pos_j) / length(pos_i - pos_j);
            auto acc_2 = -gamma * DATA_VALUE(mass, p_i) *
                         (DATA_VALUE(surface_normal, p_i) - DATA_VALUE(surface_normal, p_j));

            acc += k * (acc_1 + acc_2);
        }

        DATA_VALUE(acc, p_i) += acc;
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

}

namespace VT_Physics::dfsph { // host invoke api
    __host__ void
    init_data(Data *h_data,
              Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // init_data_cuda
        init_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data);
    }

    __host__ void
    prepare_dfsph(Data *h_data,
                  Data *d_data,
                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // update_rest_density_and_mass()
        update_mass_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                    d_nsParams);

        // compute_rigid_volume()
        compute_rigid_volume_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                             d_nsConfig,
                                                                             d_nsParams);
    }

    __host__ void
    sph_precompute(Data *h_data,
                   Data *d_data,
                   UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                   UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // compute_compression_ratio(), AKA step_sph_compute_compression_ratio()
        compute_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                  d_nsConfig,
                                                                                  d_nsParams);

        // compute_df_beta(), AKA step_df_compute_beta()
        compute_df_beta_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                        d_nsConfig,
                                                                        d_nsParams);
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
            compute_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                            d_nsParams);

            // update_delta_compression_ratio_from_vel_adv()
            update_delta_compression_ratio_from_vel_adv_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                                        d_nsConfig,
                                                                                                        d_nsParams);

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
            compute_kappa_div_from_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                                           d_nsParams);

            // vf_update_vel_adv_from_kappa_div()
            vf_update_vel_adv_from_kappa_div_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                             d_nsConfig,
                                                                                             d_nsParams);

            // check compressible_ratio
            if (compressible_ratio < h_data->div_free_threshold || iter > 100)
                break;
        }

        if (iter > 100) {
            std::cerr << "DFSPH: div-free iteration exceeds 100 times, crash!\n";
            crash = true;
        }
    }

    __host__ void
    apply_pressure_acc(Data *h_data,
                       Data *d_data,
                       UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // clear_phase_acc()
        clear_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                  d_nsParams);

        // get_acc_pressure()
        add_acc_pressure_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                         d_nsParams);

        // phase_acc_2_phase_vel()
        acc_2_vel<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                             d_nsParams);
    }

    __host__ void
    dfsph_gravity_vis_surface(Data *h_data,
                              Data *d_data,
                              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // clear_phase_acc()
        clear_acc_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                  d_nsParams);

        // add_phase_acc_gravity()
        add_acc_gravity_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                        d_nsParams);

        // add_acc_explicit_lap_vis_cuda()
        add_acc_explicit_lap_vis_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                 d_nsConfig,
                                                                                 d_nsParams);

        // compute_surface_normal
        compute_surface_normal_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                               d_nsConfig,
                                                                               d_nsParams);

        // add_acc_surface_tension_cuda()
        add_acc_surface_tension_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                d_nsConfig,
                                                                                d_nsParams);

        // acc_2_vel()
        acc_2_vel<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                             d_nsParams);
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

            // compute_delta_compression_ratio()
            compute_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                            d_nsParams);

            // update_delta_compression_ratio_from_vel_adv()
            update_delta_compression_ratio_from_vel_adv_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                                        d_nsConfig,
                                                                                                        d_nsParams);

            // update_vf_compressible_ratio()
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

            // compute_kappa_incomp_from_delta_compression_ratio()
            compute_kappa_incomp_from_delta_compression_ratio_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                                              d_nsParams);

            // vf_update_vel_adv_from_kappa_incomp()
            vf_update_vel_adv_from_kappa_incomp_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                                d_nsConfig,
                                                                                                d_nsParams);

            // check compressible_ratio
            if (compressible_ratio < h_data->incomp_threshold || iter > 100)
                break;
        }

        if (iter > 100) {
            std::cerr << "DFSPH: incomp-iter exceeds 100 times, crash!\n";
            crash = true;
        }
    }

    __host__ void
    update_pos(Data *h_data,
               Data *d_data,
               UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        update_pos_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                   d_nsParams);
    }
}