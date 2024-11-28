/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_MCTRTDATA_HPP
#define VT_PHYSICS_MCTRTDATA_HPP

#include <cstdint>
#include <iostream>

#include "Core/Math/helper_math_cu11.6.h"
#include "Core/Math/Matrix.hpp"
#include "Framework/Material.hpp"

namespace VT_Physics::mct {

    struct Data {
    private:
        bool is_malloced{false};

    public:
        // cuda kernel used
        unsigned block_num;
        unsigned thread_num;
        unsigned digging_block_num;

        // constant data
        float dt;
        float inv_dt;
        float inv_dt2;
        float cur_simTime;
        float particle_radius;
        float hr_particle_radius;
        float fPart_rest_volume;
        float h;
        float div_free_threshold;
        float incomp_threshold;
        float surface_tension_coefficient;
        float3 gravity;
        uint32_t particle_num;
        int phase_num;
        float Cf;
        float Cd;
        float intermodel_impact_factor;
        float porous_porosity;
        float *phase_porous_capillarity_strength{nullptr};
        float *phase_porous_permeability{nullptr};
        float *phase_rest_density;
        float *phase_rest_vis;
        float3 *phase_rest_color;
        float *relaxation_time;
        float *thinning_factor;
        float rest_pressure_pore;
        // for digging policy
        uint32_t digging_particle_num;

        // dynamic data
        int *mat{nullptr};
        float3 *pos{nullptr};
        float3 *pos_adv{nullptr};
        float3 *vel{nullptr};
        float3 *vel_adv{nullptr};
        float3 *acc{nullptr};
        float3 *surface_normal{nullptr};
        float3 *color{nullptr};
        float *mass{nullptr};
        float *volume{nullptr};
        float *rest_density{nullptr};
        float *compression_ratio{nullptr};
        float *delta_compression_ratio{nullptr};
        float *df_alpha{nullptr};
        float3 *df_alpha_1{nullptr};
        float *df_alpha_2{nullptr};
        float *kappa_div{nullptr};
        float *kappa_incomp{nullptr};
        float *vol_frac{nullptr};
        float *vol_frac_in{nullptr};
        float *vol_frac_out{nullptr};
        float3 *vel_phase{nullptr};
        float3 *acc_phase{nullptr};
        float3 *vel_drift_phase{nullptr};
        float *flag_negative_vol_frac{nullptr};
        Mat33f *vel_grad{nullptr};
        Mat33f *Q{nullptr};
        Mat33f *stress{nullptr};
        float *porosity{nullptr};
        float *saturation{nullptr};
        float *saturation_phase{nullptr};
        float *rest_volume_rate{nullptr};
        bool *is_in_porous{nullptr};
        float *rigid_raw_volume{nullptr};
        float *pressure_pore{nullptr};
        float *fPart_smoothed_vis{nullptr};
        // for digging policy
        float3 *digging_pos{nullptr};
        int *digging_fPart_miner_flag{nullptr};
        float *digging_porosity{nullptr};
        int *digging_pPart_alive{nullptr};
        int *digging_mat{nullptr};

        __host__
        bool malloc() {
            free();
            if (!is_malloced) {
                cudaMalloc((void **) &mat, particle_num * sizeof(int));
                cudaMalloc((void **) &pos, particle_num * sizeof(float3));
                cudaMalloc((void **) &pos_adv, particle_num * sizeof(float3));
                cudaMalloc((void **) &vel, particle_num * sizeof(float3));
                cudaMalloc((void **) &vel_adv, particle_num * sizeof(float3));
                cudaMalloc((void **) &acc, particle_num * sizeof(float3));
                cudaMalloc((void **) &surface_normal, particle_num * sizeof(float3));
                cudaMalloc((void **) &mass, particle_num * sizeof(float));
                cudaMalloc((void **) &volume, particle_num * sizeof(float));
                cudaMalloc((void **) &color, particle_num * sizeof(float3));
                cudaMalloc((void **) &rest_density, particle_num * sizeof(float));
                cudaMalloc((void **) &compression_ratio, particle_num * sizeof(float));
                cudaMalloc((void **) &delta_compression_ratio, particle_num * sizeof(float));
                cudaMalloc((void **) &df_alpha, particle_num * sizeof(float));
                cudaMalloc((void **) &df_alpha_1, particle_num * sizeof(float3));
                cudaMalloc((void **) &df_alpha_2, particle_num * sizeof(float));
                cudaMalloc((void **) &kappa_div, particle_num * sizeof(float));
                cudaMalloc((void **) &kappa_incomp, particle_num * sizeof(float));
                cudaMalloc((void **) &vol_frac, particle_num * phase_num * sizeof(float));
                cudaMalloc((void **) &vol_frac_in, particle_num * phase_num * sizeof(float));
                cudaMalloc((void **) &vol_frac_out, particle_num * phase_num * sizeof(float));
                cudaMalloc((void **) &vel_phase, particle_num * phase_num * sizeof(float3));
                cudaMalloc((void **) &acc_phase, particle_num * phase_num * sizeof(float3));
                cudaMalloc((void **) &vel_drift_phase, particle_num * phase_num * sizeof(float3));
                cudaMalloc((void **) &flag_negative_vol_frac, particle_num * sizeof(float));
                cudaMalloc((void **) &vel_grad, particle_num * sizeof(Mat33f));
                cudaMalloc((void **) &Q, particle_num * phase_num * sizeof(Mat33f));
                cudaMalloc((void **) &stress, particle_num * sizeof(Mat33f));
                cudaMalloc((void **) &porosity, particle_num * sizeof(float));
                cudaMalloc((void **) &saturation, particle_num * sizeof(float));
                cudaMalloc((void **) &saturation_phase, particle_num * phase_num * sizeof(float));
                cudaMalloc((void **) &rest_volume_rate, particle_num * sizeof(float));
                cudaMalloc((void **) &is_in_porous, particle_num * sizeof(bool));
                cudaMalloc((void **) &rigid_raw_volume, particle_num * sizeof(float));
                cudaMalloc((void **) &pressure_pore, particle_num * sizeof(float));
                cudaMalloc((void **) &fPart_smoothed_vis, particle_num * sizeof(float));

                // for digging policy
                if (digging_particle_num != 0) {
                    cudaMalloc((void **) &digging_pos, digging_particle_num * sizeof(float3));
                    cudaMalloc((void **) &digging_fPart_miner_flag, digging_particle_num * sizeof(int));
                    cudaMalloc((void **) &digging_porosity, digging_particle_num * sizeof(float));
                    cudaMalloc((void **) &digging_pPart_alive, digging_particle_num * sizeof(int));
                    cudaMalloc((void **) &digging_mat, digging_particle_num * sizeof(int));
                }

                if (cudaGetLastError() != cudaSuccess) {
                    std::cerr << "MCTrtData::malloc failed.\n";
                    return false;
                }

                is_malloced = true;
            }
            return is_malloced;
        }

        __host__
        void free() const {
            if (is_malloced) {
                cudaFree(mat);
                cudaFree(pos);
                cudaFree(pos_adv);
                cudaFree(vel);
                cudaFree(vel_adv);
                cudaFree(acc);
                cudaFree(surface_normal);
                cudaFree(mass);
                cudaFree(volume);
                cudaFree(color);
                cudaFree(rest_density);
                cudaFree(compression_ratio);
                cudaFree(delta_compression_ratio);
                cudaFree(df_alpha);
                cudaFree(df_alpha_1);
                cudaFree(df_alpha_2);
                cudaFree(kappa_div);
                cudaFree(kappa_incomp);
                cudaFree(vol_frac);
                cudaFree(vol_frac_in);
                cudaFree(vol_frac_out);
                cudaFree(vel_phase);
                cudaFree(acc_phase);
                cudaFree(vel_drift_phase);
                cudaFree(flag_negative_vol_frac);
                cudaFree(vel_grad);
                cudaFree(Q);
                cudaFree(stress);
                cudaFree(porosity);
                cudaFree(saturation);
                cudaFree(saturation_phase);
                cudaFree(rest_volume_rate);
                cudaFree(is_in_porous);
                cudaFree(rigid_raw_volume);
                cudaFree(pressure_pore);
                cudaFree(fPart_smoothed_vis);

                // for digging policy
                if (digging_particle_num != 0) {
                    cudaFree(digging_pos);
                    cudaFree(digging_fPart_miner_flag);
                    cudaFree(digging_porosity);
                    cudaFree(digging_pPart_alive);
                    cudaFree(digging_mat);
                }

                cudaFree(phase_porous_permeability);
                cudaFree(phase_porous_capillarity_strength);
                cudaFree(phase_rest_density);
                cudaFree(phase_rest_vis);
                cudaFree(phase_rest_color);
                cudaFree(relaxation_time);
                cudaFree(thinning_factor);

                if (cudaGetLastError() != cudaSuccess) {
                    std::cerr << "MCTrtData::free failed.\n";
                    return;
                }
            }
        }
    };
}

#endif //VT_PHYSICS_MCTRTDATA_HPP
