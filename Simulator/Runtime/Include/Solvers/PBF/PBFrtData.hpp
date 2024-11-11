/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_PBFRTDATA_HPP
#define VT_PHYSICS_PBFRTDATA_HPP

#include <cstdint>
#include <iostream>

#include "Core/Math/helper_math_cu11.6.h"
#include "Framework/Material.hpp"

namespace VT_Physics::pbf {

    struct Data {
    private:
        bool is_malloced{false};

    public:
        // cuda kernel used
        unsigned block_num;
        unsigned thread_num;

        // constant data
        float dt;
        float inv_dt;
        float inv_dt2;
        float cur_simTime;
        float particle_radius;
        float fPart_rest_density;
        float fPart_rest_volume;
        float bPart_rest_density;
        float h;
        float XSPH_k;
        float3 gravity;
        uint32_t particle_num;

        // dynamic data
        int *mat{nullptr};
        float3 *pos{nullptr};
        float3 *dx{nullptr};
        float3 *vel{nullptr};
        float3 *color{nullptr};
        float3 *error_grad{nullptr};
        float *mass{nullptr};
        float *volume{nullptr};
        float *density_sph{nullptr};
        float *lamb{nullptr};
        float *error{nullptr};

        __host__
        bool malloc() {
            free();
            if (!is_malloced) {
                cudaMalloc((void **) &mat, particle_num * sizeof(int));
                cudaMalloc((void **) &pos, particle_num * sizeof(float3));
                cudaMalloc((void **) &dx, particle_num * sizeof(float3));
                cudaMalloc((void **) &vel, particle_num * sizeof(float3));
                cudaMalloc((void **) &mass, particle_num * sizeof(float));
                cudaMalloc((void **) &volume, particle_num * sizeof(float));
                cudaMalloc((void **) &color, particle_num * sizeof(float3));
                cudaMalloc((void **) &error_grad, particle_num * sizeof(float3));
                cudaMalloc((void **) &density_sph, particle_num * sizeof(float));
                cudaMalloc((void **) &lamb, particle_num * sizeof(float));
                cudaMalloc((void **) &error, particle_num * sizeof(float));

                if (cudaGetLastError() != cudaSuccess) {
                    std::cerr << "PBFrtData::malloc failed.\n";
                    return false;
                }

                is_malloced = true;
            }
            return is_malloced;
        }

        __host__
        void free() const {
            if (is_malloced) {
                cudaFree(pos);
                cudaFree(dx);
                cudaFree(vel);
                cudaFree(color);
                cudaFree(error_grad);
                cudaFree(mass);
                cudaFree(volume);
                cudaFree(density_sph);
                cudaFree(lamb);
                cudaFree(error);
                cudaFree(mat);

                if (cudaGetLastError() != cudaSuccess) {
                    std::cerr << "PBFrtData::free failed.\n";
                    return;
                }
            }
        }
    };
}

#endif //VT_PHYSICS_PBFRTDATA_HPP
