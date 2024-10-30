/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_PBFRTDATA_HPP
#define VT_PHYSICS_PBFRTDATA_HPP

#include "Core/Math/helper_math_cu11.6.h"
#include "Framework/Material.hpp"
#include "Logger/Logger.hpp"

namespace VT_Physics::pbf {

    struct Data {
        // constant data
        float dt;
        float inv_dt;
        float inv_dt2;
        float cur_simTime;
        float particle_radius;
        float fPart_rest_density;
        float fPart_rest_volume;
        float h;
        float3 gravity;
        uint32_t particle_num;

        // dynamic data
        float3 *pos;
        float3 *vel;
        float3 *dx;
        float3 *acc;
        float3 *error_grad;
        float3 *color;
        float *pressure;
        float *volume;
        float *mass;
        float *density_sph;
        float *lamb;
        float *error;
        ParticleMaterial *mat;

        bool is_malloced{false};

        bool malloc() {
            if (!is_malloced) {
                size_t size_1 = particle_num * sizeof(float);
                size_t size_2 = particle_num * sizeof(float3);
                size_t size_3 = particle_num * sizeof(ParticleMaterial);

                cudaMalloc((void **) &pos, size_2);
                cudaMalloc((void **) &dx, size_2);
                cudaMalloc((void **) &vel, size_2);
                cudaMalloc((void **) &mass, size_1);
                cudaMalloc((void **) &volume, size_1);
                cudaMalloc((void **) &color, size_2);
                cudaMalloc((void **) &error_grad, size_2);
                cudaMalloc((void **) &density_sph, size_1);
                cudaMalloc((void **) &lamb, size_1);
                cudaMalloc((void **) &error, size_1);
                cudaMalloc((void **) &mat, size_3);

                if (cudaGetLastError() != cudaSuccess) {
                    LOG_ERROR("PBFrtData::malloc failed.");
                    return false;
                }

                LOG_INFO("PBFrtData::malloc success.");
                is_malloced = true;
                return true;
            }
        }

        void free() const {
            if (is_malloced) {
                cudaFree(pos);
                cudaFree(dx);
                cudaFree(vel);
                cudaFree(mass);
                cudaFree(volume);
                cudaFree(color);
                cudaFree(error_grad);
                cudaFree(density_sph);
                cudaFree(lamb);
                cudaFree(error);
                cudaFree(mat);

                if (cudaGetLastError() != cudaSuccess)
                    LOG_ERROR("PBFrtData::~Data failed.");

                LOG_INFO("PBFrtData::~Data success.")
            }
        }
    };
}

#endif //VT_PHYSICS_PBFRTDATA_HPP
