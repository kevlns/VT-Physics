/**
 * @brief Todo
 * @date 2024/11/8
 */

#ifndef VT_PHYSICS_KERNELFUNCTIONS_CUH
#define VT_PHYSICS_KERNELFUNCTIONS_CUH

#include "Core/Math/helper_math_cu11.6.h"

namespace VT_Physics::sph {

    __device__ inline float
    cubic_value(const float3 &r, float h) {
        const float r_norm = length(r);
        const float PI = 3.14159265;
        const float cubicSigma = 8.f / PI / static_cast<float>(powf(h, 3));

        float res = 0.0;
        float invH = 1 / h;
        float q = r_norm * invH;

        if (q <= 1) {
            if (q <= 0.5) {
                auto q2 = q * q;
                auto q3 = q2 * q;
                res = static_cast<float>(cubicSigma * (6.0 * q3 - 6.0 * q2 + 1));
            } else {
                res = static_cast<float>(cubicSigma * 2 * powf(1 - q, 3));
            }
        }

        return res;
    }

    __device__ inline float3
    cubic_gradient(const float3 &r, float h) {
        static const float PI = 3.14159265;
        const float cubicSigma = 8.f / PI / powf(h, 3);

        auto res = float3();
        float invH = 1 / h;
        float q = length(r) * invH;

        if (q < 1e-6 || q > 1)
            return res;

        float3 grad_q = r / (length(r) * h);
        if (q <= 0.5)
            res = (6 * (3 * q * q - 2 * q)) * grad_q * cubicSigma;
        else {
            auto factor = 1 - q;
            res = -6 * factor * factor * grad_q * cubicSigma;
        }

        return res;
    }

    __host__ __device__ inline float
    surface_tension_C(const float r_norm, const float h) {
        static const float PI = 3.14159265;
        const float cSigma = 32.f / PI / static_cast<float>(std::pow(h, 9));

        if (r_norm * 2 > h && r_norm <= h)
            return cSigma * std::pow(h - r_norm, 3) * std::pow(r_norm, 3);
        else if (r_norm > 0 && r_norm * 2 <= h)
            return cSigma * (2 * std::pow(h - r_norm, 3) * std::pow(r_norm, 3) - std::pow(h, 6) / 64);

        return 0;
    }

    __host__ __device__ inline float
    df_viscosity_kernel_laplacian(const float3 &r, const float h) {
        static const float PI = 3.14159265;
        const float r_norm = length(r);
        return (r_norm <= h) ? (45.0f * (h - r_norm) / (PI * powf(h, 6))) : 0.0f;
    }

    __device__ inline float
    adhesion_kernel_value(const float3 &r, float h) {
        static const float PI = 3.14159265;
        const float r_norm = length(r);
        const float cubicSigma = 8.f / PI / static_cast<float>(powf(h, 3));
        const float factor = 0.007f / powf(h, 3.25);

        float res = 0.0;
        float invH = 1 / h;
        float q = r_norm * invH;

        if (q <= 1 && q > 0.5)
            res = static_cast<float>(factor * powf(-4 * r_norm * r_norm * invH + 6 * r_norm - 2 * h, -4));

        return res;
    }

}

#endif //VT_PHYSICS_KERNELFUNCTIONS_CUH
