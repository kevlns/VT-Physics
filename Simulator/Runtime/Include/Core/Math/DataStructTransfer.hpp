/**
 * @brief Todo
 * @date 2024/11/12
 */

#ifndef VT_PHYSICS_DATASTRUCTTRANSFER_HPP
#define VT_PHYSICS_DATASTRUCTTRANSFER_HPP

#include "helper_math_cu11.6.h"
#include "Matrix.hpp"

#include <vector>

namespace VT_Physics {

    inline float3 make_cuFloat3(const std::vector<float> &fVec) {
        return make_float3(fVec[0], fVec[1], fVec[2]);
    }

    __host__ __device__
    inline float3 make_cuFloat3(const Vec3f &vec3f) {
        return make_float3(vec3f.x, vec3f.y, vec3f.z);
    }

    inline Vec3f make_vec3f(const std::vector<float> &fVec) {
        return {fVec[0], fVec[1], fVec[2]};
    }

    __host__ __device__
    inline Vec3f make_vec3f(const float3 &f3) {
        return {f3.x, f3.y, f3.z};
    }

    inline float2 make_cuFloat2(const std::vector<float> &fVec) {
        return make_float2(fVec[0], fVec[1]);
    }

    inline Vec2f make_vec2f(const std::vector<float> &fVec) {
        return {fVec[0], fVec[1]};
    }

    inline float3 *make_cuFloat3Ptr(std::vector<float> &fVec) {
        return reinterpret_cast<float3 *>(fVec.data());
    }

    inline std::vector<float3> make_cuFloat3Vec(const std::vector<float> &fVec) {
        std::vector<float3> f3Vec;
        for (int i = 0; i < fVec.size(); i += 3) {
            f3Vec.push_back(make_float3(fVec[i], fVec[i + 1], fVec[i + 2]));
        }
        return std::move(f3Vec);
    }

    inline std::vector<Vec3f> make_vec3fVec(const std::vector<float> &fVec) {
        std::vector<Vec3f> f3Vec;
        for (int i = 0; i < fVec.size(); i += 3) {
            f3Vec.emplace_back(fVec[i], fVec[i + 1], fVec[i + 2]);
        }
        return std::move(f3Vec);
    }

}

#endif //VT_PHYSICS_DATASTRUCTTRANSFER_HPP
