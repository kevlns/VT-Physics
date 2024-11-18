
//
// Created by ADMIN on 2024/3/5.
//

#ifndef VT_PHYSICS_MATRIX_HPP
#define VT_PHYSICS_MATRIX_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace VT_Physics {

    template<typename T>
    class Set {
    public:
        __host__ __device__ Set(size_t s = 0) :
                m_size(s) {
            m_data = new T[s];
            memset(m_data, 0, s * sizeof(T));
        };

        __host__ __device__ Set(const std::vector<T> &data) {
            assign(data);
        }

        __host__ __device__ Set(const Set &src) {
            mfree();
            m_data = new T[src.m_size];
            m_size = src.m_size;
            memcpy_s(m_data, m_size * sizeof(T), src.m_data, m_size * sizeof(T));
        }

        __host__ __device__        Set &operator=(const Set &src) {
            mfree();
            m_data = new T[src.m_size];
            m_size = src.m_size;
            memcpy_s(m_data, m_size * sizeof(T), src.m_data, m_size * sizeof(T));
            return *this;
        }

        // =
        __host__ __device__        Set &operator=(const std::vector<T> &data) {
            mfree();
            m_data = new T[data.size()];
            size_t s = sizeof(T) * data.size();
            memcpy_s(m_data, s, data.data(), s);
            m_size = data.size();
            return *this;
        }

        __host__ __device__        ~Set() {
            mfree();
        }

        __host__ __device__        size_t size() const {
            return m_size;
        }

        T *data() const {
            return m_data;
        }

        __host__ __device__        T &operator[](size_t index) {
            return *(m_data + index);
        }

        __host__
        const T &operator[](size_t index) const {
            return *(m_data + index);
        }

        __host__
        void assign(const std::vector<T> &data) {
            mfree();
            m_data = new T[data.size()];
            size_t s = sizeof(T) * data.size();
            memcpy_s(m_data, s, data.data(), s);
            m_size = data.size();
        }

    private:
        __host__ __device__        void mfree() {
            if (m_data)
                delete[] m_data;
            m_size = 0;
        }

    private:
        T *m_data{nullptr};
        size_t m_size{0};
    };

    struct Mat33f {
        float r0_0{0};
        float r0_1{0};
        float r0_2{0};
        float r1_0{0};
        float r1_1{0};
        float r1_2{0};
        float r2_0{0};
        float r2_1{0};
        float r2_2{0};

        __host__ __device__        float &operator[](int index) {
            switch (index) {
                case 0:
                    return r0_0;
                case 1:
                    return r0_1;
                case 2:
                    return r0_2;
                case 3:
                    return r1_0;
                case 4:
                    return r1_1;
                case 5:
                    return r1_2;
                case 6:
                    return r2_0;
                case 7:
                    return r2_1;
                case 8:
                    return r2_2;
                default:
                    return r0_0;
            }
        }

        __host__ __device__        const float &operator[](int index) const {
            switch (index) {
                case 0:
                    return r0_0;
                case 1:
                    return r0_1;
                case 2:
                    return r0_2;
                case 3:
                    return r1_0;
                case 4:
                    return r1_1;
                case 5:
                    return r1_2;
                case 6:
                    return r2_0;
                case 7:
                    return r2_1;
                case 8:
                    return r2_2;
                default:
                    return r0_0;
            }
        }

        __host__ __device__        void operator*=(const float n) {
            this->r0_0 *= n;
            this->r0_1 *= n;
            this->r0_2 *= n;

            this->r1_0 *= n;
            this->r1_1 *= n;
            this->r1_2 *= n;

            this->r2_0 *= n;
            this->r2_1 *= n;
            this->r2_2 *= n;
        }

        __host__ __device__        void assign(const Mat33f &rhs) {
            this->r0_0 = rhs.r0_0;
            this->r0_1 = rhs.r0_1;
            this->r0_2 = rhs.r0_2;

            this->r1_0 = rhs.r1_0;
            this->r1_1 = rhs.r1_1;
            this->r1_2 = rhs.r1_2;

            this->r2_0 = rhs.r2_0;
            this->r2_1 = rhs.r2_1;
            this->r2_2 = rhs.r2_2;
        }

        __host__ __device__
        void operator/=(const float n) {
            this->r0_0 /= n;
            this->r0_1 /= n;
            this->r0_2 /= n;

            this->r1_0 /= n;
            this->r1_1 /= n;
            this->r1_2 /= n;

            this->r2_0 /= n;
            this->r2_1 /= n;
            this->r2_2 /= n;
        }

        __host__ __device__
        void operator+=(const Mat33f &rhs) {
            this->r0_0 += rhs.r0_0;
            this->r0_1 += rhs.r0_1;
            this->r0_2 += rhs.r0_2;

            this->r1_0 += rhs.r1_0;
            this->r1_1 += rhs.r1_1;
            this->r1_2 += rhs.r1_2;

            this->r2_0 += rhs.r2_0;
            this->r2_1 += rhs.r2_1;
            this->r2_2 += rhs.r2_2;
        }

        __host__ __device__
        void operator-=(const Mat33f &rhs) {
            this->r0_0 -= rhs.r0_0;
            this->r0_1 -= rhs.r0_1;
            this->r0_2 -= rhs.r0_2;

            this->r1_0 -= rhs.r1_0;
            this->r1_1 -= rhs.r1_1;
            this->r1_2 -= rhs.r1_2;

            this->r2_0 -= rhs.r2_0;
            this->r2_1 -= rhs.r2_1;
            this->r2_2 -= rhs.r2_2;
        }

        __host__ __device__
        float trace() const {
            return r0_0 + r1_1 + r2_2;
        }

        __host__ __device__
        Mat33f transpose() {
            return {r0_0, r1_0, r2_0,
                    r0_1, r1_1, r2_1,
                    r0_2, r1_2, r2_2};
        }

        __host__ __device__
        static Mat33f eye() {
            return {1, 0, 0,
                    0, 1, 0,
                    0, 0, 1};
        }
    };

    struct Mat22f {
        float r0_0{0};
        float r0_1{0};
        float r1_0{0};
        float r1_1{0};

        Mat22f operator*(const Mat22f &rhs) const {
            float _r0_0 = this->r0_0 * rhs.r0_0 + this->r0_1 * rhs.r1_0;
            float _r0_1 = this->r0_0 * rhs.r0_1 + this->r0_1 * rhs.r1_1;

            float _r1_0 = this->r1_0 * rhs.r0_0 + this->r1_1 * rhs.r1_0;
            float _r1_1 = this->r1_0 * rhs.r0_1 + this->r1_1 * rhs.r1_1;

            return {_r0_0, _r0_1,
                    _r1_0, _r1_1};
        }

        Mat22f operator*(const float n) const {
            float _r0_0 = this->r0_0 * n;
            float _r0_1 = this->r0_1 * n;

            float _r1_0 = this->r1_0 * n;
            float _r1_1 = this->r1_1 * n;


            return {_r0_0, _r0_1,
                    _r1_0, _r1_1};
        }

        Mat22f operator/(const float n) const {
            float _r0_0 = this->r0_0 / n;
            float _r0_1 = this->r0_1 / n;

            float _r1_0 = this->r1_0 / n;
            float _r1_1 = this->r1_1 / n;


            return {_r0_0, _r0_1,
                    _r1_0, _r1_1};
        }

        Mat22f operator+(const Mat22f &rhs) const {
            float _r0_0 = this->r0_0 + rhs.r0_0;
            float _r0_1 = this->r0_1 + rhs.r0_1;

            float _r1_0 = this->r1_0 + rhs.r1_0;
            float _r1_1 = this->r1_1 + rhs.r1_1;

            return {_r0_0, _r0_1,
                    _r1_0, _r1_1};
        }

        Mat22f operator-(const Mat22f &rhs) const {
            float _r0_0 = this->r0_0 - rhs.r0_0;
            float _r0_1 = this->r0_1 - rhs.r0_1;

            float _r1_0 = this->r1_0 - rhs.r1_0;
            float _r1_1 = this->r1_1 - rhs.r1_1;

            return {_r0_0, _r0_1,
                    _r1_0, _r1_1};
        }

        void operator*=(const Mat22f &rhs) {
            float _r0_0 = this->r0_0 * rhs.r0_0 + this->r0_1 * rhs.r1_0;
            float _r0_1 = this->r0_0 * rhs.r0_1 + this->r0_1 * rhs.r1_1;

            float _r1_0 = this->r1_0 * rhs.r0_0 + this->r1_1 * rhs.r1_0;
            float _r1_1 = this->r1_0 * rhs.r0_1 + this->r1_1 * rhs.r1_1;

            this->r0_0 = _r0_0;
            this->r0_1 = _r0_1;
            this->r1_0 = _r1_0;
            this->r1_1 = _r1_1;
        }

        void operator*=(const float n) {
            this->r0_0 *= n;
            this->r0_1 *= n;

            this->r1_0 *= n;
            this->r1_1 *= n;
        }

        void operator/=(const float n) {
            this->r0_0 /= n;
            this->r0_1 /= n;

            this->r1_0 /= n;
            this->r1_1 /= n;
        }

        void operator+=(const Mat22f &rhs) {
            this->r0_0 += rhs.r0_0;
            this->r0_1 += rhs.r0_1;

            this->r1_0 += rhs.r1_0;
            this->r1_1 += rhs.r1_1;
        }

        void operator-=(const Mat22f &rhs) {
            this->r0_0 -= rhs.r0_0;
            this->r0_1 -= rhs.r0_1;

            this->r1_0 -= rhs.r1_0;
            this->r1_1 -= rhs.r1_1;
        }

        float trace() const {
            return r0_0 + r1_1;
        }

        Mat22f transpose() {
            return {r0_0, r1_0,
                    r0_1, r1_1};
        }

        static Mat33f eye() {
            return {1, 0, 0,
                    0, 1, 0,
                    0, 0, 1};
        }
    };

    struct Vec2f {
        float x{0};
        float y{0};

        __host__ __device__
        inline float length() const {
            return sqrtf(x * x + y * y);
        }
    };

    struct Vec3f {
        float x{0};
        float y{0};
        float z{0};

        __host__ __device__
        Vec3f(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}

        __host__ __device__
        inline float length() const {
            return sqrtf(x * x + y * y + z * z);
        }
    };

    struct Vec3i {
        int x{0};
        int y{0};
        int z{0};
    };

    struct Vec3ui {
        unsigned x{0};
        unsigned y{0};
        unsigned z{0};
    };

    inline std::ostream &operator<<(std::ostream &out, const Mat33f &mat) {
        out << "[" << mat.r0_0 << " " << mat.r0_1 << " " << mat.r0_2 << "\n"
            << mat.r1_0 << " " << mat.r1_1 << " " << mat.r1_2 << "\n"
            << mat.r2_0 << " " << mat.r2_1 << " " << mat.r2_2 << "]";
        return out;
    }

    inline std::ostream &operator<<(std::ostream &out, const Vec3f &vec) {
        out << "[" << vec.x << " " << vec.y << " " << vec.z << "]";
        return out;
    }

// math operating

// Mat33
    __host__ __device__
    inline Mat33f operator+(const Mat33f &lhs, const Mat33f &rhs) {
        float _r0_0 = lhs.r0_0 + rhs.r0_0;
        float _r0_1 = lhs.r0_1 + rhs.r0_1;
        float _r0_2 = lhs.r0_2 + rhs.r0_2;

        float _r1_0 = lhs.r1_0 + rhs.r1_0;
        float _r1_1 = lhs.r1_1 + rhs.r1_1;
        float _r1_2 = lhs.r1_2 + rhs.r1_2;

        float _r2_0 = lhs.r2_0 + rhs.r2_0;
        float _r2_1 = lhs.r2_1 + rhs.r2_1;
        float _r2_2 = lhs.r2_2 + rhs.r2_2;

        return {_r0_0, _r0_1, _r0_2,
                _r1_0, _r1_1, _r1_2,
                _r2_0, _r2_1, _r2_2};
    }

    __host__ __device__
    inline Mat33f operator-(const Mat33f &lhs, const Mat33f &rhs) {
        float _r0_0 = lhs.r0_0 - rhs.r0_0;
        float _r0_1 = lhs.r0_1 - rhs.r0_1;
        float _r0_2 = lhs.r0_2 - rhs.r0_2;

        float _r1_0 = lhs.r1_0 - rhs.r1_0;
        float _r1_1 = lhs.r1_1 - rhs.r1_1;
        float _r1_2 = lhs.r1_2 - rhs.r1_2;

        float _r2_0 = lhs.r2_0 - rhs.r2_0;
        float _r2_1 = lhs.r2_1 - rhs.r2_1;
        float _r2_2 = lhs.r2_2 - rhs.r2_2;

        return {_r0_0, _r0_1, _r0_2,
                _r1_0, _r1_1, _r1_2,
                _r2_0, _r2_1, _r2_2};
    }

    __host__ __device__
    inline Mat33f operator*(const Mat33f &lhs, const Mat33f &rhs) {
        float _r0_0 = lhs.r0_0 * rhs.r0_0 + lhs.r0_1 * rhs.r1_0 + lhs.r0_2 * rhs.r2_0;
        float _r0_1 = lhs.r0_0 * rhs.r0_1 + lhs.r0_1 * rhs.r1_1 + lhs.r0_2 * rhs.r2_1;
        float _r0_2 = lhs.r0_0 * rhs.r0_2 + lhs.r0_1 * rhs.r1_2 + lhs.r0_2 * rhs.r2_2;

        float _r1_0 = lhs.r1_0 * rhs.r0_0 + lhs.r1_1 * rhs.r1_0 + lhs.r1_2 * rhs.r2_0;
        float _r1_1 = lhs.r1_0 * rhs.r0_1 + lhs.r1_1 * rhs.r1_1 + lhs.r1_2 * rhs.r2_1;
        float _r1_2 = lhs.r1_0 * rhs.r0_2 + lhs.r1_1 * rhs.r1_2 + lhs.r1_2 * rhs.r2_2;

        float _r2_0 = lhs.r2_0 * rhs.r0_0 + lhs.r2_1 * rhs.r1_0 + lhs.r2_2 * rhs.r2_0;
        float _r2_1 = lhs.r2_0 * rhs.r0_1 + lhs.r2_1 * rhs.r1_1 + lhs.r2_2 * rhs.r2_1;
        float _r2_2 = lhs.r2_0 * rhs.r0_2 + lhs.r2_1 * rhs.r1_2 + lhs.r2_2 * rhs.r2_2;

        return {_r0_0, _r0_1, _r0_2,
                _r1_0, _r1_1, _r1_2,
                _r2_0, _r2_1, _r2_2};
    }

    __host__ __device__
    inline Mat33f operator*(const Vec3f &lhs, const Vec3f &rhs) {
        float _r0_0 = lhs.x * rhs.x;
        float _r0_1 = lhs.x * rhs.y;
        float _r0_2 = lhs.x * rhs.z;

        float _r1_0 = lhs.y * rhs.x;
        float _r1_1 = lhs.y * rhs.y;
        float _r1_2 = lhs.y * rhs.z;

        float _r2_0 = lhs.z * rhs.x;
        float _r2_1 = lhs.z * rhs.y;
        float _r2_2 = lhs.z * rhs.z;

        return {_r0_0, _r0_1, _r0_2,
                _r1_0, _r1_1, _r1_2,
                _r2_0, _r2_1, _r2_2};
    }

    __host__ __device__
    inline Mat33f operator*(const Mat33f &lhs, float n) {
        float _r0_0 = lhs.r0_0 * n;
        float _r0_1 = lhs.r0_1 * n;
        float _r0_2 = lhs.r0_2 * n;

        float _r1_0 = lhs.r1_0 * n;
        float _r1_1 = lhs.r1_1 * n;
        float _r1_2 = lhs.r1_2 * n;

        float _r2_0 = lhs.r2_0 * n;
        float _r2_1 = lhs.r2_1 * n;
        float _r2_2 = lhs.r2_2 * n;

        return {_r0_0, _r0_1, _r0_2,
                _r1_0, _r1_1, _r1_2,
                _r2_0, _r2_1, _r2_2};
    }

    __host__ __device__
    inline Mat33f operator*(float n, const Mat33f &rhs) {
        return rhs * n;
    }

    __host__ __device__
    inline Mat33f operator/(const Mat33f &lhs, const float n) {
        float _r0_0 = lhs.r0_0 / n;
        float _r0_1 = lhs.r0_1 / n;
        float _r0_2 = lhs.r0_2 / n;

        float _r1_0 = lhs.r1_0 / n;
        float _r1_1 = lhs.r1_1 / n;
        float _r1_2 = lhs.r1_2 / n;

        float _r2_0 = lhs.r2_0 / n;
        float _r2_1 = lhs.r2_1 / n;
        float _r2_2 = lhs.r2_2 / n;

        return {_r0_0, _r0_1, _r0_2,
                _r1_0, _r1_1, _r1_2,
                _r2_0, _r2_1, _r2_2};
    }

// Vec2f
    __host__ __device__
    inline Vec2f operator+(const Vec2f &lhs, const Vec2f &rhs) {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }

    __host__ __device__
    inline Vec2f operator-(const Vec2f &lhs, const Vec2f &rhs) {
        return {lhs.x - rhs.x, lhs.y - rhs.y};
    }

    __host__ __device__
    inline Vec2f operator*(const Vec2f &vec, float n) {
        return {vec.x * n, vec.y * n};
    }

    __host__ __device__
    inline Vec2f operator*(float n, const Vec2f &vec) {
        return vec * n;
    }

    __host__ __device__
    inline Vec2f operator/(const Vec2f &vec, float n) {
        return {vec.x / n, vec.y / n};
    }

    __host__ __device__
    inline Vec2f operator*(const Mat22f &lhs, const Vec2f &rhs) {
        float _x = lhs.r0_0 * rhs.x + lhs.r0_1 * rhs.y;
        float _y = lhs.r1_0 * rhs.x + lhs.r1_1 * rhs.y;
        return {_x, _y};
    }

    __host__ __device__
    inline void operator+=(Vec2f &lhs, const Vec2f &rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
    }

    __host__ __device__
    inline void operator-=(Vec2f &lhs, const Vec2f &rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
    }

    __host__ __device__
    inline void operator*=(Vec2f &lhs, float n) {
        lhs.x *= n;
        lhs.y *= n;
    }

    __host__ __device__
    inline void operator/=(Vec2f &lhs, float n) {
        lhs.x /= n;
        lhs.y /= n;
    }

    __host__ __device__
    inline float dot(const Vec2f &lhs, const Vec2f &rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y;
    }

// Vec3f
    __host__ __device__    inline bool operator>(const Vec3f &lhs, const Vec3f &rhs) {
        return lhs.length() > rhs.length();
    }

    __host__ __device__    inline bool operator<(const Vec3f &lhs, const Vec3f &rhs) {
        return lhs.length() < rhs.length();
    }

    __host__ __device__
    inline Vec3f operator+(const Vec3f &lhs, const Vec3f &rhs) {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }

    __host__ __device__
    inline Vec3f operator-(const Vec3f &lhs, const Vec3f &rhs) {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }

    __host__ __device__
    inline Vec3f operator*(const Vec3f &vec, float n) {
        return {vec.x * n, vec.y * n, vec.z * n};
    }

    __host__ __device__
    inline Vec3f operator*(float n, const Vec3f &vec) {
        return vec * n;
    }

    __host__ __device__
    inline Vec3f operator/(const Vec3f &vec, float n) {
        return {vec.x / n, vec.y / n, vec.z / n};
    }

    __host__ __device__
    inline Vec3f operator*(const Mat33f &lhs, const Vec3f &rhs) {
        float _x = lhs.r0_0 * rhs.x + lhs.r0_1 * rhs.y + lhs.r0_2 * rhs.z;
        float _y = lhs.r1_0 * rhs.x + lhs.r1_1 * rhs.y + lhs.r1_2 * rhs.z;
        float _z = lhs.r2_0 * rhs.x + lhs.r2_1 * rhs.y + lhs.r2_2 * rhs.z;
        return {_x, _y, _z};
    }

    __host__ __device__
    inline bool operator==(Vec3f &lhs, const Vec3f &rhs) {
        return lhs.x == rhs.x &&
               lhs.y == rhs.y &&
               lhs.z == rhs.z;
    }

    __host__ __device__
    inline void operator+=(Vec3f &lhs, const Vec3f &rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
    }

    __host__ __device__
    inline void operator-=(Vec3f &lhs, const Vec3f &rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
    }

    __host__ __device__
    inline void operator*=(Vec3f &lhs, float n) {
        lhs.x *= n;
        lhs.y *= n;
        lhs.z *= n;
    }

    __host__ __device__
    inline void operator/=(Vec3f &lhs, float n) {
        lhs.x /= n;
        lhs.y /= n;
        lhs.z /= n;
    }

    __host__ __device__
    inline float dot(const Vec3f &lhs, const Vec3f &rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    __host__ __device__
    inline float dot(const Mat33f &lhs, const Mat33f &rhs) {
        return lhs.r0_0 * rhs.r0_0 + lhs.r0_1 * rhs.r0_1 + lhs.r0_2 * rhs.r0_2 +
               lhs.r1_0 * rhs.r1_0 + lhs.r1_1 * rhs.r1_1 + lhs.r1_2 * rhs.r1_2 +
               lhs.r2_0 * rhs.r2_0 + lhs.r2_1 * rhs.r2_1 + lhs.r2_2 * rhs.r2_2;
    }

// Vec3i
    __host__ __device__
    inline Vec3i operator+(const Vec3i &lhs, const Vec3i &rhs) {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }

    __host__ __device__
    inline Vec3i operator-(const Vec3i &lhs, const Vec3i &rhs) {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }

    __host__ __device__
    inline Vec3i operator*(const Vec3i &vec, int n) {
        return {vec.x * n, vec.y * n};
    }

    __host__ __device__
    inline Vec3i operator*(int n, const Vec3i &vec) {
        return vec * n;
    }

    __host__ __device__
    inline Vec3i operator/(const Vec3i &vec, int n) {
        return {vec.x / n, vec.y / n, vec.z / n};
    }

    __host__ __device__
    inline void operator+=(Vec3i &lhs, const Vec3i &rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
    }

    __host__ __device__
    inline void operator-=(Vec3i &lhs, const Vec3i &rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
    }

    __host__ __device__
    inline void operator*=(Vec3i &lhs, float n) {
        lhs.x *= n;
        lhs.y *= n;
        lhs.z *= n;
    }

    __host__ __device__
    inline void operator/=(Vec3i &lhs, float n) {
        lhs.x /= n;
        lhs.y /= n;
        lhs.z /= n;
    }

    __host__ __device__
    inline int dot(const Vec3i &lhs, const Vec3i &rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

// Vec3ui
    __host__ __device__
    inline Vec3ui operator+(const Vec3ui &lhs, const Vec3ui &rhs) {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }

    __host__ __device__
    inline Vec3ui operator-(const Vec3ui &lhs, const Vec3ui &rhs) {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }

    __host__ __device__
    inline Vec3ui operator*(const Vec3ui &vec, int n) {
        return {vec.x * n, vec.y * n};
    }

    __host__ __device__
    inline Vec3ui operator*(int n, const Vec3ui &vec) {
        return vec * n;
    }

    __host__ __device__
    inline Vec3ui operator/(const Vec3ui &vec, int n) {
        return {vec.x / n, vec.y / n, vec.z / n};
    }

    __host__ __device__
    inline void operator+=(Vec3ui &lhs, const Vec3ui &rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
    }

    __host__ __device__
    inline void operator-=(Vec3ui &lhs, const Vec3ui &rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
    }

    __host__ __device__
    inline void operator*=(Vec3ui &lhs, float n) {
        lhs.x *= n;
        lhs.y *= n;
        lhs.z *= n;
    }

    __host__ __device__
    inline void operator/=(Vec3ui &lhs, float n) {
        lhs.x /= n;
        lhs.y /= n;
        lhs.z /= n;
    }

    __host__ __device__
    inline int dot(const Vec3ui &lhs, const Vec3ui &rhs) {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

// Vec* ops
    __host__ __device__
    inline Vec3f operator+(const Vec3f &lhs, const Vec3i &rhs) {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }

    __host__ __device__
    inline Vec3f operator-(const Vec3f &lhs, const Vec3i &rhs) {
        return {lhs.x - static_cast<float>(rhs.x), lhs.y - static_cast<float>(rhs.y),
                lhs.z - static_cast<float>(rhs.z)};
    }

    __host__ __device__
    inline Vec3f operator+(const Vec3i &lhs, const Vec3f &rhs) {
        return rhs + lhs;
    }

    __host__ __device__
    inline Vec3f operator-(const Vec3i &lhs, const Vec3f &rhs) {
        return {static_cast<float>(lhs.x) - rhs.x, static_cast<float>(lhs.y) - rhs.y,
                static_cast<float>(lhs.z) - rhs.z};
    }

    __host__ __device__
    inline Mat33f outer_product(const float3 &lhs, const float3 &rhs) {
        float _r0_0 = lhs.x * rhs.x;
        float _r0_1 = lhs.x * rhs.y;
        float _r0_2 = lhs.x * rhs.z;

        float _r1_0 = lhs.y * rhs.x;
        float _r1_1 = lhs.y * rhs.y;
        float _r1_2 = lhs.y * rhs.z;

        float _r2_0 = lhs.z * rhs.x;
        float _r2_1 = lhs.z * rhs.y;
        float _r2_2 = lhs.z * rhs.z;

        return {_r0_0, _r0_1, _r0_2,
                _r1_0, _r1_1, _r1_2,
                _r2_0, _r2_1, _r2_2};
    }

    __host__ __device__
    inline float3 operator*(const Mat33f &lhs, const float3 &rhs) {
        float _x = lhs.r0_0 * rhs.x + lhs.r0_1 * rhs.y + lhs.r0_2 * rhs.z;
        float _y = lhs.r1_0 * rhs.x + lhs.r1_1 * rhs.y + lhs.r1_2 * rhs.z;
        float _z = lhs.r2_0 * rhs.x + lhs.r2_1 * rhs.y + lhs.r2_2 * rhs.z;
        return {_x, _y, _z};
    }
}

#endif //VT_PHYSICS_MATRIX_HPP
