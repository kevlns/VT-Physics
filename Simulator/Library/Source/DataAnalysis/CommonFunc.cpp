#include "DataAnalysis/CommonFunc.hpp"

#include <vector>
#include <cuda_runtime.h>

#include "Core/Math/helper_math_cu11.6.h"

namespace VT_Physics {

    template<typename T>
    T cal_mean(T *d_ptr, unsigned raw_size, unsigned target_size, unsigned target_start) {
        std::vector<T> data(raw_size);
        cudaMemcpy(data.data(), d_ptr, raw_size * sizeof(T), cudaMemcpyDeviceToHost);

        T mean = data[0] * 0;
        for (int i = target_start; i < target_start + target_size; ++i) {
            mean += data[i];
        }

        return mean / target_size;
    }

    // explicit template instance
    template float cal_mean<float>(float *d_ptr, unsigned raw_size, unsigned target_size, unsigned target_start);

    template unsigned
    cal_mean<unsigned>(unsigned *d_ptr, unsigned raw_size, unsigned target_size, unsigned target_start);

    template float3 cal_mean<float3>(float3 *d_ptr, unsigned raw_size, unsigned target_size, unsigned target_start);

}