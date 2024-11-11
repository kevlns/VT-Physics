/**
 * @brief Todo
 * @date 2024/11/11
 */

#ifndef VT_PHYSICS_COMMONFUNC_HPP
#define VT_PHYSICS_COMMONFUNC_HPP

namespace VT_Physics {

    template<typename T>
    T cal_mean(T *d_ptr, unsigned raw_size, unsigned target_size, unsigned target_start = 0);

}

#endif //VT_PHYSICS_COMMONFUNC_HPP
