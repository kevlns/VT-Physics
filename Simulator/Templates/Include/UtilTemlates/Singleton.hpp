/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_SINGLETON_HPP
#define VT_PHYSICS_SINGLETON_HPP

namespace VANT {

    template<typename T>
    class Singleton {
    public:
        static T& getInstance() {
            static T instance;
            return instance;
        }
    };
}

#endif //VT_PHYSICS_SINGLETON_HPP
