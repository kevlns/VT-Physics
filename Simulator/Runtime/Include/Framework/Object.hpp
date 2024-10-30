/**
 * @brief Todo
 * @date 2024/10/29
 */

#ifndef VT_PHYSICS_OBJECT_HPP
#define VT_PHYSICS_OBJECT_HPP

#include <vector>

#include "ObjectComponents.hpp"

namespace VT_Physics {

    class Object {
    public:

        Object() = default;

        Object(long long id);

        ~Object();

        void transferTo(ObjectType type);

        json &getObjectComponentConfig();

        ObjectTypeComponent *getObjectComponent();

        bool update();

        void reset();

    private:
        long long m_id;
        ObjectTypeComponent *m_objectTypeComponent{nullptr};
        json m_objectComponentConfig;
    };

}

#endif //VT_PHYSICS_OBJECT_HPP
