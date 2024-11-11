/**
 * @brief Todo
 * @date 2024/10/29
 */

#ifndef VT_PHYSICS_OBJECTMANAGER_HPP
#define VT_PHYSICS_OBJECTMANAGER_HPP

#include "Framework/Object.hpp"

namespace VT_Physics {

    class ObjectManager {
    public:
        std::vector<Object *> createObjectsFromJson(json allConfiguration);

        Object *createObject(ObjectType type = OBJ_NULL_TYPE);

        void clear();

    private:
        long long m_objectNum{0};
        std::vector<Object *> m_objects;
    };

}

#endif //VT_PHYSICS_OBJECTMANAGER_HPP
