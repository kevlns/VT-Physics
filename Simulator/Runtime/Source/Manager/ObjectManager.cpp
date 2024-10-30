#include "Manager/ObjectManager.hpp"

namespace VT_Physics {

    ObjectManager::~ObjectManager() {
        for (auto obj: m_objects)
            delete obj;
    }

    Object *ObjectManager::createObject(ObjectType type) {
        Object *obj = new Object(m_objectNum++);
        obj->transferTo(type);
        m_objects.push_back(obj);
        return obj;
    }

}