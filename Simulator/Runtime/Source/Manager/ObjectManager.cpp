#include "Manager/ObjectManager.hpp"

#include <exception>

namespace VT_Physics {

    Object *ObjectManager::createObject(ObjectType type) {
        Object *obj = new Object(m_objectNum++);
        obj->transferTo(type);
        m_objects.push_back(obj);
        return obj;
    }

    void ObjectManager::clear() {
        for (auto &obj: m_objects) {
            obj->reset();
            delete obj;
        }
        m_objects.clear();
    }

    std::vector<Object *> ObjectManager::createObjectsFromJson(json allConfiguration) {
        std::vector<Object *> ret;
        auto &objectsConfig = allConfiguration["OBJECTS"];
        if (objectsConfig.empty()) {
            LOG_ERROR("ObjectManager::createObjectsFromJson: objectsConfig is empty")
            return {};
        }

        for (const auto &objConfigJson: objectsConfig.get<std::vector<json>>()) {
            auto objType = static_cast<ObjectType>(objConfigJson["objTransferType"].get<uint8_t>());
            auto obj = createObject(objType);
            auto &config = obj->getObjectComponentConfig();
            config = objConfigJson["ObjectComponentConfig"];
            obj->attachSpecificSolverObjectComponentConfig(objConfigJson["SolverObjectComponentConfig"]);
            obj->update();
            ret.push_back(obj);
        }

        return std::move(ret);
    }

}