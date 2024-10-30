#include "Framework/Object.hpp"

namespace VT_Physics {

    Object::Object(long long int id) {
        m_id = id;
        LOG_INFO("Object: " + std::to_string(m_id) + " Created.")
    }

    Object::~Object() {
        if (m_objectTypeComponent)
            delete m_objectTypeComponent;
        LOG_INFO("Object: " + std::to_string(m_id) + " Destroyed.");
    }

    void Object::transferTo(VT_Physics::ObjectType type) {
        for (const auto &component: componentTemplates) {
            if (type == component->getType()) {
                if (m_objectTypeComponent)
                    delete m_objectTypeComponent;

                m_objectTypeComponent = component->clone();
                m_objectComponentConfig = m_objectTypeComponent->getConfig();
                break;
            }
        }
    }

    json &Object::getObjectComponentConfig() {
        return m_objectComponentConfig;
    }

    ObjectTypeComponent *Object::getObjectComponent() {
        return m_objectTypeComponent;
    }

    bool Object::update() {
        m_objectTypeComponent->update(m_objectComponentConfig);
        return true;
    }

    void Object::reset() {
        m_objectComponentConfig = {};
        if (m_objectTypeComponent)
            delete m_objectTypeComponent;
    }

}