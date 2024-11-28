#include <utility>

#include "Framework/Object.hpp"

namespace VT_Physics {

    Object::Object(long long int id) {
        m_id = id;
        m_name = "obj_" + std::to_string(m_id);
        LOG_INFO("Object: " + m_name + " Created.")
    }

    Object::~Object() {
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
        m_objectComponentConfig.clear();
        m_solverSpecificComponentConfig.clear();
        if (m_objectTypeComponent) {
            delete m_objectTypeComponent;
            m_objectTypeComponent = nullptr;
        }
    }

    void Object::attachSpecificSolverObjectComponentConfig(json specificSolverObjectComponentConfig) {
        m_solverSpecificComponentConfig = specificSolverObjectComponentConfig;
    }

    json &Object::getSolverObjectComponentConfig() {
        return m_solverSpecificComponentConfig;
    }

    long long Object::getID() const {
        return m_id;
    }

    void Object::rename(std::string newName) {
        m_name = std::move(newName);
    }

    std::string Object::getName() const {
        return m_name;
    }

}