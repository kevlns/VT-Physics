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

        Object(long long id);

        ~Object();

        void transferTo(ObjectType type);

        json &getObjectComponentConfig();

        void attachSpecificSolverObjectComponentConfig(json specificSolverObjectComponentConfig);

        json &getSolverObjectComponentConfig();

        ObjectTypeComponent *getObjectComponent();

        virtual long long getID() const;

        virtual void rename(std::string newName);

        virtual std::string getName() const;

        virtual bool update();

        void reset();

    private:
        long long m_id;
        std::string m_name;
        ObjectTypeComponent *m_objectTypeComponent{nullptr};
        json m_objectComponentConfig{};
        json m_solverSpecificComponentConfig{};
    };

}

#endif //VT_PHYSICS_OBJECT_HPP
