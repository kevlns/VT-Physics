/**
 * @brief Todo
 * @date 2024/10/29
 */

#ifndef VT_PHYSICS_SIMULATOR_HPP
#define VT_PHYSICS_SIMULATOR_HPP

#include "Manager/ObjectManager.hpp"
#include "Manager/SolverManager.hpp"
#include "JSON/JSONHandler.hpp"
#include "Model/ModelHandler.hpp"
#include "Model/ExportUtil.hpp"
#include "ConfigTemplates/RealPath.h"

namespace VT_Physics {

    class Simulator final {
    public:
        ~Simulator() = default;

        static Simulator &getInstance();

        ObjectManager &getObjectManager();

        SolverManager &getSolverManager();

        static void terminate();

    private:
        ObjectManager m_objectManager;
        SolverManager m_solverManager;
    };

}

#endif //VT_PHYSICS_SIMULATOR_HPP
