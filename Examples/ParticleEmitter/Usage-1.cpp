#include "Manager/Simulator.hpp"

#include <iostream>

using namespace VT_Physics;

int main() {
    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    auto sceneConfig = JsonHandler::loadJson(
            std::string(VP_EXAMPLES_DIR) + "ParticleEmitter/Scene-1/Configuration.json");
    /**
     * Create a PBF solver and set solver config ==================================================
     */
    auto solver = solverManager->createSolver(MCT);
    solver->setConfig(sceneConfig);

    /**
     * Create particle objects for PBF simulator ===================================================
     */
    auto objs = objectManager->createObjectsFromJson(sceneConfig);

    /**
     * Attach objects to PBF solver ================================================================
     */
    solver->attachObjects(objs);

    /**
     * Run simulation ==============================================================================
     */
    solver->run();
//    solver->tickNsteps(10);

    /**
     * Terminate simulator ========================================================================
     */
    VT_Simulator.terminate();
    Simulator::clean();

    SYS_PAUSE();
}
