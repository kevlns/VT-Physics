#include "Manager/Simulator.hpp"

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    auto sceneConfig = JsonHandler::loadJson(
            std::string(VP_EXAMPLES_DIR) + "DFSPH/Scene-1/Configuration.json");

    /**
     * Create a PBF solver and set solver config ==================================================
     */
    auto dfsphSolver = solverManager->createSolver(DFSPH);
    dfsphSolver->setConfig(sceneConfig);

    /**
     * Get object component config for PBF simulator ===============================================
     */
    auto pbfObjComponentConfig = dfsphSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for PBF simulator ===================================================
     */
    auto objs = objectManager->createObjectsFromJson(sceneConfig);

    /**
     * Attach objects to PBF solver ================================================================
     */
    dfsphSolver->attachObjects(objs);

    /**
     * Run simulation ==============================================================================
     */
    dfsphSolver->run();
//    dfsphSolver->tickNsteps(10);

    /**
     * Terminate simulator ========================================================================
     */
    VT_Simulator.terminate();
    Simulator::clean();

    system("pause");
}
