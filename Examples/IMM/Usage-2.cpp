#include "Manager/Simulator.hpp"

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    auto sceneConfig = JsonHandler::loadJson(
            std::string(VP_EXAMPLES_DIR) + "IMM/Scene-1/Configuration.json");

    /**
     * Create a PBF solver and set solver config ==================================================
     */
    auto immSolver = solverManager->createSolver(IMM);
    immSolver->setConfig(sceneConfig);

    /**
     * Get object component config for PBF simulator ===============================================
     */
    auto pbfObjComponentConfig = immSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for PBF simulator ===================================================
     */
    auto objs = objectManager->createObjectsFromJson(sceneConfig);

    /**
     * Attach objects to PBF solver ================================================================
     */
    immSolver->attachObjects(objs);

    /**
     * Run simulation ==============================================================================
     */
    immSolver->run();
//    immSolver->tickNsteps(10);

    /**
     * Terminate simulator ========================================================================
     */
    VT_Simulator.terminate();
    Simulator::clean();

    system("pause");
}
