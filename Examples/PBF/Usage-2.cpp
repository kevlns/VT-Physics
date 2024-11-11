#include "Manager/Simulator.hpp"

#include <iostream>

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    auto sceneConfig = JsonHandler::loadJson(
            "G:\\Projects\\Projects.SoEngine\\VT-Physics\\Examples\\PBF\\Scene-1\\Configuration.json");

    /**
     * Create a PBF solver and set solver config ==================================================
     */
    auto pbfSolver = solverManager.createSolver(PBF);
    pbfSolver->setConfig(sceneConfig);

    /**
     * Get object component config for PBF simulator ===============================================
     */
    auto pbfObjComponentConfig = pbfSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for PBF simulator ===================================================
     */
    auto objs = objectManager.createObjectsFromJson(sceneConfig);

    /**
     * Attach objects to PBF solver ================================================================
     */
    pbfSolver->attachObjects(objs);

    /**
     * Run simulation ==============================================================================
     */
    pbfSolver->run();
//    pbfSolver->tickNsteps(10);

    /**
     * Terminate simulator ========================================================================
     */
    VT_Simulator.terminate();
    Simulator::clean();

    system("pause");
}
