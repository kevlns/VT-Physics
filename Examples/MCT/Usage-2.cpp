#include "Manager/Simulator.hpp"

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    auto sceneConfig = JsonHandler::loadJson(
            "F:\\DataSet.Research\\Multimode-CT_with_Porous\\experiments\\Vise_Ducks\\SCENE_[1phase_vis_100_rel_0.001]\\Configuration.json");

    /**
     * Create a PBF solver and set solver config ==================================================
     */
    auto immctSolver = solverManager->createSolver(MCT);
    immctSolver->setConfig(sceneConfig);

    /**
     * Get object component config for PBF simulator ===============================================
     */
    auto pbfObjComponentConfig = immctSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for PBF simulator ===================================================
     */
    auto objs = objectManager->createObjectsFromJson(sceneConfig);

    /**
     * Attach objects to PBF solver ================================================================
     */
    immctSolver->attachObjects(objs);

    /**
     * Run simulation ==============================================================================
     */
    immctSolver->run();
//    immctSolver->tickNsteps(10);

    /**
     * Terminate simulator ========================================================================
     */
    VT_Simulator.terminate();
    Simulator::clean();

    SYS_PAUSE();
}
