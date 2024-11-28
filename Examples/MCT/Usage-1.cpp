#include "Manager/Simulator.hpp"

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    /**
     * Create a IMMCT solver and set solver config ==================================================
     */
    auto immctSolver = solverManager->createSolver(MCT);
    auto immct_config = immctSolver->getSolverConfigTemplate();
    immct_config["IMMCT"]["Required"]["animationTime"] = 5.0f;
    immct_config["IMMCT"]["Required"]["timeStep"] = 0.0008f;
    immct_config["IMMCT"]["Required"]["particleRadius"] = 0.05f;
    immct_config["IMMCT"]["Required"]["simSpaceLB"] = {-10, -10, -10};
    immct_config["IMMCT"]["Required"]["simSpaceSize"] = {20, 20, 20};
    immct_config["IMMCT"]["Required"]["divFreeThreshold"] = 1e-4f;
    immct_config["IMMCT"]["Required"]["incompThreshold"] = 1e-4f;
    immct_config["IMMCT"]["Required"]["surfaceTensionCoefficient"] = 0.001;
    immct_config["IMMCT"]["Required"]["diffusionCoefficientCf"] = 0.15;
    immct_config["IMMCT"]["Required"]["momentumExchangeCoefficientCd0"] = 0.5;
    immct_config["IMMCT"]["Required"]["solventViscosity"] = 0.01;
    immct_config["IMMCT"]["Required"]["phaseRestDensity"] = {900, 1000};
    immct_config["IMMCT"]["Required"]["phaseRestColor"] = {{0,   255, 0},
                                                           {255, 0,   0}};
    immct_config["IMMCT"]["Required"]["solutionBasicViscosity"] = 8;
    immct_config["IMMCT"]["Required"]["solutionMaxViscosity"] = 12;
    immct_config["IMMCT"]["Required"]["relaxationTime"] = 0.001;
    immct_config["IMMCT"]["Required"]["shearThinningBasicFactor"] = 0.5;
    immct_config["IMMCT"]["Required"]["rheologicalThreshold"] = 0.4;

    immct_config["EXPORT"]["Common"]["exportTargetDir"] = "F:\\DataSet.Research\\VP-Examples\\MCT";
    immct_config["EXPORT"]["SolverRequired"]["enable"] = true;
    immct_config["EXPORT"]["SolverRequired"]["exportFps"] = 35;
    immctSolver->setConfig(immct_config);

    /**
     * Get object component config for IMMCT simulator ===============================================
     */
    auto immctObjComponentConfig = immctSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for IMMCT simulator ===================================================
     */
    auto cube = objectManager->createObject(Particle_Cube);
    auto &cube_config = cube->getObjectComponentConfig();
    cube_config["epmMaterial"] = EPM_FLUID;
    cube_config["particleRadius"] = immct_config["IMMCT"]["Required"]["particleRadius"];
    cube_config["lb"] = {-1, -1, -1};
    cube_config["size"] = {2, 2, 2};
    auto cube_immctObjComponentConfig = immctObjComponentConfig;
    cube_immctObjComponentConfig["exportFlag"] = true;
    cube_immctObjComponentConfig["velocityStart"] = {0, 0, 0};
    cube_immctObjComponentConfig["phaseFraction"] = {0.3, 0.7};
    cube->attachSpecificSolverObjectComponentConfig(cube_immctObjComponentConfig);
    cube->update();

    auto box = objectManager->createObject(Particle_Box);
    auto &box_config = box->getObjectComponentConfig();
    box_config["epmMaterial"] = EPM_BOUNDARY;
    box_config["particleRadius"] = immct_config["IMMCT"]["Required"]["particleRadius"];
    box_config["lb"] = {-2, -2, -2};
    box_config["size"] = {4, 4, 4};
    box_config["layerNum"] = 1;
    auto box_immctObjComponentConfig = immctObjComponentConfig;
    box_immctObjComponentConfig["exportFlag"] = false;
    box_immctObjComponentConfig["velocityStart"] = {0, 0, 0};
    box_immctObjComponentConfig["phaseFraction"] = {0.0, 0.0};
    box->attachSpecificSolverObjectComponentConfig(box_immctObjComponentConfig);
    box->update();

    /**
     * Attach objects to IMMCT solver ================================================================
     */
    immctSolver->attachObject(cube);
    immctSolver->attachObject(box);

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
