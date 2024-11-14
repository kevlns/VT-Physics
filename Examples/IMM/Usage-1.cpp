#include "Manager/Simulator.hpp"

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    /**
     * Create a IMM solver and set solver config ==================================================
     */
    auto immSolver = solverManager->createSolver(IMM);
    auto imm_config = immSolver->getSolverConfigTemplate();
    imm_config["IMM"]["Required"]["animationTime"] = 5.0f;
    imm_config["IMM"]["Required"]["timeStep"] = 0.008f;
    imm_config["IMM"]["Required"]["particleRadius"] = 0.05f;
    imm_config["IMM"]["Required"]["simSpaceLB"] = {-10, -10, -10};
    imm_config["IMM"]["Required"]["simSpaceSize"] = {20, 20, 20};
    imm_config["IMM"]["Required"]["divFreeThreshold"] = 1e-4f;
    imm_config["IMM"]["Required"]["incompThreshold"] = 1e-4f;
    imm_config["IMM"]["Required"]["surfaceTensionCoefficient"] = 0.001;
    imm_config["IMM"]["Required"]["diffusionCoefficientCf"] = 0.15;
    imm_config["IMM"]["Required"]["momentumExchangeCoefficientCd"] = 0.4;
    imm_config["IMM"]["Required"]["phaseRestDensity"] = {1000, 1500, 2000};
    imm_config["IMM"]["Required"]["phaseRestViscosity"] = {0.01, 0.01, 0.01};
    imm_config["IMM"]["Required"]["phaseRestColor"] = {{0, 255, 0},
                                                       {255, 0, 0},
                                                       {0, 0, 255}};
    imm_config["EXPORT"]["Common"]["exportTargetDir"] = "F:\\DataSet.Research\\VP-Examples\\IMM";
    imm_config["EXPORT"]["SolverRequired"]["enable"] = true;
    imm_config["EXPORT"]["SolverRequired"]["exportFps"] = 35;
    immSolver->setConfig(imm_config);

    /**
     * Get object component config for IMM simulator ===============================================
     */
    auto immObjComponentConfig = immSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for IMM simulator ===================================================
     */
    auto cube = objectManager->createObject(Particle_Cube);
    auto &cube_config = cube->getObjectComponentConfig();
    cube_config["epmMaterial"] = EPM_FLUID;
    cube_config["particleRadius"] = imm_config["IMM"]["Required"]["particleRadius"];
    cube_config["lb"] = {-1, -1, -1};
    cube_config["size"] = {2, 2, 2};
    auto cube_immObjComponentConfig = immObjComponentConfig;
    cube_immObjComponentConfig["exportFlag"] = true;
    cube_immObjComponentConfig["velocityStart"] = {-1, 0, 0};
    cube_immObjComponentConfig["phaseFraction"] = {0.2, 0.3, 0.5};
    cube->attachSpecificSolverObjectComponentConfig(cube_immObjComponentConfig);
    cube->update();

    auto box = objectManager->createObject(Particle_Box);
    auto &box_config = box->getObjectComponentConfig();
    box_config["epmMaterial"] = EPM_BOUNDARY;
    box_config["particleRadius"] = imm_config["IMM"]["Required"]["particleRadius"];
    box_config["lb"] = {-2, -2, -2};
    box_config["size"] = {4, 4, 4};
    box_config["layerNum"] = 2;
    auto box_immObjComponentConfig = immObjComponentConfig;
    box_immObjComponentConfig["exportFlag"] = false;
    box_immObjComponentConfig["velocityStart"] = {0, 0, 0};
    box_immObjComponentConfig["phaseFraction"] = {0.0, 0.0, 0.0};
    box->attachSpecificSolverObjectComponentConfig(box_immObjComponentConfig);
    box->update();

    /**
     * Attach objects to IMM solver ================================================================
     */
    immSolver->attachObject(cube);
    immSolver->attachObject(box);

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

    SYS_PAUSE();
}
