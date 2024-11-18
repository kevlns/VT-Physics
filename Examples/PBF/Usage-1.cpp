#include "Manager/Simulator.hpp"

#include <iostream>

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    /**
     * Create a PBF solver and set solver config ==================================================
     */
    auto pbfSolver = solverManager->createSolver(PBF);
    auto pbf_config = pbfSolver->getSolverConfigTemplate();
    pbf_config["PBF"]["Required"]["animationTime"] = 5.0f;
    pbf_config["PBF"]["Required"]["timeStep"] = 0.01f;
    pbf_config["PBF"]["Required"]["particleRadius"] = 0.05f;
    pbf_config["PBF"]["Required"]["simSpaceLB"] = {-10, -10, -10};
    pbf_config["PBF"]["Required"]["simSpaceSize"] = {20, 20, 20};
    pbf_config["PBF"]["Required"]["iterationNum"] = 25;
    pbf_config["PBF"]["Required"]["XSPH_k"] = 0.01;
    pbf_config["PBF"]["Required"]["fPartRestDensity"] = 1000.f;
    pbf_config["PBF"]["Required"]["bPartRestDensity"] = 1500.f;
    pbf_config["EXPORT"]["Common"]["exportTargetDir"] = "F:\\DataSet.Research\\VP-Examples\\PBF";
    pbf_config["EXPORT"]["SolverRequired"]["enable"] = true;
    pbf_config["EXPORT"]["SolverRequired"]["exportFps"] = 35;
    pbfSolver->setConfig(pbf_config);

    /**
     * Get object component config for PBF simulator ===============================================
     */
    auto pbfObjComponentConfig = pbfSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for PBF simulator ===================================================
     */
    auto cube = objectManager->createObject(Particle_Cube);
    auto &cube_config = cube->getObjectComponentConfig();
    cube_config["epmMaterial"] = EPM_FLUID;
    cube_config["particleRadius"] = pbf_config["PBF"]["Required"]["particleRadius"];
    cube_config["lb"] = {-1, -1, -1};
    cube_config["size"] = {2, 2, 2};
    auto cube_pbfObjComponentConfig = pbfObjComponentConfig;
    cube_pbfObjComponentConfig["exportFlag"] = true;
    cube_pbfObjComponentConfig["velocityStart"] = {0, 0, 0};
    cube_pbfObjComponentConfig["accelerationStart"] = {0, 0, 0};
    cube_pbfObjComponentConfig["colorStart"] = {0, 1, 0};
    cube->attachSpecificSolverObjectComponentConfig(cube_pbfObjComponentConfig);
    cube->update();

    auto box = objectManager->createObject(Particle_Box);
    auto &box_config = box->getObjectComponentConfig();
    box_config["epmMaterial"] = EPM_BOUNDARY;
    box_config["particleRadius"] = pbf_config["PBF"]["Required"]["particleRadius"];
    box_config["lb"] = {-3, -2.5, -3};
    box_config["size"] = {6, 6, 6};
    box_config["layerNum"] = 1;
    auto box_pbfObjComponentConfig = pbfObjComponentConfig;
    box_pbfObjComponentConfig["exportFlag"] = false;
    box_pbfObjComponentConfig["velocityStart"] = {0, 0, 0};
    box_pbfObjComponentConfig["accelerationStart"] = {0, 0, 0};
    box_pbfObjComponentConfig["colorStart"] = {0, 0, 0};
    box->attachSpecificSolverObjectComponentConfig(box_pbfObjComponentConfig);
    box->update();

    /**
     * Attach objects to PBF solver ================================================================
     */
    pbfSolver->attachObject(cube);
    pbfSolver->attachObject(box);

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

    SYS_PAUSE();
}
