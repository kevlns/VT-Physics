#include "Manager/Simulator.hpp"

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    /**
     * Create a DFSPH solver and set solver config ==================================================
     */
    auto dfsphSolver = solverManager->createSolver(DFSPH);
    auto dfsph_config = dfsphSolver->getSolverConfigTemplate();
    dfsph_config["DFSPH"]["Required"]["animationTime"] = 5.0f;
    dfsph_config["DFSPH"]["Required"]["timeStep"] = 0.01f;
    dfsph_config["DFSPH"]["Required"]["particleRadius"] = 0.05f;
    dfsph_config["DFSPH"]["Required"]["simSpaceLB"] = {-10, -10, -10};
    dfsph_config["DFSPH"]["Required"]["simSpaceSize"] = {20, 20, 20};
    dfsph_config["DFSPH"]["Required"]["fPartRestDensity"] = 1000.f;
    dfsph_config["DFSPH"]["Required"]["bPartRestDensity"] = 1500.f;
    dfsph_config["DFSPH"]["Required"]["fPartRestViscosity"] = 0.1f;
    dfsph_config["DFSPH"]["Required"]["divFreeThreshold"] = 1e-4f;
    dfsph_config["DFSPH"]["Required"]["incompThreshold"] = 1e-4f;
    dfsph_config["DFSPH"]["Required"]["surfaceTensionCoefficient"] = 0.001;
    dfsph_config["EXPORT"]["Common"]["exportTargetDir"] = "F:\\DataSet.Research\\VP-Examples\\DFSPH";
    dfsph_config["EXPORT"]["SolverRequired"]["enable"] = false;
    dfsph_config["EXPORT"]["SolverRequired"]["exportFps"] = 35;
    dfsphSolver->setConfig(dfsph_config);
    
    /**
     * Get object component config for DFSPH simulator ===============================================
     */
    auto dfsphObjComponentConfig = dfsphSolver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for DFSPH simulator ===================================================
     */
    auto cube = objectManager->createObject(Particle_Cube);
    auto &cube_config = cube->getObjectComponentConfig();
    cube_config["epmMaterial"] = EPM_FLUID;
    cube_config["particleRadius"] = dfsph_config["DFSPH"]["Required"]["particleRadius"];
    cube_config["lb"] = {-1, -1, -1};
    cube_config["size"] = {2, 2, 2};
    auto cube_dfsphObjComponentConfig = dfsphObjComponentConfig;
    cube_dfsphObjComponentConfig["exportFlag"] = true;
    cube_dfsphObjComponentConfig["velocityStart"] = {0, 0, 0};
    cube_dfsphObjComponentConfig["colorStart"] = {0, 1, 0};
    cube->attachSpecificSolverObjectComponentConfig(cube_dfsphObjComponentConfig);
    cube->update();

    auto box = objectManager->createObject(Particle_Box);
    auto &box_config = box->getObjectComponentConfig();
    box_config["epmMaterial"] = EPM_BOUNDARY;
    box_config["particleRadius"] = dfsph_config["DFSPH"]["Required"]["particleRadius"];
    box_config["lb"] = {-3, -2.5, -3};
    box_config["size"] = {6, 6, 6};
    box_config["layerNum"] = 1;
    auto box_dfsphObjComponentConfig = dfsphObjComponentConfig;
    box_dfsphObjComponentConfig["exportFlag"] = false;
    box_dfsphObjComponentConfig["velocityStart"] = {0, 0, 0};
    box_dfsphObjComponentConfig["accelerationStart"] = {0, 0, 0};
    box_dfsphObjComponentConfig["colorStart"] = {0, 0, 0};
    box->attachSpecificSolverObjectComponentConfig(box_dfsphObjComponentConfig);
    box->update();

    /**
     * Attach objects to DFSPH solver ================================================================
     */
    dfsphSolver->attachObject(cube);
    dfsphSolver->attachObject(box);

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
