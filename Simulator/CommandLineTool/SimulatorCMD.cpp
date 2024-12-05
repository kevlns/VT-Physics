#include "Manager/Simulator.hpp"

#include <set>
#include <string>

using namespace VT_Physics;

inline const std::set<std::string> cmdOptions = {
        "--config",
        "--rns",
        "--h"
};

#define DUMP_HELP() \
    LOG_INFO("\n" \
    "vp usage: vp [OPT-1 OPT_VAL-1 | OPT-2 OPT_VAL-2 | ...]\n "  \
    "Options:\n" \
    "   --config: scene configuration json file.\n" \
    "   --rns   : run N steps of solver.\n");

int main(int argc, char *argv[]) {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    std::string configFile;
    int rns_stepNums = 0;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--h") {
            DUMP_HELP();
            continue;
        }
        if (arg == "--config" && i + 1 < argc) {
            configFile = argv[i + 1];
            continue;
        }
        if (arg == "--rns" && i + 1 < argc) {
            rns_stepNums = std::stoi(argv[i + 1]);
            continue;
        }
    }

    if (configFile.empty())
        return 0;
    auto sceneConfig = JsonHandler::loadJson(configFile);
    if (sceneConfig.empty())
        return 0;

    /**
     * Create the solver and set solver config =====================================================
     */
    Solver *cur_solver{nullptr};
    for (const auto &stmap_iter: solverTypeMapping) {
        if (sceneConfig.contains(stmap_iter.first)) {
            cur_solver = solverManager->createSolver(stmap_iter.second);
        }
    }
    if (!cur_solver) {
        LOG_ERROR("Unsupported Solver Type.");
        return 0;
    }

    /**
     * set Scene Config ============================================================================
     */
    cur_solver->setConfig(sceneConfig);

    /**
     * Get object component config for Solver ======================================================
     */
    auto solverObjComponentConfig = cur_solver->getSolverObjectComponentConfigTemplate();

    /**
     * Create particle objects for simulator =======================================================
     */
    auto objs = objectManager->createObjectsFromJson(sceneConfig);

    /**
     * Attach objects to PBF solver ================================================================
     */
    cur_solver->attachObjects(objs);

    /**
     * Run simulation ==============================================================================
     */
    if (rns_stepNums == 0)
        cur_solver->run();
    else
        cur_solver->tickNsteps(rns_stepNums);

    /**
     * Terminate simulator ========================================================================
     */
    VT_Simulator.terminate();
    Simulator::clean();
}