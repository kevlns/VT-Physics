#include "Manager/Simulator.hpp"

#include <iostream>

using namespace VT_Physics;

int main() {

    auto VT_Simulator = Simulator::getInstance();
    auto solverManager = VT_Simulator.getSolverManager();
    auto objectManager = VT_Simulator.getObjectManager();

    auto obj = objectManager.createObject(Particle_Box);
    auto &config = obj->getObjectComponentConfig();

    JsonHandler::saveJson(std::string(VP_CONFIG_DIR) + "ParticleBoxConfigTemplate.json",
                          config);

    Simulator::terminate();
}
