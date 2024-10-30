#include "Framework/Solver.hpp"

namespace VT_Physics {

    Solver::~Solver() {}

    json Solver::getDefaultConfig() const{
        return {};
    }

    bool Solver::setConfig(json config) {
        return false;
    }

    bool Solver::setConfigByFile(std::string solver_config) {
        return false;
    }

    bool Solver::run(float simTime) {
        return false;
    }

    bool Solver::tickNsteps(uint32_t n) {
        return false;
    }

    bool Solver::attachObject(Object *obj) {
        return false;
    }

    bool Solver::initialize() {
        return false;
    }

    bool Solver::reset() {
        return false;
    }

    void Solver::destroy() {}

    bool Solver::tick() {
        return false;
    }

    bool Solver::checkConfig() const {
        return false;
    }
}