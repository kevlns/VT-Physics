#include "Framework/Solver.hpp"

namespace VT_Physics {

    Solver::~Solver() {}

    json Solver::getSolverConfigTemplate() const {
        return {};
    }

    bool Solver::setConfig(json config) {
        return false;
    }

    bool Solver::setConfigByFile(std::string config_file) {
        return false;
    }

    json Solver::getSolverObjectComponentConfigTemplate() {
        return {};
    }

    bool Solver::run() {
        return false;
    }

    bool Solver::tickNsteps(uint32_t n) {
        return false;
    }

    bool Solver::attachObject(Object *obj) {
        return false;
    }

    bool Solver::attachObjects(std::vector<Object *> objs) {
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