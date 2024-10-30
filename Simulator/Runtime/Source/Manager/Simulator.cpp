#include "Manager/Simulator.hpp"

namespace VT_Physics {

    Simulator &Simulator::getInstance() {
        static Simulator instance;
        return instance;
    }

    ObjectManager &Simulator::getObjectManager() {
        return m_objectManager;
    }

    SolverManager &Simulator::getSolverManager() {
        return m_solverManager;
    }

    void Simulator::terminate() {
        // free static object component resources
        for (auto ptr: componentTemplates)
            delete ptr;
    }
}