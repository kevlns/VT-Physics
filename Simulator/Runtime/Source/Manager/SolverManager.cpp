#include "Manager/SolverManager.hpp"

#include "Logger/Logger.hpp"
#include "Solvers/PBF/PBFSolver.hpp"

namespace VT_Physics {

    SolverManager::SolverManager() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        m_cuda_threadNum_per_block = prop.maxThreadsPerBlock;
    }

    SolverManager::~SolverManager() {
        for (auto &solver: m_solvers) {
            solver->destroy();
        }
    }

    Solver *SolverManager::createSolver(eSolverType solverType) {
        Solver *solver{nullptr};
        switch (solverType) {
            case eSolverType::PBF:
                solver = new pbf::PBFSolver(m_cuda_threadNum_per_block);
                m_solvers.push_back(solver);
                break;
        }
        return solver;
    }

}