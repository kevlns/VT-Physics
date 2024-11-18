#include "Manager/SolverManager.hpp"

#include "Logger/Logger.hpp"
#include "Solvers/PBF/PBFSolver.hpp"
#include "Solvers/DFSPH/DFSPHSolver.hpp"
#include "Solvers/IMM/IMMSolver.hpp"
#include "Solvers/IMM-CT/IMMCTSolver.hpp"

namespace VT_Physics {

    SolverManager::SolverManager() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        m_cuda_threadNum_per_block = prop.maxThreadsPerBlock;
    }

    Solver *SolverManager::createSolver(eSolverType solverType) {
        Solver *solver{nullptr};
        switch (solverType) {
            case eSolverType::PBF:
                solver = new pbf::PBFSolver(m_cuda_threadNum_per_block);
                m_solvers.push_back(solver);
                break;
            case eSolverType::DFSPH:
                solver = new dfsph::DFSPHSolver(m_cuda_threadNum_per_block);
                m_solvers.push_back(solver);
                break;
            case eSolverType::IMM:
                solver = new imm::IMMSolver(m_cuda_threadNum_per_block);
                m_solvers.push_back(solver);
                break;
            case eSolverType::IMMCT:
                solver = new immct::IMMCTSolver(m_cuda_threadNum_per_block);
                m_solvers.push_back(solver);
                break;
        }
        return solver;
    }

    void SolverManager::clear() {
        for (auto solver: m_solvers) {
            solver->destroy();
            delete solver;
        }
        m_solvers.clear();
    }

}