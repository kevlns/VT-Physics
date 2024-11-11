/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_SOLVERMANAGER_HPP
#define VT_PHYSICS_SOLVERMANAGER_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include <tuple>
#include <any>

#include "Framework/Solver.hpp"

namespace VT_Physics {

    class SolverManager {
    public:
        SolverManager();

        Solver *createSolver(eSolverType solverType);

        void clear();

    private:
        uint32_t m_cuda_threadNum_per_block{1024};
        std::vector<Solver *> m_solvers;

    };


}

#endif //VT_PHYSICS_SOLVERMANAGER_HPP
