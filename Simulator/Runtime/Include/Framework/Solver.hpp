/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_SOLVER_HPP
#define VT_PHYSICS_SOLVER_HPP

#include <cstdint>
#include <string>

#include "Framework/Object.hpp"
#include "Framework/Material.hpp"
#include "nlohmann/json.hpp"

namespace VT_Physics {

    using json = nlohmann::json;

    typedef enum eSolverType : uint8_t {
        PBF = 0,
        DFSPH = 1,
    } SolverType;

    class Solver {
    public:
        Solver() = default;

        virtual ~Solver() = 0;

        virtual json getSolverConfigTemplate() const = 0;

        virtual bool setConfig(json config) = 0;

        virtual bool setConfigByFile(std::string config_file) = 0;

        virtual json getSolverObjectComponentConfigTemplate() = 0;

        virtual bool run() = 0;

        [[maybe_unused]] virtual bool tickNsteps(uint32_t n) = 0;

        virtual bool attachObject(Object *obj) = 0;

        virtual bool attachObjects(std::vector<Object *> objs) = 0;

        virtual bool initialize() = 0;

        virtual bool reset() = 0;

        virtual void destroy() = 0;

    protected:
        virtual bool tick() = 0;

        virtual bool checkConfig() const = 0;

    private:

    };
}

#endif //VT_PHYSICS_SOLVER_HPP
