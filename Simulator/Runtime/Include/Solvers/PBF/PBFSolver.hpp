/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_PBFSOLVER_HPP
#define VT_PHYSICS_PBFSOLVER_HPP

#include <vector>
#include <set>

#include "Framework/Solver.hpp"
#include "PBFrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"

namespace VT_Physics::pbf {

    inline const std::vector<std::string> PBFConfigRequiredKeys = {
            "animationTime",
            "timeStep",
            "particleRadius",
            "simSpaceLB",
            "simSpaceSize",
            "maxNeighborNum",
            "iterationNum",
            "XSPH_k",
            "fPartRestDensity",
            "bPartRestDensity"
    };

    inline const std::vector<std::string> PBFConfigOptionalKeys = {
            "enable",
            "gravity",
    };

    inline const std::vector<std::string> PBFSolverObjectComponentConfigRequiredKeys = {
            "solverType",
            "exportFlag",
            "velocityStart",
            "colorStart",
    };

    inline const std::set<uint8_t> PBFSolverSupportedMaterials = {
            EPM_FLUID,
            EPM_BOUNDARY
    };

    class PBFSolver : public Solver {
    public:
        PBFSolver() = delete;

        PBFSolver(uint32_t cudaThreadSize);

        virtual ~PBFSolver() override = default;

        virtual json getSolverConfigTemplate() const override;

        virtual bool setConfig(json config) override;

        virtual bool setConfigByFile(std::string config_file) override;

        virtual json getSolverObjectComponentConfigTemplate() override;

        virtual bool initialize() override;

        virtual bool run() override;

        virtual bool tickNsteps(uint32_t n) override;

        virtual bool attachObject(Object *obj) override;

        virtual bool attachObjects(std::vector<Object *> objs) override;

        virtual bool reset() override;

        virtual void destroy() override;

    protected:
        virtual bool tick() override;

        virtual bool checkConfig() const override;

    private:
        void exportData();

    private:
        json m_configData;
        Data *m_host_data{nullptr};
        Data *m_device_data{nullptr};
        bool m_isInitialized{false};
        bool m_isCrashed{false};
        uint32_t m_frameCount{0};
        uint32_t m_outputFrameCount{0};
        bool m_doExportFlag{false};
        std::vector<Object *> m_attached_objs;
        UGNS::UniformGirdNeighborSearcher m_neighborSearcher;

        std::vector<float3> m_host_pos;
        std::vector<float3> m_host_vel;
        std::vector<int> m_host_mat;
        std::vector<float3> m_host_color;
    };
}

#endif //VT_PHYSICS_PBFSOLVER_HPP
