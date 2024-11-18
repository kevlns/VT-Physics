/**
 * @brief Todo
 * @date 2024/11/11
 */

#ifndef VT_PHYSICS_IMMCTSOLVER_HPP
#define VT_PHYSICS_IMMCTSOLVER_HPP

#include <vector>
#include <set>

#include "Framework/Solver.hpp"
#include "IMMCTrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"

namespace VT_Physics::immct {

    inline const std::vector<std::string> IMMCTConfigRequiredKeys = {
            "animationTime",
            "timeStep",
            "particleRadius",
            "simSpaceLB",
            "simSpaceSize",
            "maxNeighborNum",
            "divFreeThreshold",
            "incompThreshold",
            "surfaceTensionCoefficient",
            "diffusionCoefficientCf",
            "momentumExchangeCoefficientCd0",
            "solventViscosity",
            "phaseRestDensity",
            "phaseRestColor",
            "solutionBasicViscosity",
            "solutionMaxViscosity",
            "relaxationTime",
            "shearThinningBasicFactor",
            "rheologicalThreshold"
    };

    inline const std::vector<std::string> IMMCTConfigOptionalKeys = {
            "enable",
            "gravity",
    };

    inline const std::vector<std::string> IMMCTSolverObjectComponentConfigRequiredKeys = {
            "solverType",
            "exportFlag",
            "velocityStart",
            "phaseFraction",
    };

    inline const std::set<uint8_t> IMMCTSolverSupportedMaterials = {
            EPM_FLUID,
            EPM_BOUNDARY
    };

    class IMMCTSolver : public Solver {
    public:
        IMMCTSolver() = delete;

        IMMCTSolver(uint32_t cudaThreadSize);

        virtual ~IMMCTSolver() override = default;

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
        std::vector<float> m_host_phaseFrac;
    };
}

#endif //VT_PHYSICS_IMMCTSOLVER_HPP
