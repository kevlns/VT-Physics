/**
 * @brief Todo
 * @date 2024/11/11
 */

#ifndef VT_PHYSICS_MCTSOLVER_HPP
#define VT_PHYSICS_MCTSOLVER_HPP

#include <vector>
#include <set>

#include "Framework/Solver.hpp"
#include "MCTrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"

namespace VT_Physics::mct {

    inline const std::vector<std::string> MCTConfigRequiredKeys = {
            "animationTime",
            "timeStep",
            "particleRadius",
            "simSpaceLB",
            "simSpaceSize",
            "maxNeighborNum",
            "divFreeThreshold",
            "incompThreshold",
            "surfaceTensionCoefficient",
            "boundViscousFactor",
            "diffusionCoefficientCf",
            "momentumExchangeCoefficientCd",
            "phaseRestDensity",
            "phaseRestColor",
            "phaseRestViscosity",
            "phaseRelaxationTime",
            "phaseThinningFactor",
            "phaseModelImpactFactor",
            "enablePorous",
            "phasePorousPermeability",
            "phasePorousCapillarityStrength",
            "porousPorosity",
            "RestPressurePore",
            "enableDigging",
            "porousHRRate",
    };

    inline const std::vector<std::string> MCTConfigOptionalKeys = {
            "enable",
            "gravity",
    };

    inline const std::vector<std::string> MCTSolverObjectComponentConfigRequiredKeys = {
            "solverType",
            "exportFlag",
            "velocityStart",
            "phaseFraction",
            "fPartMinerRate",
            "hrModelPath"
    };

    inline const std::set<uint8_t> MCTSolverSupportedMaterials = {
            EPM_FLUID,
            EPM_POROUS,
            EPM_BOUNDARY,
            EPM_PCR_FAN
    };

    class MCTSolver : public Solver {
    public:
        MCTSolver() = delete;

        MCTSolver(uint32_t cudaThreadSize);

        virtual ~MCTSolver() override = default;

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

        void generateMiners(std::vector<int> &fParts, int minerNum);

        void digging_porous();

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

        std::vector<int> m_host_digging_fPart_minerFlag;
        std::vector<int> m_host_digging_pPart_alive;
        std::vector<float3> m_host_digging_pPart_hrPos;
        std::vector<float3> m_host_digging_pos;
        std::vector<int> m_host_digging_mat;
        std::vector<float> m_host_digging_porosity;
        UGNS::UniformGirdNeighborSearcher m_neighborSearcher_digging;
        int m_fluid_part_num{0};
    };
}

#endif //VT_PHYSICS_MCTSOLVER_HPP
