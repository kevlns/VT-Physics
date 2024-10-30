/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_PBFSOLVER_HPP
#define VT_PHYSICS_PBFSOLVER_HPP

#include <vector>

#include "Framework/Solver.hpp"
#include "PBFrtData.hpp"

namespace VT_Physics::pbf {

    static const std::vector<std::string> PBFConfigRequiredKeys = {
            "animationTime",
            "timeStep",
            "particleRadius",
            "simSpace_lb",
            "simSpace_size",
            "iterationNum",
            "XSPH_k",
            "fPartRestDensity"
    };

    static const std::vector<std::string> PBFConfigOptionalKeys = {
            "enable",
            "gravity",
    };

    class PBFSolver : public Solver {
    public:
        PBFSolver() = delete;

        PBFSolver(uint32_t cudaThreadSize);

        virtual ~PBFSolver() override = default;

        virtual json getDefaultConfig() const override;

        virtual bool setConfig(json config) override;

        virtual bool setConfigByFile(std::string solver_config) override;

        virtual bool initialize() override;

        virtual bool run(float simTime) override;

        virtual bool tickNsteps(uint32_t n) override;

        virtual bool attachObject(Object *obj) override;

        virtual bool reset() override;

        virtual void destroy() override;

    protected:
        virtual bool tick() override;

        virtual bool checkConfig() const override;

    private:
        void swizzle();

        void exportData();

    private:
        json m_configData;
        uint32_t m_cuBlockNum{0};
        uint32_t m_cuThreadNum{0};
        Data *m_host_data{nullptr};
        Data *m_device_data{nullptr};
        bool m_isInitialized{false};
        bool m_isCrashed{false};
        uint32_t m_frameCount{0};
        uint32_t m_outputFrameCount{0};
        bool m_exportFlag{false};
        std::vector<Object *> m_attached_obj;

        std::vector<float3> m_host_pos;
        std::vector<float3> m_host_vel;
        std::vector<ParticleMaterial> m_host_mat;
        std::vector<float3> m_host_color;
        json m_defaultConfig;
    };
}

#endif //VT_PHYSICS_PBFSOLVER_HPP
