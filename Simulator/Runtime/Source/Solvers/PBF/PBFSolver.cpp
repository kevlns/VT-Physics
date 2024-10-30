#include "Solvers/PBF/PBFSolver.hpp"

#include <string>

#include "Logger/Logger.hpp"
#include "Model/ExportUtil.hpp"
#include "JSON/JSONHandler.hpp"

namespace VT_Physics::pbf {

    PBFSolver::PBFSolver(uint32_t cudaThreadSize) {
        m_defaultConfig = JsonHandler::loadPBFConfigTemplateJson();
        m_configData = m_defaultConfig;
        m_host_data = new Data();
        m_cuThreadNum = cudaThreadSize;
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("PBFSolver fail to allocate memory for device data.")
            delete m_host_data;
            return;
        }

        LOG_INFO("PBFSolver Created.");
    }

    json PBFSolver::getDefaultConfig() const {
        return m_defaultConfig;
    }

    bool PBFSolver::setConfig(json config) {
        m_configData = config;
        return true;
    }

    bool PBFSolver::setConfigByFile(std::string solver_config) {
        m_configData = JsonHandler::loadJson(solver_config);
        if (m_configData.empty()) {
            LOG_ERROR("PBF config json is empty.");
            return false;
        }

        if (!checkConfig()) {
            LOG_ERROR("Invalid PBF config file.");
            return false;
        }

        auto pbf_config = m_configData["PBF"];
        m_host_data->dt = pbf_config["Required"]["timeStep"].get<float>();
        m_host_data->inv_dt = 1 / m_host_data->dt;
        m_host_data->inv_dt2 = m_host_data->inv_dt * m_host_data->inv_dt;
        m_host_data->cur_simTime = 0.f;
        m_host_data->particle_radius = pbf_config["Required"]["particleRadius"].get<float>();
        m_host_data->fPart_rest_density = pbf_config["Required"]["fPartRestDensity"].get<float>();
        m_host_data->fPart_rest_volume = std::powf(2 * m_host_data->particle_radius, 3);
        m_host_data->h = m_host_data->particle_radius * 4;
        m_host_data->gravity = make_float3(0.f, -9.8f, 0.f);

        if (pbf_config["Optional"]["enable"]) {
            auto g = pbf_config["Optional"]["gravity"].get<std::vector<float>>();
            m_host_data->gravity = make_float3(g[0], g[1], g[2]);
        }

        LOG_INFO("PBFSolver Configured.");
        return true;
    }

    bool PBFSolver::initialize() {

        return false;
    }

    bool PBFSolver::run(float simTime) {
        if (!m_isInitialized) {
            if (!initialize()) {
                LOG_ERROR("PBFSolver fail to initialize.");
                return false;
            }
        }

        while (m_host_data->cur_simTime < simTime) {
            if (!m_isCrashed) {
                tick();
                LOG_WARNING("PBFSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("PBFSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool PBFSolver::tickNsteps(uint32_t n) {
        if (!m_isInitialized) {
            if (!initialize()) {
                LOG_ERROR("PBFSolver fail to initialize.");
                return false;
            }
        }

        for (uint32_t i = 0; i < n; ++i) {
            if (!m_isCrashed)
                tick();
            else {
                LOG_ERROR("PBFSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool PBFSolver::attachObject(Object *obj) {
        if (static_cast<uint8_t>(obj->getObjectComponent()->getType()) > 20) {
            LOG_ERROR("PBF solver only support particle geometry yet.");
            return false;
        }

        m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].push_back(m_host_pos.size());
        m_configData["EXPORT"]["SolverRequired"]["exportSourceObjectMaterialList"].push_back(EPM_FLUID);
        auto pos = obj->getObjectComponent()->getElements();
        auto pos_f3ptr = reinterpret_cast<float3 *>(pos.data());
        m_host_pos.insert(m_host_pos.end(), pos_f3ptr, pos_f3ptr + pos.size() / 3);

        return true;
    }

    bool PBFSolver::reset() {
        m_configData = m_defaultConfig;
        return true;
    }

    void PBFSolver::destroy() {
        cudaFree(m_device_data);
        m_host_data->free();
        delete m_host_data;
        LOG_INFO("PBFSolver Destroyed.");
    }

    void PBFSolver::exportData() {
        if (m_configData["EXPORT"]["enable"] && m_exportFlag) {
//            ExportUtil::exportData(m_configData["EXPORT"],
//                                   m_host_data->pos,
//                                   m_host_data->color);
            m_exportFlag = false;
        }
    }

    bool PBFSolver::tick() {
        static const float export_gap = 1 / m_configData["EXPORT"]["exportFps"].get<float>();

        // TODO steps


        m_host_data->cur_simTime += m_host_data->dt;
        m_frameCount++;
        if (m_host_data->cur_simTime >= export_gap * static_cast<float>(m_outputFrameCount)) {
            m_outputFrameCount++;
            m_exportFlag = true;
        }
        return true;
    }

    bool PBFSolver::checkConfig() const {
        if (!m_configData.contains("PBF")) {
            LOG_ERROR("PBF config missing main domain: PBF")
            return false;
        }

        auto pbf_config = m_configData["PBF"];
        if (!pbf_config.contains("Required")) {
            LOG_ERROR("PBF config missing necessary domain: Required")
            return false;
        }

        for (const auto &key: PBFConfigRequiredKeys) {
            if (!pbf_config["Required"].contains(key)) {
                LOG_ERROR("PBF config missing Required key: " + key)
                return false;
            }
        }

        if (pbf_config.contains("Optional") && pbf_config["Optional"].contains("enable")
            && pbf_config["Optional"]["enable"]) {
            for (const auto &key: PBFConfigOptionalKeys) {
                if (!pbf_config["Optional"].contains(key)) {
                    LOG_ERROR("PBF config missing Optional key: " + key)
                    return false;
                }
            }
        }

        return true;
    }

}