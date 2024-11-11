#include "Solvers/DFSPH/DFSPHSolver.hpp"

#include <string>

#include "Logger/Logger.hpp"
#include "Model/ExportUtil.hpp"
#include "JSON/JSONHandler.hpp"
#include "DFSPHCudaApi.cuh"

namespace VT_Physics::dfsph {

    DFSPHSolver::DFSPHSolver(uint32_t cudaThreadSize) {
        m_configData = JsonHandler::loadDFSPHConfigTemplateJson();
        m_host_data = new Data();
        m_host_data->thread_num = cudaThreadSize;
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("DFSPHSolver fail to allocate memory for device data.")
            delete m_host_data;
            return;
        }

        LOG_INFO("DFSPHSolver Created.");
    }

    json DFSPHSolver::getSolverConfigTemplate() const {
        return JsonHandler::loadDFSPHConfigTemplateJson();
    }

    bool DFSPHSolver::setConfig(json config) {
        m_configData = config;
        if (m_configData.empty()) {
            LOG_ERROR("DFSPH config json is empty.");
            return false;
        }

        if (!checkConfig()) {
            LOG_ERROR("Invalid DFSPH config file.");
            return false;
        }

        auto dfsph_config = m_configData["DFSPH"];
        m_host_data->dt = dfsph_config["Required"]["timeStep"].get<float>();
        m_host_data->inv_dt = 1 / m_host_data->dt;
        m_host_data->inv_dt2 = m_host_data->inv_dt * m_host_data->inv_dt;
        m_host_data->cur_simTime = 0.f;
        m_host_data->particle_radius = dfsph_config["Required"]["particleRadius"].get<float>();
        m_host_data->fPart_rest_density = dfsph_config["Required"]["fPartRestDensity"].get<float>();
        m_host_data->bPart_rest_density = dfsph_config["Required"]["bPartRestDensity"].get<float>();
        m_host_data->fPart_rest_volume = std::powf(2 * m_host_data->particle_radius, 3);
        m_host_data->h = m_host_data->particle_radius * 4;
        m_host_data->fPart_rest_viscosity = dfsph_config["Required"]["fPartRestViscosity"].get<float>();
        m_host_data->div_free_threshold = dfsph_config["Required"]["divFreeThreshold"].get<float>();
        m_host_data->incomp_threshold = dfsph_config["Required"]["incompThreshold"].get<float>();
        m_host_data->surface_tension_coefficient = dfsph_config["Required"]["surfaceTensionCoefficient"].get<float>();
        m_host_data->gravity = make_float3(0.f, -9.8f, 0.f);

        if (dfsph_config["Optional"]["enable"]) {
            auto g = dfsph_config["Optional"]["gravity"].get<std::vector<float>>();
            m_host_data->gravity = make_float3(g[0], g[1], g[2]);
        }

        LOG_INFO("DFSPHSolver Configured.");
        return true;
    }

    bool DFSPHSolver::setConfigByFile(std::string config_file) {
        setConfig(JsonHandler::loadJson(config_file));
        return true;
    }

    json DFSPHSolver::getSolverObjectComponentConfigTemplate() {
        return JsonHandler::loadDFSPHObjectComponentConfigTemplateJson();
    }

    bool DFSPHSolver::initialize() {
        m_host_data->particle_num = m_host_pos.size();
        m_host_data->block_num = (m_host_data->particle_num + m_host_data->thread_num - 1) / m_host_data->thread_num;

        if (!ExportUtil::checkConfig(m_configData["EXPORT"])) {
            LOG_ERROR("DFSPHSolver export config not available.");
            return false;
        }

        auto ugns_config = JsonHandler::loadUGNSConfigTemplateJson();
        ugns_config["simSpaceLB"] = m_configData["DFSPH"]["Required"]["simSpaceLB"];
        ugns_config["simSpaceSize"] = m_configData["DFSPH"]["Required"]["simSpaceSize"];
        ugns_config["totalParticleNum"] = m_host_data->particle_num;
        ugns_config["maxNeighborNum"] = m_configData["DFSPH"]["Required"]["maxNeighborNum"];
        ugns_config["cuKernelBlockNum"] = m_host_data->block_num;
        ugns_config["cuKernelThreadNum"] = m_host_data->thread_num;
        ugns_config["gridCellSize"] = m_host_data->h;
        m_neighborSearcher.setConfig(ugns_config);

        // cudaMalloc and memcpy
        if (!m_host_data->malloc()) {
            LOG_ERROR("DFSPHSolver fail to allocate memory for rt data.");
            return false;
        }
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (!m_neighborSearcher.malloc()) {
            return false;
        }

        cudaMemcpy(m_host_data->pos, m_host_pos.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->pos_adv, m_host_pos.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->vel, m_host_vel.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->vel_adv, m_host_vel.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->color, m_host_color.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->mat, m_host_mat.data(), m_host_data->particle_num * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_data, m_host_data, sizeof(Data), cudaMemcpyHostToDevice);

        m_neighborSearcher.update(m_host_data->pos);
        init_data(m_host_data,
                  m_device_data,
                  m_neighborSearcher.m_config_cuData,
                  m_neighborSearcher.m_params_cuData);

        prepare_dfsph(m_host_data,
                      m_device_data,
                      m_neighborSearcher.m_config_cuData,
                      m_neighborSearcher.m_params_cuData);

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("DFSPHSolver fail to initialize.")
            return false;
        }

        m_isInitialized = true;
        return true;
    }

    bool DFSPHSolver::run() {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        while (m_host_data->cur_simTime < m_configData["DFSPH"]["Required"]["animationTime"].get<float>()) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("DFSPHSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("DFSPHSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool DFSPHSolver::tickNsteps(uint32_t n) {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        for (uint32_t i = 0; i < n; ++i) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("DFSPHSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("DFSPHSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool DFSPHSolver::attachObject(Object *obj) {
        if (static_cast<uint8_t>(obj->getObjectComponent()->getType()) > 20) {
            LOG_ERROR("DFSPHSolver only support particle geometry yet.");
            return false;
        }

        if (DFSPHSolverSupportedMaterials.count(obj->getObjectComponentConfig()["epmMaterial"]) == 0) {
            LOG_ERROR("DFSPHSolver unsupported material.");
            return false;
        }

        for (const auto &key: DFSPHSolverObjectComponentConfigRequiredKeys) {
            if (!obj->getSolverObjectComponentConfig().contains(key)) {
                LOG_ERROR(
                        "Object: " + std::to_string(obj->getID()) + " has no DFSPH object component config key: " +
                        key);
                return false;
            }
        }

        if (obj->getSolverObjectComponentConfig()["solverType"] != static_cast<uint8_t>(SolverType::DFSPH)) {
            LOG_ERROR("Object: " + std::to_string(obj->getID()) + " has no DFSPH object component config.");
            return false;
        }

        if (!m_configData["EXPORT"]["SolverRequired"].contains("exportObjectStartIndex")) {
            m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportFlags"] = json::array();
        }

        m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"].push_back(
                obj->getObjectComponentConfig()["epmMaterial"]);
        m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].push_back(m_host_pos.size());
        m_configData["EXPORT"]["SolverRequired"]["exportFlags"].push_back(
                obj->getSolverObjectComponentConfig()["exportFlag"].get<bool>());

        auto pos = obj->getObjectComponent()->getElements();
        auto part_num = pos.size() / 3;
        auto pos_f3ptr = reinterpret_cast<float3 *>(pos.data());

        auto vel_start = obj->getSolverObjectComponentConfig()["velocityStart"].get<std::vector<float>>();
        auto color_start = obj->getSolverObjectComponentConfig()["colorStart"].get<std::vector<float>>();

        m_host_pos.insert(m_host_pos.end(), pos_f3ptr, pos_f3ptr + part_num);
        m_host_vel.insert(m_host_vel.end(), part_num, make_float3(vel_start[0], vel_start[1], vel_start[2]));
        m_host_mat.insert(m_host_mat.end(), part_num, obj->getObjectComponentConfig()["epmMaterial"].get<int>());
        m_host_color.insert(m_host_color.end(), part_num, make_float3(color_start[0], color_start[1], color_start[2]));

        m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"].push_back(m_host_pos.size());
        m_attached_objs.push_back(obj);
        return true;
    }

    bool DFSPHSolver::attachObjects(std::vector<Object *> objs) {
        for (auto obj: objs) {
            if (!attachObject(obj)) {
                return false;
            }
        }

        return true;
    }

    bool DFSPHSolver::reset() {
        m_configData = JsonHandler::loadDFSPHConfigTemplateJson();
        return true;
    }

    void DFSPHSolver::destroy() {
        m_host_data->free();
        delete m_host_data;
        cudaFree(m_device_data);
        m_attached_objs.clear();
        m_neighborSearcher.freeMemory();

        LOG_INFO("DFSPHSolver Destroyed.");
    }

    void DFSPHSolver::exportData() {
        static auto &exportConfig = m_configData["EXPORT"];
        if (exportConfig["SolverRequired"]["enable"] && m_doExportFlag) {
            if (exportConfig["SolverRequired"]["exportGroupPolicy"] == "MERGE") {
                std::set<int> mat_exported;
                for (int i = 0; i < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++i) {
                    if (exportConfig["SolverRequired"]["exportFlags"][i].get<bool>()) {
                        auto cur_mat = exportConfig["SolverRequired"]["exportObjectMaterials"][i].get<int>();
                        if (mat_exported.count(cur_mat) != 0) {
                            continue;
                        } else {
                            mat_exported.insert(cur_mat);
                            std::vector<float3> pos_merged;
                            std::vector<float3> color_merged;
                            for (int j = 0; j < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++j) {
                                if (exportConfig["SolverRequired"]["exportObjectMaterials"][j].get<int>() != cur_mat)
                                    continue;
                                int data_size = exportConfig["SolverRequired"]["exportObjectEndIndex"][j].get<int>() -
                                                exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>();
                                std::vector<float3> pos_tmp(data_size);
                                std::vector<float3> color_tmp(data_size);
                                cudaMemcpy(pos_tmp.data(),
                                           m_host_data->pos +
                                           exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>(),
                                           data_size * sizeof(float3),
                                           cudaMemcpyDeviceToHost);
                                cudaMemcpy(color_tmp.data(),
                                           m_host_data->color +
                                           exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>(),
                                           data_size * sizeof(float3),
                                           cudaMemcpyDeviceToHost);
                                pos_merged.insert(pos_merged.end(), pos_tmp.begin(), pos_tmp.end());
                                color_merged.insert(color_merged.end(), color_tmp.begin(), color_tmp.end());
                            }
                            auto exportConfig_tmp = exportConfig;
                            exportConfig_tmp["Common"]["exportTargetDir"] =
                                    exportConfig_tmp["Common"]["exportTargetDir"].get<std::string>() + "/" +
                                    EPMString.at(cur_mat);
                            exportConfig_tmp["Common"]["exportFilePrefix"] = std::to_string(m_outputFrameCount);
                            ExportUtil::exportData(exportConfig_tmp,
                                                   pos_merged);
                        }
                    }
                }
            } else if (exportConfig["SolverRequired"]["exportGroupPolicy"] == "SPLIT") {
                for (int i = 0; i < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++i) {
                    if (exportConfig["SolverRequired"]["exportFlags"][i].get<bool>()) {
                        int data_size = exportConfig["SolverRequired"]["exportObjectEndIndex"][i].get<int>() -
                                        exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>();
                        std::vector<float3> pos_tmp(data_size);
                        std::vector<float3> color_tmp(data_size);
                        cudaMemcpy(pos_tmp.data(),
                                   m_host_data->pos +
                                   exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>(),
                                   data_size * sizeof(float3),
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(color_tmp.data(),
                                   m_host_data->color +
                                   exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>(),
                                   data_size * sizeof(float3),
                                   cudaMemcpyDeviceToHost);
                        auto exportConfig_tmp = exportConfig;
                        exportConfig_tmp["Common"]["exportTargetDir"] += "/" + m_attached_objs[i]->getName();
                        exportConfig_tmp["Common"]["exportFilePrefix"] += std::to_string(m_outputFrameCount);
                        ExportUtil::exportData(exportConfig_tmp,
                                               pos_tmp,
                                               color_tmp);
                    }
                }
            } else {
                LOG_ERROR("DFSPHSolver exportGroupPolicy" +
                          exportConfig["SolverRequired"]["exportGroupPolicy"].get<std::string>() + " not supported.");
            }

            m_doExportFlag = false;
        }
    }

    bool DFSPHSolver::tick() {
        static const float export_gap = 1 / m_configData["EXPORT"]["SolverRequired"]["exportFps"].get<float>();
        static const std::vector<int> exportObjectStartIndex =
                m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].get<std::vector<int>>();
        static const std::vector<int> exportObjectEndIndex =
                m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"].get<std::vector<int>>();
        static const std::vector<int> exportObjectMats =
                m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"].get<std::vector<int>>();

        m_neighborSearcher.update(m_host_data->pos);

        sph_precompute(m_host_data,
                       m_device_data,
                       m_neighborSearcher.m_config_cuData,
                       m_neighborSearcher.m_params_cuData);

        vfsph_div(m_host_data,
                  m_device_data,
                  exportObjectStartIndex,
                  exportObjectEndIndex,
                  exportObjectMats,
                  m_neighborSearcher.m_config_cuData,
                  m_neighborSearcher.m_params_cuData,
                  m_isCrashed);

        apply_pressure_acc(m_host_data,
                           m_device_data,
                           m_neighborSearcher.m_config_cuData,
                           m_neighborSearcher.m_params_cuData);

        dfsph_gravity_vis_surface(m_host_data,
                                  m_device_data,
                                  m_neighborSearcher.m_config_cuData,
                                  m_neighborSearcher.m_params_cuData);

        vfsph_incomp(m_host_data,
                     m_device_data,
                     exportObjectStartIndex,
                     exportObjectEndIndex,
                     exportObjectMats,
                     m_neighborSearcher.m_config_cuData,
                     m_neighborSearcher.m_params_cuData,
                     m_isCrashed);

        apply_pressure_acc(m_host_data,
                           m_device_data,
                           m_neighborSearcher.m_config_cuData,
                           m_neighborSearcher.m_params_cuData);

        update_pos(m_host_data,
                   m_device_data,
                   m_neighborSearcher.m_config_cuData,
                   m_neighborSearcher.m_params_cuData);

        m_host_data->cur_simTime += m_host_data->dt;
        m_frameCount++;
        if (m_host_data->cur_simTime >= export_gap * static_cast<float>(m_outputFrameCount)) {
            m_outputFrameCount++;
            m_doExportFlag = true;
        }

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("DFSPHSolver tick failed.");
            m_isCrashed = true;
            return false;
        }
        return true;
    }

    bool DFSPHSolver::checkConfig() const {
        if (!m_configData.contains("DFSPH")) {
            LOG_ERROR("DFSPH config missing main domain: DFSPH")
            return false;
        }

        auto dfsph_config = m_configData["DFSPH"];
        if (!dfsph_config.contains("Required")) {
            LOG_ERROR("DFSPH config missing necessary domain: Required")
            return false;
        }

        for (const auto &key: DFSPHConfigRequiredKeys) {
            if (!dfsph_config["Required"].contains(key)) {
                LOG_ERROR("DFSPH config missing Required key: " + key)
                return false;
            }
        }

        for (const auto &key: DFSPHConfigOptionalKeys) {
            if (!dfsph_config["Optional"].contains(key)) {
                LOG_ERROR("DFSPH config missing Optional key: " + key)
                return false;
            }
        }

        return true;
    }

}