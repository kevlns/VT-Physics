#include "Solvers/IMM/IMMSolver.hpp"

#include <string>

#include "Logger/Logger.hpp"
#include "Model/ExportUtil.hpp"
#include "JSON/JSONHandler.hpp"
#include "IMMCudaApi.cuh"
#include "Core/Math/DataStructTransfer.hpp"

namespace VT_Physics::imm {

    IMMSolver::IMMSolver(uint32_t cudaThreadSize) {
        m_configData = JsonHandler::loadIMMConfigTemplateJson();
        m_host_data = new Data();
        m_host_data->thread_num = cudaThreadSize;
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("IMMSolver fail to allocate memory for device data.")
            delete m_host_data;
            return;
        }

        LOG_INFO("IMMSolver Created.");
    }

    json IMMSolver::getSolverConfigTemplate() const {
        return JsonHandler::loadIMMConfigTemplateJson();
    }

    bool IMMSolver::setConfig(json config) {
        m_configData = config;
        if (m_configData.empty()) {
            LOG_ERROR("IMM config json is empty.");
            return false;
        }

        if (!checkConfig()) {
            LOG_ERROR("Invalid IMM config file.");
            return false;
        }

        auto imm_config = m_configData["IMM"];
        m_host_data->dt = imm_config["Required"]["timeStep"].get<float>();
        m_host_data->inv_dt = 1 / m_host_data->dt;
        m_host_data->inv_dt2 = m_host_data->inv_dt * m_host_data->inv_dt;
        m_host_data->cur_simTime = 0.f;
        m_host_data->particle_radius = imm_config["Required"]["particleRadius"].get<float>();
        m_host_data->fPart_rest_volume = std::powf(2 * m_host_data->particle_radius, 3);
        m_host_data->h = m_host_data->particle_radius * 4;
        m_host_data->div_free_threshold = imm_config["Required"]["divFreeThreshold"].get<float>();
        m_host_data->incomp_threshold = imm_config["Required"]["incompThreshold"].get<float>();
        m_host_data->surface_tension_coefficient = imm_config["Required"]["surfaceTensionCoefficient"].get<float>();
        m_host_data->gravity = make_float3(0.f, -9.8f, 0.f);
        m_host_data->Cf = imm_config["Required"]["diffusionCoefficientCf"].get<float>();
        m_host_data->Cd = imm_config["Required"]["momentumExchangeCoefficientCd"].get<float>();
        m_host_data->phase_num = imm_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size();

        cudaMalloc((void **) &m_host_data->phase_rest_density, m_host_data->phase_num * sizeof(float));
        cudaMalloc((void **) &m_host_data->phase_rest_vis, m_host_data->phase_num * sizeof(float));
        cudaMalloc((void **) &m_host_data->phase_rest_color, m_host_data->phase_num * sizeof(float3));
        cudaMemcpy(m_host_data->phase_rest_density,
                   imm_config["Required"]["phaseRestDensity"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->phase_rest_vis,
                   imm_config["Required"]["phaseRestViscosity"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        auto phase_colors = imm_config["Required"]["phaseRestColor"].get<std::vector<std::vector<float>>>();
        std::vector<float3> phase_colors_f3;
        for (auto &color: phase_colors) {
            phase_colors_f3.push_back(make_cuFloat3(color));
        }
        cudaMemcpy(m_host_data->phase_rest_color,
                   phase_colors_f3.data(),
                   m_host_data->phase_num * sizeof(float3),
                   cudaMemcpyHostToDevice);

        if (imm_config["Optional"]["enable"]) {
            auto g = imm_config["Optional"]["gravity"].get<std::vector<float>>();
            m_host_data->gravity = make_float3(g[0], g[1], g[2]);
        }

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("IMMSolver fail to set config.");
            return false;
        }

        LOG_INFO("IMMSolver Configured.");
        return true;
    }

    bool IMMSolver::setConfigByFile(std::string config_file) {
        setConfig(JsonHandler::loadJson(config_file));
        return true;
    }

    json IMMSolver::getSolverObjectComponentConfigTemplate() {
        return JsonHandler::loadIMMObjectComponentConfigTemplateJson();
    }

    bool IMMSolver::initialize() {
        m_host_data->particle_num = m_host_pos.size();
        m_host_data->block_num = (m_host_data->particle_num + m_host_data->thread_num - 1) / m_host_data->thread_num;
        m_configData["EXPORT"]["Common"]["exportPhaseNum"] = m_host_data->phase_num;

        if (!ExportUtil::checkConfig(m_configData["EXPORT"])) {
            LOG_ERROR("IMMSolver export config not available.");
            return false;
        }

        auto ugns_config = JsonHandler::loadUGNSConfigTemplateJson();
        ugns_config["simSpaceLB"] = m_configData["IMM"]["Required"]["simSpaceLB"];
        ugns_config["simSpaceSize"] = m_configData["IMM"]["Required"]["simSpaceSize"];
        ugns_config["totalParticleNum"] = m_host_data->particle_num;
        ugns_config["maxNeighborNum"] = m_configData["IMM"]["Required"]["maxNeighborNum"];
        ugns_config["cuKernelBlockNum"] = m_host_data->block_num;
        ugns_config["cuKernelThreadNum"] = m_host_data->thread_num;
        ugns_config["gridCellSize"] = m_host_data->h;
        m_neighborSearcher.setConfig(ugns_config);

        // cudaMalloc and memcpy
        if (!m_host_data->malloc()) {
            LOG_ERROR("IMMSolver fail to allocate memory for rt data.");
            return false;
        }
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (!m_neighborSearcher.malloc()) {
            return false;
        }

        cudaMemcpy(m_host_data->mat, m_host_mat.data(), m_host_data->particle_num * sizeof(int),
                   cudaMemcpyHostToDevice);
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
        cudaMemcpy(m_host_data->vol_frac, m_host_phaseFrac.data(),
                   m_host_data->particle_num * m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_data, m_host_data, sizeof(Data), cudaMemcpyHostToDevice);

        m_neighborSearcher.update(m_host_data->pos);
        init_data(m_host_data,
                  m_device_data);

        prepare_imm(m_host_data,
                    m_device_data,
                    m_neighborSearcher.m_config_cuData,
                    m_neighborSearcher.m_params_cuData);

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("IMMSolver fail to initialize.")
            return false;
        }

        m_isInitialized = true;
        return m_isInitialized;
    }

    bool IMMSolver::run() {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        while (m_host_data->cur_simTime < m_configData["IMM"]["Required"]["animationTime"].get<float>()) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("IMMSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("IMMSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool IMMSolver::tickNsteps(uint32_t n) {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        for (uint32_t i = 0; i < n; ++i) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("IMMSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("IMMSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool IMMSolver::attachObject(Object *obj) {
        if (static_cast<uint8_t>(obj->getObjectComponent()->getType()) > 20) {
            LOG_ERROR("IMMSolver only support particle geometry yet.");
            return false;
        }

        if (IMMSolverSupportedMaterials.count(obj->getObjectComponentConfig()["epmMaterial"]) == 0) {
            LOG_ERROR("IMMSolver unsupported material.");
            return false;
        }

        for (const auto &key: IMMSolverObjectComponentConfigRequiredKeys) {
            if (!obj->getSolverObjectComponentConfig().contains(key)) {
                LOG_ERROR(
                        "Object: " + std::to_string(obj->getID()) + " has no IMM object component config key: " +
                        key);
                return false;
            }
        }

        if (obj->getSolverObjectComponentConfig()["solverType"] != static_cast<uint8_t>(SolverType::IMM)) {
            LOG_ERROR("Object: " + std::to_string(obj->getID()) + " has no IMM object component config.");
            return false;
        }

        if (!m_configData["EXPORT"]["SolverRequired"].contains("exportObjectStartIndex")) {
            m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportFlags"] = json::array();
        }

        if (obj->getSolverObjectComponentConfig()["phaseFraction"].get<std::vector<float>>().size() !=
            m_host_data->phase_num) {
            LOG_ERROR("Object: " + std::to_string(obj->getID()) + " phaseFraction size not match with solver.");
            return false;
        }

        m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"].push_back(
                obj->getObjectComponentConfig()["epmMaterial"]);
        m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].push_back(m_host_pos.size());
        m_configData["EXPORT"]["SolverRequired"]["exportFlags"].push_back(
                obj->getSolverObjectComponentConfig()["exportFlag"].get<bool>());

        auto pos = obj->getObjectComponent()->getElements();

        auto pos_tmp = make_cuFloat3Vec(obj->getObjectComponent()->getElements());
        auto part_num = pos_tmp.size();

        std::vector<float3> vel_tmp(part_num,
                                    make_cuFloat3(
                                            obj->getSolverObjectComponentConfig()["velocityStart"].get<std::vector<float>>()));

        std::vector<int> mat_tmp(part_num, obj->getObjectComponentConfig()["epmMaterial"].get<int>());

        std::vector<float> vol_frac_tmp(part_num * m_host_data->phase_num);
        float3 color_start = {0, 0, 0};
        for (int i = 0; i < m_host_data->phase_num; ++i) {
            color_start += obj->getSolverObjectComponentConfig()["phaseFraction"].get<std::vector<float>>()[i] *
                           make_cuFloat3(
                                   m_configData["IMM"]["Required"]["phaseRestColor"].get<std::vector<std::vector<float>>>()[i]);
            for (int j = 0; j < part_num; ++j)
                vol_frac_tmp[j * m_host_data->phase_num + i] =
                        obj->getSolverObjectComponentConfig()["phaseFraction"].get<std::vector<float>>()[i];
        }

        m_host_pos.insert(m_host_pos.end(), pos_tmp.begin(), pos_tmp.end());
        m_host_vel.insert(m_host_vel.end(), vel_tmp.begin(), vel_tmp.end());
        m_host_mat.insert(m_host_mat.end(), mat_tmp.begin(), mat_tmp.end());
        m_host_color.insert(m_host_color.end(), part_num, color_start);
        m_host_phaseFrac.insert(m_host_phaseFrac.end(), vol_frac_tmp.begin(), vol_frac_tmp.end());

        m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"].push_back(m_host_pos.size());
        m_attached_objs.push_back(obj);
        return true;
    }

    bool IMMSolver::attachObjects(std::vector<Object *> objs) {
        for (auto obj: objs) {
            if (!attachObject(obj)) {
                return false;
            }
        }

        return true;
    }

    bool IMMSolver::reset() {
        m_configData = JsonHandler::loadIMMConfigTemplateJson();
        return true;
    }

    void IMMSolver::destroy() {
        m_host_data->free();
        delete m_host_data;
        cudaFree(m_device_data);
        m_attached_objs.clear();
        m_neighborSearcher.freeMemory();

        LOG_INFO("IMMSolver Destroyed.");
    }

    void IMMSolver::exportData() {
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
                            std::vector<float> phase_merged;
                            for (int j = 0; j < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++j) {
                                if (exportConfig["SolverRequired"]["exportObjectMaterials"][j].get<int>() != cur_mat)
                                    continue;
                                int data_size = exportConfig["SolverRequired"]["exportObjectEndIndex"][j].get<int>() -
                                                exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>();
                                std::vector<float3> pos_tmp(data_size);
                                std::vector<float3> color_tmp(data_size);
                                std::vector<float> phase_tmp(data_size * m_host_data->phase_num);
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
                                cudaMemcpy(phase_tmp.data(),
                                           m_host_data->vol_frac +
                                           exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>(),
                                           data_size * m_host_data->phase_num * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                                pos_merged.insert(pos_merged.end(), pos_tmp.begin(), pos_tmp.end());
                                color_merged.insert(color_merged.end(), color_tmp.begin(), color_tmp.end());
                                phase_merged.insert(phase_merged.end(), phase_tmp.begin(), phase_tmp.end());
                            }
                            auto exportConfig_tmp = exportConfig;
                            exportConfig_tmp["Common"]["exportTargetDir"] =
                                    exportConfig_tmp["Common"]["exportTargetDir"].get<std::string>() + "/" +
                                    EPMString.at(cur_mat);
                            exportConfig_tmp["Common"]["exportFilePrefix"] = std::to_string(m_outputFrameCount);
                            ExportUtil::exportData(exportConfig_tmp,
                                                   pos_merged,
                                                   color_merged,
                                                   phase_merged);
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
                LOG_ERROR("IMMSolver exportGroupPolicy" +
                          exportConfig["SolverRequired"]["exportGroupPolicy"].get<std::string>() + " not supported.");
            }

            m_doExportFlag = false;
        }
    }

    bool IMMSolver::tick() {
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
                           m_neighborSearcher.m_params_cuData);

        imm_gravity_vis_surface(m_host_data,
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
                           m_neighborSearcher.m_params_cuData);

        update_pos(m_host_data,
                   m_device_data,
                   m_neighborSearcher.m_params_cuData);

        phase_transfer(m_host_data,
                       m_device_data,
                       m_neighborSearcher.m_config_cuData,
                       m_neighborSearcher.m_params_cuData,
                       m_isCrashed);

        update_mass_and_vel(m_host_data,
                            m_device_data,
                            m_neighborSearcher.m_config_cuData,
                            m_neighborSearcher.m_params_cuData);

        update_color(m_host_data,
                     m_device_data,
                     m_neighborSearcher.m_params_cuData);

        m_host_data->cur_simTime += m_host_data->dt;
        m_frameCount++;
        if (m_host_data->cur_simTime >= export_gap * static_cast<float>(m_outputFrameCount)) {
            m_outputFrameCount++;
            m_doExportFlag = true;
        }

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("IMMSolver tick failed.");
            m_isCrashed = true;
            return false;
        }
        return true;
    }

    bool IMMSolver::checkConfig() const {
        if (!m_configData.contains("IMM")) {
            LOG_ERROR("IMM config missing main domain: IMM")
            return false;
        }

        auto imm_config = m_configData["IMM"];
        if (!imm_config.contains("Required")) {
            LOG_ERROR("IMM config missing necessary domain: Required")
            return false;
        }

        for (const auto &key: IMMConfigRequiredKeys) {
            if (!imm_config["Required"].contains(key)) {
                LOG_ERROR("IMM config missing Required key: " + key)
                return false;
            }
        }

        if (imm_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size() !=
            imm_config["Required"]["phaseRestViscosity"].get<std::vector<float>>().size() &&
            imm_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size() !=
            imm_config["Required"]["phaseRestColor"].get<std::vector<std::vector<float>>>().size()) {
            LOG_ERROR("IMM config phase-related parameter size not match.");
            return false;
        }

        for (const auto &key: IMMConfigOptionalKeys) {
            if (!imm_config["Optional"].contains(key)) {
                LOG_ERROR("IMM config missing Optional key: " + key)
                return false;
            }
        }

        return true;
    }

}