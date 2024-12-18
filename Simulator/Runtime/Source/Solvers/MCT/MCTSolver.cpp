#include "Solvers/MCT/MCTSolver.hpp"

#include <string>
#include <random>
#include <algorithm>
#include <omp.h>

#include "Logger/Logger.hpp"
#include "Model/ExportUtil.hpp"
#include "JSON/JSONHandler.hpp"
#include "MCTCudaApi.cuh"
#include "Core/Math/DataStructTransfer.hpp"

namespace VT_Physics::mct {

    MCTSolver::MCTSolver(uint32_t cudaThreadSize) {
        m_configData = JsonHandler::loadMCTConfigTemplateJson();
        m_host_data = new Data();
        m_host_data->thread_num = cudaThreadSize;
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("MCTSolver fail to allocate memory for device data.")
            delete m_host_data;
            return;
        }

        LOG_INFO("MCTSolver Created.");
    }

    json MCTSolver::getSolverConfigTemplate() const {
        return JsonHandler::loadMCTConfigTemplateJson();
    }

    bool MCTSolver::setConfig(json config) {
        m_configData = config;
        if (m_configData.empty()) {
            LOG_ERROR("MCT config json is empty.");
            return false;
        }

        if (!checkConfig()) {
            LOG_ERROR("Invalid MCT config file.");
            return false;
        }

        auto mct_config = m_configData["MCT"];
        m_host_data->dt = mct_config["Required"]["timeStep"].get<float>();
        m_host_data->inv_dt = 1 / m_host_data->dt;
        m_host_data->inv_dt2 = m_host_data->inv_dt * m_host_data->inv_dt;
        m_host_data->cur_simTime = 0.f;
        m_host_data->particle_radius = mct_config["Required"]["particleRadius"].get<float>();
        m_host_data->fPart_rest_volume = std::powf(2 * m_host_data->particle_radius, 3);
        m_host_data->h = m_host_data->particle_radius * 4;
        m_host_data->div_free_threshold = mct_config["Required"]["divFreeThreshold"].get<float>();
        m_host_data->incomp_threshold = mct_config["Required"]["incompThreshold"].get<float>();
        m_host_data->surface_tension_coefficient = mct_config["Required"]["surfaceTensionCoefficient"].get<float>();
        m_host_data->bound_vis_factor = mct_config["Required"]["boundViscousFactor"].get<float>();
        m_host_data->gravity = make_float3(0.f, -9.8f, 0.f);
        m_host_data->Cf = mct_config["Required"]["diffusionCoefficientCf"].get<float>();
        m_host_data->Cd = mct_config["Required"]["momentumExchangeCoefficientCd"].get<float>();
        m_host_data->phase_num = mct_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size();
        m_host_data->intermodel_impact_factor = mct_config["Required"]["phaseModelImpactFactor"].get<float>();
        m_host_data->porous_porosity = mct_config["Required"]["porousPorosity"].get<float>();
        m_host_data->hr_particle_radius =
                m_host_data->particle_radius / mct_config["Required"]["porousHRRate"].get<float>();
        m_host_data->rest_pressure_pore = mct_config["Required"]["RestPressurePore"].get<float>();

        cudaMalloc((void **) &m_host_data->phase_porous_permeability, m_host_data->phase_num * sizeof(float));
        cudaMalloc((void **) &m_host_data->phase_porous_capillarity_strength, m_host_data->phase_num * sizeof(float));
        cudaMalloc((void **) &m_host_data->phase_rest_density, m_host_data->phase_num * sizeof(float));
        cudaMalloc((void **) &m_host_data->phase_rest_vis, m_host_data->phase_num * sizeof(float));
        cudaMalloc((void **) &m_host_data->phase_rest_color, m_host_data->phase_num * sizeof(float3));
        cudaMalloc((void **) &m_host_data->relaxation_time, m_host_data->phase_num * sizeof(float3));
        cudaMalloc((void **) &m_host_data->thinning_factor, m_host_data->phase_num * sizeof(float3));
        cudaMemcpy(m_host_data->phase_porous_permeability,
                   mct_config["Required"]["phasePorousPermeability"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->phase_porous_capillarity_strength,
                   mct_config["Required"]["phasePorousCapillarityStrength"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->phase_rest_density,
                   mct_config["Required"]["phaseRestDensity"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->phase_rest_vis,
                   mct_config["Required"]["phaseRestViscosity"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->relaxation_time,
                   mct_config["Required"]["phaseRelaxationTime"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->thinning_factor,
                   mct_config["Required"]["phaseThinningFactor"].get<std::vector<float>>().data(),
                   m_host_data->phase_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        auto phase_colors = mct_config["Required"]["phaseRestColor"].get<std::vector<std::vector<float>>>();
        std::vector<float3> phase_colors_f3;
        for (auto &color: phase_colors) {
            phase_colors_f3.push_back(make_cuFloat3(color));
        }
        cudaMemcpy(m_host_data->phase_rest_color,
                   phase_colors_f3.data(),
                   m_host_data->phase_num * sizeof(float3),
                   cudaMemcpyHostToDevice);

        if (mct_config["Optional"]["enable"]) {
            auto g = mct_config["Optional"]["gravity"].get<std::vector<float>>();
            m_host_data->gravity = make_float3(g[0], g[1], g[2]);
        }

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("MCTSolver fail to set config.");
            return false;
        }

        LOG_INFO("MCTSolver Configured.");
        return true;
    }

    bool MCTSolver::setConfigByFile(std::string config_file) {
        setConfig(JsonHandler::loadJson(config_file));
        return true;
    }

    json MCTSolver::getSolverObjectComponentConfigTemplate() {
        return JsonHandler::loadMCTObjectComponentConfigTemplateJson();
    }

    bool MCTSolver::initialize() {
        m_host_data->particle_num = m_host_pos.size();
        m_host_data->digging_particle_num = m_host_digging_pos.size();
        m_host_data->block_num = (m_host_data->particle_num + m_host_data->thread_num - 1) / m_host_data->thread_num;
        m_host_data->digging_block_num = (m_host_data->digging_particle_num + m_host_data->thread_num - 1) /
                                         m_host_data->thread_num;
        m_configData["EXPORT"]["Common"]["exportPhaseNum"] = m_host_data->phase_num;

        if (!ExportUtil::checkConfig(m_configData["EXPORT"])) {
            LOG_ERROR("MCTSolver export config not available.");
            return false;
        }

        auto ugns_config = JsonHandler::loadUGNSConfigTemplateJson();
        ugns_config["simSpaceLB"] = m_configData["MCT"]["Required"]["simSpaceLB"];
        ugns_config["simSpaceSize"] = m_configData["MCT"]["Required"]["simSpaceSize"];
        ugns_config["totalParticleNum"] = m_host_data->particle_num;
        ugns_config["maxNeighborNum"] = m_configData["MCT"]["Required"]["maxNeighborNum"];
        ugns_config["cuKernelBlockNum"] = m_host_data->block_num;
        ugns_config["cuKernelThreadNum"] = m_host_data->thread_num;
        ugns_config["gridCellSize"] = m_host_data->h;
        m_neighborSearcher.setConfig(ugns_config);

        if (m_configData["MCT"]["Required"]["enableDigging"].get<bool>()) {
            auto ugns_digging_config = JsonHandler::loadUGNSConfigTemplateJson();
            ugns_digging_config["simSpaceLB"] = m_configData["MCT"]["Required"]["simSpaceLB"];
            ugns_digging_config["simSpaceSize"] = m_configData["MCT"]["Required"]["simSpaceSize"];
            ugns_digging_config["totalParticleNum"] = m_host_data->digging_particle_num;
            ugns_digging_config["maxNeighborNum"] = static_cast<int>(
                    m_configData["MCT"]["Required"]["maxNeighborNum"].get<int>() *
                    m_configData["MCT"]["Required"]["porousHRRate"].get<int>());
            ugns_digging_config["cuKernelBlockNum"] = m_host_data->digging_block_num;
            ugns_digging_config["cuKernelThreadNum"] = m_host_data->thread_num;
            ugns_digging_config["gridCellSize"] = m_host_data->h;
            m_neighborSearcher_digging.setConfig(ugns_digging_config);
            if (!m_neighborSearcher_digging.malloc()) {
                return false;
            }
        }

        // cudaMalloc and memcpy
        if (!m_host_data->malloc()) {
            LOG_ERROR("MCTSolver fail to allocate memory for rt data.");
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

        // for digging policy
        cudaMemcpy(m_host_data->digging_pos, m_host_digging_pos.data(),
                   m_host_data->digging_particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->digging_fPart_miner_flag, m_host_digging_fPart_minerFlag.data(),
                   m_host_data->digging_particle_num * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->digging_porosity, m_host_digging_porosity.data(),
                   m_host_data->digging_particle_num * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->digging_pPart_alive, m_host_digging_pPart_alive.data(),
                   m_host_data->digging_particle_num * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->digging_mat, m_host_digging_mat.data(),
                   m_host_data->digging_particle_num * sizeof(int),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(m_device_data, m_host_data, sizeof(Data), cudaMemcpyHostToDevice);

        for (auto obj: m_attached_objs) {
            if (obj->getObjectComponent()->getType() == Particle_Emitter) {
                auto emitterComponent = dynamic_cast<ParticleEmitterComponent *>(obj->getObjectComponent());
                emitterComponent->attachedPosBuffers = {m_host_data->pos};
                emitterComponent->attachedEPMBuffers = {m_host_data->mat};
            }
        }

        init_data(m_host_data,
                  m_device_data);

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("MCTSolver fail to initialize.")
            return false;
        }

        m_isInitialized = true;
        return m_isInitialized;
    }

    bool MCTSolver::run() {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        while (m_host_data->cur_simTime < m_configData["MCT"]["Required"]["animationTime"].get<float>()) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("MCTSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("MCTSolver is crashed.");
                return false;
            }
        }

        if (m_configData["MCT"]["Required"]["enableDigging"].get<bool>()) {

            std::vector<float3> res_pParts;
            cudaMemcpy(m_host_digging_pPart_hrPos.data(),
                       m_host_data->digging_pos + m_fluid_part_num,
                       m_host_digging_pPart_hrPos.size() * sizeof(float3),
                       cudaMemcpyDeviceToHost);
            std::vector<int> pPart_alive(m_host_digging_pPart_hrPos.size());
            cudaMemcpy(pPart_alive.data(),
                       m_host_data->digging_pPart_alive + m_fluid_part_num,
                       m_host_digging_pPart_hrPos.size() * sizeof(int),
                       cudaMemcpyDeviceToHost);
            for (int i = 0; i < m_host_digging_pPart_hrPos.size(); ++i) {
                if (pPart_alive[i] == 1) {
                    res_pParts.push_back(m_host_digging_pPart_hrPos[i]);
                }
            }

            auto exportConfig_tmp = m_configData["EXPORT"];
            exportConfig_tmp["Common"]["exportTargetDir"] =
                    exportConfig_tmp["Common"]["exportTargetDir"].get<std::string>() + "/Digged_Porous";
            exportConfig_tmp["Common"]["exportFilePrefix"] = std::to_string(m_outputFrameCount);
            ExportUtil::exportData(exportConfig_tmp,
                                   res_pParts);
        }

        return true;
    }

    bool MCTSolver::tickNsteps(uint32_t n) {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        for (uint32_t i = 0; i < n; ++i) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("MCTSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("MCTSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool MCTSolver::attachObject(Object *obj) {
        if (static_cast<uint8_t>(obj->getObjectComponent()->getType()) > 20) {
            LOG_ERROR("MCTSolver only support particle geometry yet.");
            m_isCrashed = true;
            return false;
        }

        if (MCTSolverSupportedMaterials.count(obj->getObjectComponentConfig()["epmMaterial"]) == 0) {
            LOG_ERROR("MCTSolver unsupported material.");
            m_isCrashed = true;
            return false;
        }

        if (obj->getObjectComponent()->getType() == Particle_Emitter &&
            obj->getObjectComponentConfig()["epmMaterial"].get<int>() != EPM_FLUID) {
            LOG_ERROR("MCTSolver unsupported emit material.");
            m_isCrashed = true;
            return false;
        }

        for (const auto &key: MCTSolverObjectComponentConfigRequiredKeys) {
            if (!obj->getSolverObjectComponentConfig().contains(key)) {
                LOG_ERROR(
                        "Object: " + std::to_string(obj->getID()) + " has no MCT object component config key: " +
                        key);
                m_isCrashed = true;
                return false;
            }
        }

        if (obj->getSolverObjectComponentConfig()["solverType"] != static_cast<uint8_t>(SolverType::MCT)) {
            LOG_ERROR("Object: " + std::to_string(obj->getID()) + " has no MCT object component config.");
            m_isCrashed = true;
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
            m_isCrashed = true;
            return false;
        }

        m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"].push_back(
                obj->getObjectComponentConfig()["epmMaterial"]);
        m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].push_back(m_host_pos.size());
        m_configData["EXPORT"]["SolverRequired"]["exportFlags"].push_back(
                obj->getSolverObjectComponentConfig()["exportFlag"].get<bool>());

        if (obj->getObjectComponent()->getType() == Particle_Emitter) {
            auto emitterComponent = dynamic_cast<ParticleEmitterComponent *>(obj->getObjectComponent());
            emitterComponent->bufferInsertOffset = m_host_pos.size();
        }

        auto pos = obj->getObjectComponent()->getElements();

        auto pos_tmp = make_cuFloat3Vec(obj->getObjectComponent()->getElements());
        if (obj->getObjectComponent()->getType() == Particle_Emitter) {
            auto _pos = make_cuFloat3(
                    m_configData["MCT"]["Required"]["simSpaceLB"].get<std::vector<float>>()) +
                        make_cuFloat3(
                                m_configData["MCT"]["Required"]["simSpaceSize"].get<std::vector<float>>()) -
                        make_float3(0.01, 0.01, 0.01);
            pos_tmp = std::vector<float3>(pos_tmp.size(), _pos);
        }

        auto part_num = pos_tmp.size();

        float3 meta_vel = make_cuFloat3(
                obj->getSolverObjectComponentConfig()["velocityStart"].get<std::vector<float>>());
        if (obj->getObjectComponent()->getType() == Particle_Emitter) {
            auto emitterComponent = dynamic_cast<ParticleEmitterComponent *>(obj->getObjectComponent());
            meta_vel = emitterComponent->emitDirection * emitterComponent->emitVel +
                       emitterComponent->emitDirection * emitterComponent->emitAcc * m_host_data->dt;
        }
        std::vector<float3> vel_tmp(part_num, meta_vel);

        auto meta_mat = obj->getObjectComponentConfig()["epmMaterial"].get<int>();
        if (obj->getObjectComponent()->getType() == Particle_Emitter)
            meta_mat = EPM_PE_PREPARE;
        std::vector<int> mat_tmp(part_num, meta_mat);

        std::vector<float> vol_frac_tmp(part_num * m_host_data->phase_num);
        std::vector<float> phasePerm_tmp(part_num * m_host_data->phase_num);
        std::vector<float> phaseCap_tmp(part_num * m_host_data->phase_num);
        float3 color_start = {0, 0, 0};
        for (int i = 0; i < m_host_data->phase_num; ++i) {
            color_start += obj->getSolverObjectComponentConfig()["phaseFraction"].get<std::vector<float>>()[i] *
                           make_cuFloat3(
                                   m_configData["MCT"]["Required"]["phaseRestColor"].get<std::vector<std::vector<float>>>()[i]);
            for (int j = 0; j < part_num; ++j) {
                vol_frac_tmp[j * m_host_data->phase_num + i] =
                        obj->getSolverObjectComponentConfig()["phaseFraction"].get<std::vector<float>>()[i];
            }
        }

        if (obj->getObjectComponentConfig()["epmMaterial"].get<int>() == EPM_FLUID) {
            std::vector<int> minerFlag_tmp(part_num, 0);
            generateMiners(minerFlag_tmp,
                           std::floor(part_num * obj->getSolverObjectComponentConfig()["fPartMinerRate"].get<float>()));
            m_host_digging_fPart_minerFlag.insert(m_host_digging_fPart_minerFlag.end(), minerFlag_tmp.begin(),
                                                  minerFlag_tmp.end());
            m_host_digging_pos.insert(m_host_digging_pos.end(), pos_tmp.begin(), pos_tmp.end());
            m_host_digging_mat.insert(m_host_digging_mat.end(), mat_tmp.begin(), mat_tmp.end());
            std::vector<int> pPart_alive(part_num, 0);
            m_host_digging_pPart_alive.insert(m_host_digging_pPart_alive.end(), pPart_alive.begin(), pPart_alive.end());
            std::vector<float> porosity_tmp(part_num, 0);
            m_host_digging_porosity.insert(m_host_digging_porosity.end(), porosity_tmp.begin(), porosity_tmp.end());
            m_fluid_part_num += pos_tmp.size();
        }

        if (obj->getObjectComponentConfig()["epmMaterial"].get<int>() == EPM_POROUS) {
            json obj_comp_config = obj->getObjectComponentConfig();
            json solver_comp_config = obj->getSolverObjectComponentConfig();
            std::vector<float3> pPart_hr_tmp;
            if (static_cast<uint8_t>(obj->getObjectComponent()->getType()) == 0) {
                if (!solver_comp_config["hrModelPath"].get<std::string>().empty()) {
                    obj_comp_config["particleGeometryPath"] = solver_comp_config["hrModelPath"];
                }
                pPart_hr_tmp = ModelHandler::generateObjectElements(obj_comp_config);
            } else {
                obj_comp_config["particleRadius"] = obj_comp_config["particleRadius"].get<float>() /
                                                    m_configData["MCT"]["Required"]["porousHRRate"].get<float>();
                pPart_hr_tmp = ModelHandler::generateObjectElements(obj_comp_config);
            }
            m_host_digging_pPart_hrPos.insert(m_host_digging_pPart_hrPos.end(), pPart_hr_tmp.begin(),
                                              pPart_hr_tmp.end());
            std::vector<int> minerFlag_tmp(pPart_hr_tmp.size(), 0);
            m_host_digging_fPart_minerFlag.insert(m_host_digging_fPart_minerFlag.end(), minerFlag_tmp.begin(),
                                                  minerFlag_tmp.end());
            m_host_digging_pos.insert(m_host_digging_pos.end(), pPart_hr_tmp.begin(), pPart_hr_tmp.end());
            std::vector<int> hr_mat_tmp(pPart_hr_tmp.size(), EPM_POROUS);
            m_host_digging_mat.insert(m_host_digging_mat.end(), hr_mat_tmp.begin(), hr_mat_tmp.end());
            std::vector<int> pPart_alive(pPart_hr_tmp.size(), 1);
            m_host_digging_pPart_alive.insert(m_host_digging_pPart_alive.end(), pPart_alive.begin(), pPart_alive.end());
            std::vector<float> porosity_tmp(pPart_hr_tmp.size(),
                                            m_configData["MCT"]["Required"]["porousPorosity"].get<float>());
            m_host_digging_porosity.insert(m_host_digging_porosity.end(), porosity_tmp.begin(), porosity_tmp.end());
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

    bool MCTSolver::attachObjects(std::vector<Object *> objs) {
        for (auto obj: objs) {
            if (!attachObject(obj)) {
                return false;
            }
        }

        return true;
    }

    bool MCTSolver::reset() {
        m_configData = JsonHandler::loadMCTConfigTemplateJson();
        return true;
    }

    void MCTSolver::destroy() {
        m_host_data->free();
        delete m_host_data;
        cudaFree(m_device_data);
        m_attached_objs.clear();
        m_neighborSearcher.freeMemory();
        m_neighborSearcher_digging.freeMemory();

        LOG_INFO("MCTSolver Destroyed.");
    }

    void MCTSolver::exportData() {
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
                LOG_ERROR("MCTSolver exportGroupPolicy" +
                          exportConfig["SolverRequired"]["exportGroupPolicy"].get<std::string>() + " not supported.");
            }

            m_doExportFlag = false;
        }
    }

    bool MCTSolver::tick() {
        static const float export_gap = 1 / m_configData["EXPORT"]["SolverRequired"]["exportFps"].get<float>();
        static const std::vector<int> exportObjectStartIndex =
                m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].get<std::vector<int>>();
        static const std::vector<int> exportObjectEndIndex =
                m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"].get<std::vector<int>>();
        static const std::vector<int> exportObjectMats =
                m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"].get<std::vector<int>>();

        dynamicUpdatingObjects();

        // NOTE: custom step, not necessary
        if (m_host_data->cur_simTime > 1)
            stir_fan(m_host_data,
                     m_device_data,
                     m_neighborSearcher.m_params_cuData);

        m_neighborSearcher.update(m_host_data->pos);

        sph_precompute(m_host_data,
                       m_device_data,
                       m_neighborSearcher.m_config_cuData,
                       m_neighborSearcher.m_params_cuData);

        // NOTE: bug step, used with caution
        if (m_configData["MCT"]["Required"]["enablePorous"].get<bool>()) {
            apply_porous_medium(m_host_data,
                                m_device_data,
                                m_neighborSearcher.m_config_cuData,
                                m_neighborSearcher.m_params_cuData);
        }

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

        apply_adhesion_force(m_host_data,
                             m_device_data,
                             m_neighborSearcher.m_config_cuData,
                             m_neighborSearcher.m_params_cuData);

        mct_gravity_surface(m_host_data,
                            m_device_data,
                            m_neighborSearcher.m_config_cuData,
                            m_neighborSearcher.m_params_cuData);

        mpct(m_host_data,
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

        std::vector<float> dens(m_host_data->particle_num);
        std::vector<float> comp(m_host_data->particle_num);
        std::vector<float> bf(m_host_data->particle_num);
        std::vector<float> mass(m_host_data->particle_num);
        std::vector<int> nsn(m_host_data->particle_num);
        std::vector<int> ns(m_host_data->particle_num * 60);
        cudaMemcpy(dens.data(), m_host_data->rest_density, m_host_data->particle_num * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(comp.data(), m_host_data->compression_ratio, m_host_data->particle_num * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(bf.data(), m_host_data->delta_compression_ratio, m_host_data->particle_num * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(mass.data(), m_host_data->mass, m_host_data->particle_num * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(nsn.data(), m_neighborSearcher.m_params_hostData.neighborNum_cuData,
                   m_host_data->particle_num * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(ns.data(), m_neighborSearcher.m_params_hostData.neighbors_cuData,
                   m_host_data->particle_num * 60 * sizeof(int), cudaMemcpyDeviceToHost);


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

            if (m_configData["MCT"]["Required"]["enableDigging"].get<bool>())
                digging_porous();
        }

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("MCTSolver tick failed.");
            m_isCrashed = true;
            return false;
        }
        return true;
    }

    bool MCTSolver::checkConfig() const {
        if (!m_configData.contains("MCT")) {
            LOG_ERROR("MCT config missing main domain: MCT")
            return false;
        }

        auto mct_config = m_configData["MCT"];
        if (!mct_config.contains("Required")) {
            LOG_ERROR("MCT config missing necessary domain: Required")
            return false;
        }

        for (const auto &key: MCTConfigRequiredKeys) {
            if (!mct_config["Required"].contains(key)) {
                LOG_ERROR("MCT config missing Required key: " + key)
                return false;
            }
        }

        if (mct_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size() !=
            mct_config["Required"]["phaseRestViscosity"].get<std::vector<float>>().size()
            &&
            mct_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size() !=
            mct_config["Required"]["phaseRestColor"].get<std::vector<std::vector<float>>>().size()
            &&
            mct_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size() !=
            mct_config["Required"]["phaseRelaxationTime"].get<std::vector<std::vector<float>>>().size()
            &&
            mct_config["Required"]["phaseRestDensity"].get<std::vector<float>>().size() !=
            mct_config["Required"]["phaseThinningFactor"].get<std::vector<std::vector<float>>>().size()
                ) {
            LOG_ERROR("MCT config phase-related parameter size not match.");
            return false;
        }

        for (const auto &key: MCTConfigOptionalKeys) {
            if (!mct_config["Optional"].contains(key)) {
                LOG_ERROR("MCT config missing Optional key: " + key)
                return false;
            }
        }

        return true;
    }

    void MCTSolver::generateMiners(std::vector<int> &fParts, int minerNum) {
        std::random_device rd;  // 随机设备
        std::mt19937 gen(rd()); // 随机数生成器
        std::uniform_int_distribution<> dist(0, fParts.size() - 1);

        // 随机挑选 30 个不同的索引
        std::vector<int> indices(fParts.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i; // 初始化索引
        }

        // 打乱索引
        std::shuffle(indices.begin(), indices.end(), gen);

        // 挑选前 30 个索引并设置为 true
        for (int i = 0; i < minerNum; ++i) {
            fParts[indices[i]] = 1;
        }
    }

    void MCTSolver::digging_porous() {
        static auto &exportConfig = m_configData["EXPORT"];

        int cpy_start_index = 0;
        for (int i = 0; i < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++i) {
            auto cur_mat = exportConfig["SolverRequired"]["exportObjectMaterials"][i].get<int>();
            if (cur_mat == EPM_FLUID) {
                int data_size = exportConfig["SolverRequired"]["exportObjectEndIndex"][i].get<int>() -
                                exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>();
                std::vector<float3> fPart_pos(data_size);
                cudaMemcpy(m_host_data->digging_pos + cpy_start_index,
                           m_host_data->pos +
                           exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>(),
                           data_size * sizeof(float3),
                           cudaMemcpyDeviceToDevice);
                cpy_start_index += data_size;
            }
        }

        m_neighborSearcher_digging.update(m_host_data->digging_pos);
        digging(m_host_data,
                m_device_data,
                m_neighborSearcher_digging.m_config_cuData,
                m_neighborSearcher_digging.m_params_cuData);
    }

    void MCTSolver::dynamicUpdatingObjects() {
#ifdef VP_TURN_ON_OMP
#pragma omp parallel for
        for (int i = 0; i < m_attached_objs.size(); ++i)
            m_attached_objs[i]->getObjectComponent()->dynamicUpdate(m_host_data->cur_simTime);
#else
        for (auto &obj: m_attached_objs)
            obj->getObjectComponent()->dynamicUpdate(m_host_data->cur_simTime);
#endif
    }

}