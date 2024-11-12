#include "JSON/JSONHandler.hpp"

#include "Logger/Logger.hpp"
#include "ConfigTemplates/RealPath.h"

namespace VT_Physics {

    json JsonHandler::loadJson(const std::string &path) {
        std::ifstream jsonFile(path);
        if (jsonFile.is_open()) {
            json j;
            jsonFile >> j;
            jsonFile.close();
            return j;
        }

        LOG_ERROR("Failed to load json file: " + path);
        return {};
    }

    bool JsonHandler::saveJson(const std::string &path, const json &j) {
        std::ofstream jsonFile(path);
        if (jsonFile.is_open()) {
            jsonFile << j.dump();
            jsonFile.close();
            return true;
        }

        LOG_ERROR("Failed to load json file: " + path);
        return false;
    }

    json JsonHandler::loadExportConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "ExportConfigTemplate.json");
    }

    json JsonHandler::loadUGNSConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "UGNS/UGNSConfigTemplate.json");
    }

    json JsonHandler::loadParticleCubeConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "ObjectConfig/ParticleCubeConfigTemplate.json");
    }

    json JsonHandler::loadParticleSphereConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "ObjectConfig/ParticleSphereConfigTemplate.json");
    }

    json JsonHandler::loadParticleCylinderConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "ObjectConfig/ParticleCylinderConfigTemplate.json");
    }

    json JsonHandler::loadParticlePlaneConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "ObjectConfig/ParticlePlaneConfigTemplate.json");
    }

    json JsonHandler::loadParticleBoxConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "ObjectConfig/ParticleBoxConfigTemplate.json");
    }

    json JsonHandler::loadParticleCommonConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "ParticleCommonConfigTemplate.json");
    }

    json JsonHandler::loadPBFConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "PBFSolver/PBFSolverConfigTemplate.json");
    }

    json JsonHandler::loadPBFObjectComponentConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "PBFSolver/PBFSolverObjectComponentConfigTemplate.json");
    }

    json JsonHandler::loadDFSPHConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "DFSPHSolver/DFSPHSolverConfigTemplate.json");
    }

    json JsonHandler::loadDFSPHObjectComponentConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "DFSPHSolver/DFSPHSolverObjectComponentConfigTemplate.json");
    }

    json JsonHandler::loadIMMConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "IMMSolver/IMMSolverConfigTemplate.json");
    }

    json JsonHandler::loadIMMObjectComponentConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR) + "IMMSolver/IMMSolverObjectComponentConfigTemplate.json");
    }

}