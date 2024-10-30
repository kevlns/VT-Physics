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
        return loadJson(std::string(VP_CONFIG_DIR)+ "ExportConfigTemplate.json");
    }

    json JsonHandler::loadPBFConfigTemplateJson() {
        return loadJson(std::string(VP_CONFIG_DIR)+ "PBFConfigTemplate.json");
    }
}