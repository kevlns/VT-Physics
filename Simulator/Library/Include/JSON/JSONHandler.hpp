/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_JSONHANDLER_HPP
#define VT_PHYSICS_JSONHANDLER_HPP

#include <fstream>

#include "nlohmann/json.hpp"

namespace VT_Physics {

    using json = nlohmann::json;

    class JsonHandler {
    public:
        static json loadJson(const std::string &path);

        static bool saveJson(const std::string &path, const json &j);

        static json loadExportConfigTemplateJson();

        static json loadPBFConfigTemplateJson();

        static json loadPBFObjectComponentConfigTemplateJson();

        static json loadDFSPHConfigTemplateJson();

        static json loadDFSPHObjectComponentConfigTemplateJson();

        static json loadIMMConfigTemplateJson();

        static json loadIMMObjectComponentConfigTemplateJson();

        static json loadIMMCTConfigTemplateJson();

        static json loadIMMCTObjectComponentConfigTemplateJson();

        static json loadUGNSConfigTemplateJson();

        static json loadParticleCubeConfigTemplateJson();

        static json loadParticleSphereConfigTemplateJson();

        static json loadParticleCylinderConfigTemplateJson();

        static json loadParticlePlaneConfigTemplateJson();

        static json loadParticleBoxConfigTemplateJson();

        static json loadParticleCommonConfigTemplateJson();
    };

}

#endif //VT_PHYSICS_JSONHANDLER_HPP
