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
    };

}

#endif //VT_PHYSICS_JSONHANDLER_HPP
