/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_EXPORTUTIL_HPP
#define VT_PHYSICS_EXPORTUTIL_HPP

#include <vector_types.h>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace VT_Physics {

    using json = nlohmann::json;

    inline const std::vector<std::string> exportConfigKeys = {
            "enable",
            "exportPath",
            "exportFilePrefix",
            "exportFileType",
            "exportFps",
            "exportObjectMaterial",
            "exportGroupPolicy",
            "exportObjectsStartIndex"
    };

    inline const std::set<std::string> supportedFileType = {
            "PLY"
    };

    inline const std::set<std::string> supportedExportPolicy = {
            "MERGE",
            "SEPARATE"
    };

    class ExportUtil {
    public:
        static void exportData(const json &exportConfig,
                               const std::vector<float3> &pos,
                               const std::vector<float3> &color = {},
                               const std::vector<float> &phase = {});

        static void exportData(const json &exportConfig,
                               std::vector<float> &pos);

    private:
        static bool checkConfig(const json &config);

        static void exportAsPly(const json &exportConfig,
                                const std::vector<float3> &pos,
                                const std::vector<float3> &color = {},
                                const std::vector<float> &phase = {});
    };

}

#endif //VT_PHYSICS_EXPORTUTIL_HPP
