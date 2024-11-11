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

    inline const std::vector<std::string> exportConfigRequiredCommonKeys = {
            "exportTargetDir",
            "exportFilePrefix",
            "exportFileType"
    };

    inline const std::vector<std::string> exportConfigRequiredSolverRequiredKeys = {
            "enable",
            "exportFps",
            "exportGroupPolicy",
            "exportObjectMaterials",
            "exportObjectStartIndex",
            "exportObjectEndIndex",
            "exportFlags"
    };

    inline const std::set<std::string> supportedFileType = {
            "PLY"
    };

    inline const std::set<std::string> supportedExportPolicy = {
            "MERGE",
            "SPLIT"
    };

    class ExportUtil {
    public:
        static void exportData(const json &exportConfig,
                               const std::vector<float3> &pos,
                               const std::vector<float3> &color = {},
                               const std::vector<float> &phase = {});

        static void exportData(const json &exportConfig,
                               std::vector<float> &pos);

        static bool checkConfig(const json &config);

    private:

        static void exportAsPly(const json &exportConfig,
                                const std::vector<float3> &pos,
                                const std::vector<float3> &color = {},
                                const std::vector<float> &phase = {});
    };

}

#endif //VT_PHYSICS_EXPORTUTIL_HPP
