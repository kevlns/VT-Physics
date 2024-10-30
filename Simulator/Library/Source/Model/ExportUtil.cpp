#include "Model/ExportUtil.hpp"

#include <fstream>

#include "Logger/Logger.hpp"

namespace VT_Physics {

    void ExportUtil::exportData(const json &exportConfig,
                                const std::vector<float3> &pos,
                                const std::vector<float3> &color,
                                const std::vector<float> &phase) {
        LOG_WARNING("Exporting data...");
        if (exportConfig["SolverRequired"]["enable"]) {
            // TODO
        } else {
            auto exportCommonConfig = exportConfig["Common"];
            if (supportedFileType.count(exportCommonConfig["exportFileType"]) != 0) {
                if (exportCommonConfig["exportFileType"] == "PLY")
                    exportAsPly(exportCommonConfig, pos, color, phase);
            } else {
                LOG_ERROR("Export type not supported.");
                return;
            }
        }
    }

    void ExportUtil::exportData(const json &exportConfig, std::vector<float> &pos) {
        auto pos_f3ptr = reinterpret_cast<float3 *>(pos.data());
        exportData(exportConfig, std::vector<float3>(pos_f3ptr, pos_f3ptr + pos.size() / 3));
    }

    void ExportUtil::exportAsPly(const json &exportConfig,
                                 const std::vector<float3> &pos,
                                 const std::vector<float3> &color,
                                 const std::vector<float> &phase) {
        auto dir_ = exportConfig["exportTargetDir"].get<std::string>();
        auto name = exportConfig["exportFilePrefix"].get<std::string>();
        const bool exportColor = !color.empty();
        const bool exportPhase = !phase.empty();

#ifdef WIN32
        size_t fPos = 0;
        while ((fPos = dir_.find('/', fPos)) != std::string::npos) {
            dir_.replace(fPos, 1, "\\");
            fPos += 1;
        }
#endif

        if (!std::filesystem::exists(dir_))
            std::filesystem::create_directories(dir_);

        auto target_file_path = dir_ + "\\" + name + ".ply";
        std::ofstream target_file(target_file_path);

        target_file << "ply\n";
        target_file << "format ascii 1.0\n";
        target_file << "element vertex " << pos.size() << "\n";
        target_file << "property float x\n";
        target_file << "property float y\n";
        target_file << "property float z\n";

        if (exportColor) {
            target_file << "property uchar red" << std::endl;
            target_file << "property uchar green" << std::endl;
            target_file << "property uchar blue" << std::endl;
        }

        // TODO if multiphase, set phase num from config
        //        if (!phase.empty()) {
        //            target_file << "property float p1" << std::endl;
        //            target_file << "property float p2" << std::endl;
        //            target_file << "property float p3" << std::endl;
        //        }

        target_file << "end_header\n";

        for (int i = 0; i < pos.size(); ++i) {
            target_file << pos[i].x << " " << pos[i].y << " " << pos[i].z << " ";
            if (exportColor)
                target_file << color[i].x << " " << color[i].y << " " << color[i].z << " ";
            // TODO export phase
            // if (exportPhase) threadFile << phase[ind] << " ";
            target_file << "\n";
        }

        target_file.close();
        LOG_INFO("Exported file: " + target_file_path);
    }

    bool ExportUtil::checkConfig(const json &config) {
        for (const auto &key: exportConfigKeys) {
            if (!config.contains(key)) {
                LOG_ERROR("Export Config missing key: " + key);
                return false;
            }
        }
        return true;
    }
}