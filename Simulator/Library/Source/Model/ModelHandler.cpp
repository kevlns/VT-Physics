#include "Model/ModelHandler.hpp"

#include <algorithm>

#include "Logger/Logger.hpp"
#include "Model/ExportUtil.hpp"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

namespace VT_Physics {

    std::vector<float3> ModelHandler::generateObjectElements(json config) {
        if (!config.contains("genType")) {
            LOG_ERROR("No generation type specified.");
            return {};
        }

        switch (static_cast<uint8_t>(config["genType"])) {
            case 0:
                // TODO load particle model
                return {};
            case 1:
                return generateParticleSphereElements(config);
            case 2:
                return generateParticleCubeElements(config);
            case 3:
                return generateParticleCylinderElements(config);
            case 4:
                return generateParticlePlaneElements(config);
            case 5:
                return generateParticleBoxElements(config);

            case 20:
                // TODO load mesh model
                return {};
            default:
                LOG_ERROR("Unknown or Unsupported generation type specified.");
                return {};
        };
    }

    std::vector<float3> ModelHandler::generateParticleCubeElements(json config) {
        std::vector<float3> particles;
        auto particleRadius = config["particleRadius"].get<float>();
        auto lb = config["lb"].get<std::vector<float>>();
        auto size = config["size"].get<std::vector<float>>();
        float diameter = 2 * particleRadius;

        float z = particleRadius + lb[2];
        while (z < lb[2] + size[2]) {
            float y = particleRadius + lb[1];
            while (y < lb[1] + size[1]) {
                float x = particleRadius + lb[0];
                while (x < lb[0] + size[0]) {
                    float3 particle = {x, y, z};
                    particles.push_back(particle);

                    x += diameter;
                }
                y += diameter;
            }
            z += diameter;
        }

        return std::move(particles);
    }

    std::vector<float3> ModelHandler::generateParticleCylinderElements(json config) {
        std::vector<float3> particles;
        auto particleRadius = config["particleRadius"].get<float>();
        auto center = config["center"].get<std::vector<float>>();
        auto bottomAreaRadius = config["bottomAreaRadius"].get<float>();
        auto height = config["height"].get<float>();
        float diameter = 2 * particleRadius;

        float3 topCenter = {center[0], center[1] + height / 2, center[2]};
        float y0 = topCenter.y;

        for (float y = y0 - particleRadius; y >= y0 - height; y -= diameter) {
            float x0 = topCenter.x - bottomAreaRadius;

            for (float x = x0 + particleRadius; x <= topCenter.x + bottomAreaRadius; x += diameter) {

                float m_cos = fabs(topCenter.x - x) / bottomAreaRadius;
                float length = bottomAreaRadius * sqrt(1 - m_cos * m_cos);
                float z0 = topCenter.z - length;
                for (float z = z0 + particleRadius; z <= topCenter.z + length; z += diameter) {
                    float3 particle = {x, y, z};
                    particles.push_back(particle);
                }
            }
        }

        return std::move(particles);
    }

    std::vector<float3> ModelHandler::generateParticleSphereElements(json config) {
        std::vector<float3> particles;
        auto particleRadius = config["particleRadius"].get<float>();
        auto center = config["volumeCenter"].get<std::vector<float>>();
        auto volumeRadius = config["volumeRadius"].get<float>();
        float gap = particleRadius * 2.0f;

        int num_particles_per_side = std::ceil(volumeRadius / gap);
        for (int i = -num_particles_per_side; i <= num_particles_per_side; ++i) {
            for (int j = -num_particles_per_side; j <= num_particles_per_side; ++j) {
                for (int k = -num_particles_per_side; k <= num_particles_per_side; ++k) {
                    float3 particle = {float(i) * gap + center[0], float(j) * gap + center[1],
                                       float(k) * gap + center[2]};

                    if ((particle.x - center[0]) * (particle.x - center[0]) +
                        (particle.y - center[1]) * (particle.y - center[1]) +
                        (particle.z - center[2]) * (particle.z - center[2]) <= volumeRadius * volumeRadius) {
                        particles.push_back(particle);
                    }
                }
            }
        }

        return std::move(particles);
    }

    std::vector<float3> ModelHandler::generateParticleBoxElements(json config) {
        std::vector<float3> particles;
        auto particleRadius = config["particleRadius"].get<float>();
        auto lb = config["lb"].get<std::vector<float>>();
        auto size = config["size"].get<std::vector<float>>();
        auto layerNum = config["layerNum"].get<int>();
        int numParticles[] = {
                static_cast<int>((size[0] + particleRadius) / (2.0 * particleRadius)),
                static_cast<int>((size[1] + particleRadius) / (2.0 * particleRadius)),
                static_cast<int>((size[2] + particleRadius) / (2.0 * particleRadius))
        };

        for (int i = 0; i <= numParticles[0]; ++i) {
            for (int j = 0; j <= numParticles[1]; ++j) {
                for (int k = 0; k <= numParticles[2]; ++k) {
                    // If this particle is in the first two or last two layers in any dimension...
                    if (i < layerNum || i >= numParticles[0] - layerNum || j < layerNum ||
                        j >= numParticles[1] - layerNum ||
                        k < layerNum || k >= numParticles[2] - layerNum) {
                        float3 particle;
                        particle.x = static_cast<float>(lb[0] + particleRadius + 2.0 * particleRadius * i);
                        particle.y = static_cast<float>(lb[1] + particleRadius + 2.0 * particleRadius * j);
                        particle.z = static_cast<float>(lb[2] + particleRadius + 2.0 * particleRadius * k);
                        particles.push_back(particle);
                    }
                }
            }
        }

        return std::move(particles);
    }

    std::vector<float3> ModelHandler::generateParticlePlaneElements(json config) {
        std::vector<float3> particles;
        auto particleRadius = config["particleRadius"].get<float>();
        auto lb = config["lb"].get<std::vector<float>>();
        auto size = config["size"].get<std::vector<float>>();
        auto layerNum = config["layerNum"].get<int>();
        auto diameter = 2 * particleRadius;

        for (float z = particleRadius + lb[2]; z < lb[2] + size[2]; z += diameter) {
            for (float y = particleRadius + lb[1], cnt = 0; cnt < layerNum; y += diameter, cnt += 1) {
                for (float x = particleRadius + lb[0]; x < lb[0] + size[0]; x += diameter) {
                    float3 particle = {x, y, z};

                    particles.push_back(particle);
                }
            }
        }

        return std::move(particles);
    }

    std::vector<float3> ModelHandler::loadObjectElements(json config) {
        if (config["loadType"] < 20) {
            return std::move(loadParticleElements(config));
        } else {
            // TODO
            return {};
        }
    }

    std::string ModelHandler::getFileExtensionUpper(const std::string &filePath) {
        size_t dotPosition = filePath.find_last_of(".");
        if (dotPosition != std::string::npos && dotPosition != 0 && dotPosition != filePath.length() - 1) {
            auto postfix = filePath.substr(dotPosition + 1);
            std::transform(postfix.begin(), postfix.end(), postfix.begin(), ::toupper);
            return postfix;
        }
        return ""; // 如果没有找到后缀，返回空字符串
    }

    std::vector<float3> ModelHandler::loadParticleElements(json config) {
        auto filePath = config["particleGeometryPath"].get<std::string>();
        if (supportedFileType.count(getFileExtensionUpper(filePath)) != 0) {
            if (getFileExtensionUpper(filePath) == "PLY")
                return std::move(load_PLY_ParticleElements(filePath));
        }

        LOG_ERROR("Unsupported file type: " + getFileExtensionUpper(filePath))
        return {};
    }

    std::vector<float3> ModelHandler::load_PLY_ParticleElements(std::string filePath) {
        std::vector<float3> particles;

        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(filePath, aiProcess_Triangulate);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            LOG_ERROR(importer.GetErrorString());
            return {};
        }

        for (unsigned int m = 0; m < scene->mNumMeshes; m++) {
            const aiMesh *mesh = scene->mMeshes[m];
            for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
                const aiVector3D &vertex = mesh->mVertices[v];
                float3 particle = {vertex.x, vertex.y, vertex.z};
                particles.emplace_back(particle);
            }
        }

        return std::move(particles);
    }

}