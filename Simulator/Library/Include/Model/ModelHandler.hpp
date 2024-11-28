/**
 * @brief Todo
 * @date 2024/10/30
 */

#ifndef VT_PHYSICS_MODELHANDLER_HPP
#define VT_PHYSICS_MODELHANDLER_HPP

#include <vector>
#include <vector_types.h>

#include "JSON/JSONHandler.hpp"

namespace VT_Physics {

    class ModelHandler final {
    public:
        static std::vector<float3> generateObjectElements(json config);

    private:
        static std::vector<float3> loadObjectElements(json config);

        static std::vector<float3> generateParticleCubeElements(json config);

        static std::vector<float3> generateParticleCylinderElements(json config);

        static std::vector<float3> generateParticleSphereElements(json config);

        static std::vector<float3> generateParticleBoxElements(json config);

        static std::vector<float3> generateParticlePlaneElements(json config);

        static std::string getFileExtensionUpper(const std::string &filename);

        static std::vector<float3> loadParticleElements(json config);

        static std::vector<float3> load_PLY_ParticleElements(std::string filePath);

    };

}

#endif //VT_PHYSICS_MODELHANDLER_HPP
