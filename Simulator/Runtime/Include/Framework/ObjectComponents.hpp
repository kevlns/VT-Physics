/**
 * @brief Todo
 * @date 2024/10/29
 */

#ifndef VT_PHYSICS_OBJECTCOMPONENTS_HPP
#define VT_PHYSICS_OBJECTCOMPONENTS_HPP

#include <vector>
#include <vector_types.h>

#include "nlohmann/json.hpp"
#include "Logger/Logger.hpp"
#include "Model/ModelHandler.hpp"

namespace VT_Physics {

    using json = nlohmann::json;

    typedef enum eObjectType : uint8_t {
        Particle_Common = 0,
        Particle_Sphere = 1,
        Particle_Cube = 2,
        Particle_Cylinder = 3,
        Particle_Plane = 4,
        Particle_Box = 5,

        MeshGeometry = 20,
        Mesh_Sphere = 21,
        Mesh_Cube = 22,
        Mesh_Cylinder = 23,
        Mesh_Plane = 24,
        Mesh_Particle = 25,

        OBJ_NULL_TYPE = UINT8_MAX
    } ObjectType;

    struct ObjectTypeComponent {
        virtual ObjectType getType() const = 0;

        virtual ObjectTypeComponent *clone() const = 0;

        virtual json getConfig() = 0;

        virtual void update(json &config) = 0;

        virtual std::vector<float> getElements() = 0;
    };

    struct ParticleGeometryComponent : public ObjectTypeComponent {
        // TODO
        //        float3 massCenter{0, 0, 0};
        //        float3 geoCenter{0, 0, 0};
        std::vector<float3> pos;
        float particleRadius{0.05f};
        uint8_t epmMaterial{UINT8_MAX};
        std::string particleGeometryPath;

        ObjectType getType() const override;

        ObjectTypeComponent *clone() const override;

        json getConfig() override;

        void update(json &config) override;

        std::vector<float> getElements() override;

    private:
        bool configIsComplete(json &config);

    private:
        const ObjectType m_type = ObjectType::Particle_Common;
    };

    struct ParticleSphereComponent : public ParticleGeometryComponent {
        float3 volumeCenter{0, 0, 0};
        float volumeRadius{1};

        ObjectType getType() const override;

        ObjectTypeComponent *clone() const override;

        json getConfig() override;

        void update(json &config) override;

    private:
        bool configIsComplete(json &config);

    private:
        const ObjectType m_type = ObjectType::Particle_Sphere;
    };

    struct ParticleCubeComponent : public ParticleGeometryComponent {
        float3 lb{-1, -1, -1};
        float3 size{2, 2, 2};

        ObjectType getType() const override;

        ObjectTypeComponent *clone() const override;

        json getConfig() override;

        virtual void update(json &config) override;

    private:
        bool configIsComplete(json &config);

    private:
        const ObjectType m_type = ObjectType::Particle_Cube;
    };

    struct ParticleCylinderComponent : public ParticleGeometryComponent {
        float3 center{0, 0, 0};
        float bottomAreaRadius{1};
        float height{2};

        ObjectType getType() const override;

        ObjectTypeComponent *clone() const override;

        json getConfig() override;

        void update(json &config) override;

    private:
        bool configIsComplete(json &config);

    private:
        const ObjectType m_type = ObjectType::Particle_Cylinder;
    };

    struct ParticlePlaneComponent : public ParticleGeometryComponent {
        float3 lb{0, 1, 0};
        float3 size{0, 0, 0};
        uint32_t layerNum{2};

        ObjectType getType() const override;

        ObjectTypeComponent *clone() const override;

        json getConfig() override;

        void update(json &config) override;

    private:
        bool configIsComplete(json &config);

    private:
        const ObjectType m_type = ObjectType::Particle_Plane;
    };

    struct ParticleBoxComponent : public ParticleGeometryComponent {
        float3 lb{-1, -1, -1};
        float3 size{2, 2, 2};
        uint32_t layerNum{1};

        ObjectType getType() const override;

        ObjectTypeComponent *clone() const override;

        json getConfig() override;

        virtual void update(json &config) override;

    private:
        bool configIsComplete(json &config);

    private:
        const ObjectType m_type = ObjectType::Particle_Box;
    };

    struct MeshGeometryComponent : ObjectTypeComponent {
        // TODO
        //        float3 geoCenter{0, 0, 0};

        ObjectType getType() const override;

        ObjectTypeComponent *clone() const override;

        json getConfig() override;

        void update(json &config) override;

        std::vector<float> getElements() override;

    private:
        bool configIsComplete(json &config);

    private:
        const ObjectType m_type = ObjectType::MeshGeometry;
    };

    inline const std::vector<ObjectTypeComponent *> componentTemplates = {
            new ParticleGeometryComponent(),
            new ParticleSphereComponent(),
            new ParticleCubeComponent(),
            new ParticleCylinderComponent(),
            new ParticlePlaneComponent(),
            new ParticleBoxComponent(),

            new MeshGeometryComponent(),
    };

}

#endif //VT_PHYSICS_OBJECTCOMPONENTS_HPP
