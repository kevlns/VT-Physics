#include "Framework/ObjectComponents.hpp"

namespace VT_Physics {

    /**
     * ObjectTypeComponent impl ===========================================================
     */
    ObjectType ObjectTypeComponent::getType() const { return ObjectType::MeshGeometry; }

    ObjectTypeComponent *ObjectTypeComponent::clone() const { return nullptr; }

    json ObjectTypeComponent::getConfig() { return {}; }

    void ObjectTypeComponent::update(json &config) {}

    std::vector<float> ObjectTypeComponent::getElements() { return {}; }
    // ====================================================================================

#define FILL_CLONE(type) \
auto ret = new type();\
    memcpy_s(ret, sizeof(type), \
    this, sizeof(type)); \
    return static_cast<ObjectTypeComponent *>(ret);

    /**
     * ParticleGeometryComponent impl ===========================================================
     */
    ObjectType ParticleGeometryComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *ParticleGeometryComponent::clone() const {
        FILL_CLONE(ParticleGeometryComponent);
    }

    json ParticleGeometryComponent::getConfig() {
        json ret = {
                {"epmMaterial",          epmMaterial},
                {"particleRadius",       particleRadius},
                {"particleGeometryPath", particleGeometryPath}
        };
        return ret;
    }

    void ParticleGeometryComponent::update(json &config) {
        if (configIsComplete(config)) {
            config["genType"] = static_cast<uint8_t>(m_type);
            if (config.contains("epmMaterial"))
                epmMaterial = config["epmMaterial"];
            particleRadius = config["particleRadius"];
            particleGeometryPath = config["particleGeometryPath"];
            auto p = ModelHandler::generateObjectElements(config);
            pos = p;
            LOG_INFO("ParticleGeometryComponent config updated.");
        }
    }

    std::vector<float> ParticleGeometryComponent::getElements() {
        auto pos_fptr = reinterpret_cast<float *>(pos.data());
        auto size = pos.size();
        auto ret = std::vector<float>(pos_fptr, pos_fptr + 3 * size);
        return std::move(ret);
    }

    bool ParticleGeometryComponent::configIsComplete(json &config) {
        bool ret = true;
        if (!config.contains("particleRadius")) {
            LOG_ERROR("ParticleGeometryComponent config missing key: particleRadius");
            ret = false;
        }
        if (!config.contains("epmMaterial")) LOG_WARNING("ParticleGeometryComponent config missing key: epmMaterial");
        return ret;
    }
    // ====================================================================================


    /**
     * ParticleSphereComponent impl ===========================================================
     */
    ObjectType ParticleSphereComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *ParticleSphereComponent::clone() const {
        FILL_CLONE(ParticleSphereComponent);
    }

    json ParticleSphereComponent::getConfig() {
        json ret = {
                {"epmMaterial",    epmMaterial},
                {"particleRadius", particleRadius},
                {"volumeCenter",   {volumeCenter.x, volumeCenter.y, volumeCenter.z}},
                {"volumeRadius",   volumeRadius}
        };
        return ret;
    }

    void ParticleSphereComponent::update(json &config) {
        if (configIsComplete(config)) {
            config["genType"] = static_cast<uint8_t>(m_type);
            if (config.contains("epmMaterial"))
                epmMaterial = config["epmMaterial"].get<uint8_t>();
            particleRadius = config["particleRadius"];
            volumeCenter = {config["volumeCenter"][0], config["volumeCenter"][1], config["volumeCenter"][2]};
            volumeRadius = config["volumeRadius"];
            auto p = ModelHandler::generateObjectElements(config);
            pos = p;
            LOG_INFO("ParticleSphereComponent config updated.");
        }
    }

    bool ParticleSphereComponent::configIsComplete(json &config) {
        bool ret = true;
        if (!config.contains("particleRadius")) {
            LOG_ERROR("ParticleSphereComponent config missing key: particleRadius");
            ret = false;
        }
        if (!config.contains("volumeCenter")) {
            LOG_ERROR("ParticleSphereComponent config missing key: volumeCenter");
            ret = false;
        }
        if (!config.contains("volumeRadius")) {
            LOG_ERROR("ParticleSphereComponent config missing key: volumeRadius");
            ret = false;
        }
        if (!config.contains("epmMaterial")) LOG_WARNING("ParticleSphereComponent config missing key: epmMaterial");

        return ret;
    }
    // ====================================================================================


    /**
     * ParticleCubeComponent impl ===========================================================
     */
    ObjectType ParticleCubeComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *ParticleCubeComponent::clone() const {
        FILL_CLONE(ParticleCubeComponent);
    }

    json ParticleCubeComponent::getConfig() {
        json ret = {
                {"epmMaterial",    epmMaterial},
                {"particleRadius", particleRadius},
                {"lb",             {lb.x,   lb.y,   lb.z}},
                {"size",           {size.x, size.y, size.z}}
        };
        return ret;
    }

    void ParticleCubeComponent::update(json &config) {
        if (configIsComplete(config)) {
            config["genType"] = static_cast<uint8_t>(m_type);
            if (config.contains("epmMaterial"))
                epmMaterial = config["epmMaterial"].get<uint8_t>();
            particleRadius = config["particleRadius"];
            lb = {config["lb"][0], config["lb"][1], config["lb"][2]};
            size = {config["size"][0], config["size"][1], config["size"][2]};
            auto p = ModelHandler::generateObjectElements(config);
            pos = p;
            LOG_INFO("ParticleCubeComponent config updated.");
        }
    }

    bool ParticleCubeComponent::configIsComplete(json &config) {
        bool ret = true;
        if (!config.contains("particleRadius")) {
            LOG_ERROR("ParticleCubeComponent config missing key: particleRadius");
            ret = false;
        }
        if (!config.contains("lb")) {
            LOG_ERROR("ParticleCubeComponent config missing key: lb");
            ret = false;
        }
        if (!config.contains("size")) {
            LOG_ERROR("ParticleCubeComponent config missing key: size");
            ret = false;
        }
        if (!config.contains("epmMaterial")) LOG_WARNING("ParticleCubeComponent config missing key: epmMaterial");
        return ret;
    }
    // ====================================================================================


    /**
     * ParticleCylinderComponent impl ===========================================================
     */
    ObjectType ParticleCylinderComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *ParticleCylinderComponent::clone() const {
        FILL_CLONE(ParticleCylinderComponent);
    }

    json ParticleCylinderComponent::getConfig() {
        json ret = {
                {"epmMaterial",      epmMaterial},
                {"center",           {center.x, center.y, center.z}},
                {"bottomAreaRadius", bottomAreaRadius},
                {"height",           height},
        };
        return ret;
    }

    void ParticleCylinderComponent::update(json &config) {
        if (configIsComplete(config)) {
            config["genType"] = static_cast<uint8_t>(m_type);
            if (config.contains("epmMaterial"))
                epmMaterial = config["epmMaterial"].get<uint8_t>();
            particleRadius = config["particleRadius"];
            center = {config["center"][0], config["center"][1], config["center"][2]};
            bottomAreaRadius = config["bottomAreaRadius"];
            height = config["height"];
            auto p = ModelHandler::generateObjectElements(config);
            pos = p;
            LOG_INFO("ParticleCylinderComponent config updated.");
        }
    }

    bool ParticleCylinderComponent::configIsComplete(json &config) {
        bool ret = true;
        if (!config.contains("particleRadius")) {
            LOG_ERROR("ParticleCylinderComponent config missing key: particleRadius");
            ret = false;
        }
        if (!config.contains("center")) {
            LOG_ERROR("ParticleCylinderComponent config missing key: center");
            ret = false;
        }
        if (!config.contains("bottomAreaRadius")) {
            LOG_ERROR("ParticleCylinderComponent config missing key: bottomAreaRadius");
            ret = false;
        }
        if (!config.contains("height")) {
            LOG_ERROR("ParticleCylinderComponent config missing key: height");
            ret = false;
        }
        if (!config.contains("epmMaterial")) LOG_WARNING("ParticleCylinderComponent config missing key: epmMaterial");
        return ret;
    }
    // ====================================================================================


    /**
     * ParticlePlaneComponent impl ===========================================================
     */
    ObjectType ParticlePlaneComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *ParticlePlaneComponent::clone() const {
        FILL_CLONE(ParticlePlaneComponent);
    }

    json ParticlePlaneComponent::getConfig() {
        json ret = {
                {"epmMaterial", epmMaterial},
                {"lb",          {lb.x,   lb.y,   lb.z}},
                {"size",        {size.x, size.y, size.z}},
                {"layerNum",    layerNum}
        };
        return ret;
    }

    void ParticlePlaneComponent::update(json &config) {
        if (configIsComplete(config)) {
            config["genType"] = static_cast<uint8_t>(m_type);
            if (config.contains("epmMaterial"))
                epmMaterial = config["epmMaterial"].get<uint8_t>();
            particleRadius = config["particleRadius"];
            lb = {config["lb"][0], config["lb"][1], config["lb"][2]};
            size = {config["size"][0], config["size"][1], config["size"][2]};
            layerNum = config["layerNum"];
            auto p = ModelHandler::generateObjectElements(config);
            pos = p;
            LOG_INFO("ParticlePlaneComponent config updated.");
        }
    }

    bool ParticlePlaneComponent::configIsComplete(json &config) {
        bool ret = true;
        if (!config.contains("particleRadius")) {
            LOG_ERROR("ParticlePlaneComponent config missing key: particleRadius");
            ret = false;
        }
        if (!config.contains("lb")) {
            LOG_ERROR("ParticlePlaneComponent config missing key: lb");
            ret = false;
        }
        if (!config.contains("size")) {
            LOG_ERROR("ParticlePlaneComponent config missing key: size");
            ret = false;
        }
        if (!config.contains("layerNum")) {
            LOG_ERROR("ParticlePlaneComponent config missing key: layerNum");
            ret = false;
        }
        if (!config.contains("epmMaterial")) LOG_WARNING("ParticlePlaneComponent config missing key: epmMaterial");
        return ret;
    }
    // ====================================================================================


    /**
     * ParticleBoxComponent impl ===========================================================
     */
    ObjectType ParticleBoxComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *ParticleBoxComponent::clone() const {
        FILL_CLONE(ParticleBoxComponent);
    }

    json ParticleBoxComponent::getConfig() {
        json ret = {
                {"epmMaterial", epmMaterial},
                {"lb",          {lb.x,   lb.y,   lb.z}},
                {"size",        {size.x, size.y, size.z}},
                {"layerNum",    layerNum}
        };
        return ret;
    }

    void ParticleBoxComponent::update(json &config) {
        if (configIsComplete(config)) {
            config["genType"] = static_cast<uint8_t>(m_type);
            if (config.contains("epmMaterial"))
                epmMaterial = config["epmMaterial"].get<uint8_t>();
            lb = {config["lb"][0], config["lb"][1], config["lb"][2]};
            size = {config["size"][0], config["size"][1], config["size"][2]};
            layerNum = config["layerNum"];
            auto p = ModelHandler::generateObjectElements(config);
            pos = p;
            LOG_INFO("ParticleBoxComponent config updated.");
        }
    }

    bool ParticleBoxComponent::configIsComplete(json &config) {
        bool ret = true;
        if (!config.contains("lb")) {
            LOG_ERROR("ParticleBoxComponent config missing key: lb");
            ret = false;
        }
        if (!config.contains("size")) {
            LOG_ERROR("ParticleBoxComponent config missing key: size");
            ret = false;
        }
        if (!config.contains("layerNum")) {
            LOG_ERROR("ParticleBoxComponent config missing key: layerNum");
            ret = false;
        }
        if (!config.contains("epmMaterial")) LOG_WARNING("ParticleBoxComponent config missing key: epmMaterial");
        return ret;
    }
    // ====================================================================================


    /**
     * MeshGeometryComponent impl ===========================================================
     */
    ObjectType MeshGeometryComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *MeshGeometryComponent::clone() const {
        FILL_CLONE(MeshGeometryComponent);
    }

    json MeshGeometryComponent::getConfig() {
        json ret = {
        };
        return ret;
    }

    void MeshGeometryComponent::update(json &config) {
        // TODO
    }

    std::vector<float> MeshGeometryComponent::getElements() {
        return {};
    }

    bool MeshGeometryComponent::configIsComplete(json &config) {
        bool ret = true;
        // TODO
        return ret;
    }
    // ====================================================================================

}