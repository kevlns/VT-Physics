#include "Framework/ObjectComponents.hpp"

#include "Core/Math/DataStructTransfer.hpp"

namespace VT_Physics {

    /**
     * ObjectTypeComponent impl ===========================================================
     */
    ObjectType ObjectTypeComponent::getType() const { return ObjectType::MeshGeometry; }

    ObjectTypeComponent *ObjectTypeComponent::clone() const { return nullptr; }

    json ObjectTypeComponent::getConfig() { return {}; }

    void ObjectTypeComponent::update(json &config) {}

    void ObjectTypeComponent::dynamicUpdate(float time) {}

    void ObjectTypeComponent::destroy() {}

    std::vector<float> ObjectTypeComponent::getElements() { return {}; }
    // ====================================================================================

#define FILL_CLONE(type) \
auto ret = new type();\
    memcpy_s(ret, sizeof(type), \
    this, sizeof(type)); \
    return static_cast<ObjectTypeComponent *>(ret);

    /**
     * ParticleEmitterComponent impl ===========================================================
     */
    ObjectType ParticleEmitterComponent::getType() const {
        return m_type;
    }

    ObjectTypeComponent *ParticleEmitterComponent::clone() const {
        FILL_CLONE(ParticleEmitterComponent);
    }

    json ParticleEmitterComponent::getConfig() {
        json ret = {
                {"epmMaterial",               epmMaterial},
                {"particleRadius",            particleRadius},
                {"agentParticleGeometryFile", agentParticleGeometryFile},
                {"agentEmitDirectionFile",    agentEmitDirectionFile},
                {"emitDirection",             {emitDirection.x, emitDirection.y, emitDirection.z}},
                {"emitVel",                   emitVel},
                {"emitAcc",                   emitAcc},
                {"emitParticleMaxNum",        emitParticleMaxNum},
                {"emitOnCudaDevice",          emitOnCudaDevice}
        };
        return ret;
    }

    void ParticleEmitterComponent::update(json &config) {
        if (configIsComplete(config)) {
            if (config.contains("epmMaterial"))
                epmMaterial = config["epmMaterial"];
            particleRadius = config["particleRadius"].get<float>();
            emitDirection = make_cuFloat3(config["emitDirection"].get<std::vector<float>>());
            agentParticleGeometryFile = config["agentParticleGeometryFile"].get<std::string>();
            agentEmitDirectionFile = config["agentEmitDirectionFile"].get<std::string>();
            emitDirection = make_cuFloat3(config["emitDirection"].get<std::vector<float>>());
            emitVel = config["emitVel"].get<float>();
            emitAcc = config["emitAcc"].get<float>();
            emitStartTime = config["emitStartTime"].get<float>();
            emitParticleMaxNum = config["emitParticleMaxNum"].get<unsigned>();
            emitGapScaleFactor = config["emitGapScaleFactor"].get<float>();
            emitOnCudaDevice = config["emitOnCudaDevice"].get<bool>();
            config["genType"] = 0;
            config["particleGeometryPath"] = agentParticleGeometryFile;
            auto agent_pos = ModelHandler::generateObjectElements(config);
            if (!agentEmitDirectionFile.empty())
                emitDirection = ModelHandler::loadEmitterAgentNormal(agentEmitDirectionFile);
            if (length(emitDirection) <= 0) {
                LOG_INFO("No emit direction provided, using default direction: {0, -1, 0}.");
                emitDirection = {0, -1, 0};
            }

            if (emitVel <= 0) {
                LOG_INFO("No emit velocity provided, using default velocity: 1.");
                emitVel = 1;
            }

            if (emitGapScaleFactor < 1) {
                LOG_INFO("Emit gap scale factor should be greater than 1, using default value: 1.");
                emitGapScaleFactor = 1;
            }

            emitGap = (particleRadius * 2.f * emitGapScaleFactor) / emitVel;
            templateParticleNum = agent_pos.size();
            std::vector<int> epm(templateParticleNum, epmMaterial);
            if (emitOnCudaDevice) {
                cudaMalloc((void **) &templatePos, templateParticleNum * sizeof(float3));
                cudaMemcpy(templatePos, agent_pos.data(), templateParticleNum * sizeof(float3), cudaMemcpyHostToDevice);

                cudaMalloc((void **) &templateEPM, templateParticleNum * sizeof(int));
                cudaMemcpy(templateEPM, epm.data(), templateParticleNum * sizeof(int), cudaMemcpyHostToDevice);
            } else {
                templatePos = new float3[templateParticleNum];
                memcpy_s(templatePos, templateParticleNum * sizeof(float3), agent_pos.data(),
                         templateParticleNum * sizeof(float3));

                templateEPM = new int[templateParticleNum];
                memcpy_s(templateEPM, templateParticleNum * sizeof(int), epm.data(),
                         templateParticleNum * sizeof(int));
            }

            pos = std::vector<float3>(emitParticleMaxNum, {0, 0, 0});

            LOG_INFO("ParticleEmitterComponent config updated.");
        }
    }

    void ParticleEmitterComponent::dynamicUpdate(float simTime) {
        if ((simTime - emitStartTime) < emitGap * emitTimes ||
            emittedParticleNum + templateParticleNum > emitParticleMaxNum)
            return;

        if (emitOnCudaDevice) {
            if (!attachedPosBuffers.empty()) {
                for (auto buffer: attachedPosBuffers) {
                    cudaMemcpy(buffer + bufferInsertOffset,
                               templatePos, templateParticleNum * sizeof(float3),
                               cudaMemcpyDeviceToDevice);
                }
            }

            if (!attachedEPMBuffers.empty()) {
                for (auto buffer: attachedEPMBuffers) {
                    cudaMemcpy(buffer + bufferInsertOffset,
                               templateEPM, templateParticleNum * sizeof(int),
                               cudaMemcpyDeviceToDevice);
                }
            }
        } else {
            if (!attachedPosBuffers.empty()) {
                for (auto buffer: attachedPosBuffers)
                    memcpy_s(buffer + bufferInsertOffset, templateParticleNum * sizeof(float3),
                             templatePos, templateParticleNum * sizeof(float3));
            }

            if (!attachedEPMBuffers.empty()) {
                for (auto buffer: attachedEPMBuffers)
                    memcpy_s(buffer + bufferInsertOffset, templateParticleNum * sizeof(int),
                             templateEPM, templateParticleNum * sizeof(int));
            }
        }

        emitTimes++;
        emittedParticleNum += templateParticleNum;
        bufferInsertOffset += templateParticleNum;
    }

    void ParticleEmitterComponent::destroy() {
        if (emitOnCudaDevice) {
            if (templatePos)
                cudaFree(templatePos);
            if (templateEPM)
                cudaFree(templateEPM);
        } else {
            if (templatePos)
                delete[] templatePos;
            if (templateEPM)
                delete[] templateEPM;
        }
    }

    std::vector<float> ParticleEmitterComponent::getElements() {
        auto pos_fptr = reinterpret_cast<float *>(pos.data());
        auto size = pos.size();
        auto ret = std::vector<float>(pos_fptr, pos_fptr + 3 * size);
        return std::move(ret);
    }

    bool ParticleEmitterComponent::configIsComplete(json &config) {
        bool ret = true;
        if (!config.contains("particleRadius")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: particleRadius");
            ret = false;
        }
        if (!config.contains("epmMaterial")) LOG_WARNING("ParticleEmitterComponent config missing key: epmMaterial");
        if (!config.contains("agentParticleGeometryFile")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: agentParticleGeometryFile");
            ret = false;
        }
        if (!config.contains("agentEmitDirectionFile")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: agentEmitDirectionFile");
            ret = false;
        }
        if (!config.contains("emitDirection")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: emitDirection");
            ret = false;
        }
        if (!config.contains("emitVel")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: emitVel");
            ret = false;
        }
        if (!config.contains("emitAcc")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: emitAcc");
            ret = false;
        }
        if (!config.contains("emitStartTime")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: emitStartTime");
            ret = false;
        }
        if (!config.contains("emitParticleMaxNum")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: emitParticleMaxNum");
            ret = false;
        }
        if (!config.contains("emitOnCudaDevice")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: emitOnCudaDevice");
            ret = false;
        }
        if (!config.contains("emitGapScaleFactor")) {
            LOG_ERROR("ParticleEmitterComponent config missing key: emitGapScaleFactor");
            ret = false;
        }
        return ret;
    }
    // ====================================================================================


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