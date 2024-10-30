/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_MATERIAL_HPP
#define VT_PHYSICS_MATERIAL_HPP

namespace VT_Physics {

    typedef enum eParticleMaterial : uint8_t {
        epm_Fluid,
        epm_Boundary,

        epm_Null = UINT8_MAX
    } ParticleMaterial;

#define EPM_FLUID static_cast<uint8_t>(ParticleMaterial::epm_Fluid)
#define EPM_BOUNDARY static_cast<uint8_t>(ParticleMaterial::epm_Boundary)

}

#endif //VT_PHYSICS_MATERIAL_HPP
