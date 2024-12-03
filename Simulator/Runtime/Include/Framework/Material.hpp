/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_MATERIAL_HPP
#define VT_PHYSICS_MATERIAL_HPP

#include <cstdint>
#include <unordered_map>

namespace VT_Physics {

    typedef enum eParticleMaterial : int {
        epm_Fluid = 0,
        epm_Boundary = 1,
        epm_Porous = 2,

        epm_ProgramControlledRigidFan = 10,

        epm_Null = INT_MAX
    } ParticleMaterial;

#define EPM_FLUID static_cast<int>(ParticleMaterial::epm_Fluid)
#define EPM_BOUNDARY static_cast<int>(ParticleMaterial::epm_Boundary)
#define EPM_POROUS static_cast<int>(ParticleMaterial::epm_Porous)
#define EPM_PCR_FAN static_cast<int>(ParticleMaterial::epm_ProgramControlledRigidFan)

    inline const std::unordered_map<int, std::string> EPMString = {
            {EPM_FLUID,    "Fluid"},
            {EPM_BOUNDARY, "Boundary"},
            {EPM_POROUS,   "Porous"},
            {EPM_PCR_FAN,  "PCR_Fan"},
    };
}

#endif //VT_PHYSICS_MATERIAL_HPP
