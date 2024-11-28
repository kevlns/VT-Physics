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

        epm_Null = INT_MAX
    } ParticleMaterial;

#define EPM_FLUID static_cast<int>(ParticleMaterial::epm_Fluid)
#define EPM_BOUNDARY static_cast<int>(ParticleMaterial::epm_Boundary)
#define EPM_POROUS static_cast<int>(ParticleMaterial::epm_Porous)

    inline const std::unordered_map<int, std::string> EPMString = {
            {EPM_FLUID,    "Fluid"},
            {EPM_BOUNDARY, "Boundary"},
            {EPM_POROUS,   "Porous"}
    };
}

#endif //VT_PHYSICS_MATERIAL_HPP
