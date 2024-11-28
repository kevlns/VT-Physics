## Multiphase Viscoelastic Non-Newtonian Fluid Simulation

This is an implementation of the IMM-CT method. The code is based on the paper
"Multiphase Viscoelastic Non-Newtonian Fluid Simulation" by Yalan Zhang, Long Shen etc.  
**Paper:** <u>2024.CGF."Multiphase Viscoelastic Non-Newtonian Fluid Simulation", Yalan Zhang, Long Shen etc.</u>  
**WebLink:** <u>https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.15180</u>

### Solver Usage:

The Solver includes two main components:

1. **Solver Config and Export Config:** The configuration json file that holds the solver-level defined parameters.

```json
{
  "IMMCT" : {
    "Required": {
      // Animation time in seconds
      "animationTime"                 : 5,
      // Time step in seconds
      "timeStep"                      : 0.001,
      // Particle radius
      "particleRadius"                : 0.05,
      // Simulation space lower bound axis: [x,y,z]
      "simSpaceLB"                    : [-1, -1, -1],
      // Simulation space size axis: [x,y,z]
      "simSpaceSize"                  : [2, 2, 2],
      // Maximum number of neighbors
      "maxNeighborNum"                : 60,
      // Fluid rest viscosity
      "fPartRestViscosity"            : 0.01,
      // Divergence free threshold
      "divFreeThreshold"              : 1e-4,
      // Incompressibility threshold
      "incompThreshold"               : 1e-4,
      // Surface Tension Coefficient
      "surfaceTensionCoefficient"     : 0.001,
      // Diffusion coefficient
      "diffusionCoefficientCf"        : 0.1,
      // Momentum exchange basic coefficient
      "momentumExchangeCoefficientCd0": 0.5,
      // Viscosity of solvent, denotes the viscosity of the first phase
      "solventViscosity"              : 0.01,
      // Rest densities of phases, list size denotes phase num, only support 2 phases
      "phaseRestDensity"              : [900, 1000],
      // Rest colors of phases, list size denotes phase num, only support 2 phases
      "phaseRestColor"                : [[255, 0, 0], [0, 0, 255]],
      // Basic viscosity of the solution, Pa.s
      "solutionBasicViscosity"        : 12,
      // Maximum viscosity of the solution, Pa.s
      "solutionMaxViscosity"          : 12,
      // Relaxation time of the fluid
      "relaxationTime"                : 0.001,
      // Shear thinning basic factor, [0, 1]
      "shearThinningBasicFactor"      : 0.5,
      // rheological threshold, [0, 1]
      "rheologicalThreshold"          : 0.4
    },
    "Optional": {
      // Enable the Optional parameters, when true, the optional parameters are used to replace the default values
      "enable" : false,
      // User defined gravity
      "gravity": [0, -9.81, 0]
    }
  },
  "EXPORT": {
    "Common"        : {
      // Export target directory, without the last '/' or '\\'
      "exportTargetDir" : "path/to/output",
      // Export file prefix, etc. obj.ply
      "exportFilePrefix": "obj",
      // Export file type: PLY, ...
      "exportFileType"  : "PLY"
    },
    "SolverRequired": {
      // Enalbe the exporting, when true, the data will be exported
      "enable"           : false,
      // Export frame per second
      "exportFps"        : 35,
      // Export group policy: MERGE, SPLIT
      "exportGroupPolicy": "MERGE"
    }
  }
}

```

2. **Solver Object Component:** The object configuration json file that holds the solver required parameters.

```json
{
  // User no need to care and DO NOT modify
  "solverType"   : 3,
  // Object export flag, when true, the object will be exported
  "exportFlag"   : false,
  // Object start velocity
  "velocityStart": [0, 0, 0],
  // Phase fraction of the object, list size denotes phase num and items sum up to 1, only support 2 phases
  "phaseFraction": [0.4, 0.6]
}

```

When you use these configs, pls copy template files from `VT-Physics/ConfigTemplates/IMMCTSolver` to your target
directory and modify the parameters.

Related examples can be found in the `VT-Physics/Examples/IMMCT` directory.