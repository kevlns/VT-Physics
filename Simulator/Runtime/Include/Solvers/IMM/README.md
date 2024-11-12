## An Implicitly Stable Mixture Model for Dynamic Multi-fluid Simulations

This is an implementation of the IMM method. The code is based on the paper
" An Implicitly Stable Mixture Model for Dynamic Multi-fluid Simulations" by Yanrui Xu etc.  
**Paper:** <u>2023.SigAsia."An Implicitly Stable Mixture Model for Dynamic Multi-fluid Simulations", Yanrui Xu
etc.</u>  
**WebLink:** <u>https://dl.acm.org/doi/10.1145/3610548.3618215</u>

### Solver Usage:

The Solver includes two main components:

1. **Solver Config and Export Config:** The configuration json file that holds the solver-level defined parameters.

```json
{
  "PBF"   : {
    "Required": {
      // Animation time in seconds
      "animationTime"                : 10,
      // Time step in seconds
      "timeStep"                     : 0.01,
      // Particle radius
      "particleRadius"               : 0.075,
      // Simulation space lower bound axis: [x,y,z]
      "simSpaceLB"                   : [-1, -1, -1],
      // Simulation space size axis: [x,y,z]
      "simSpaceSize"                 : [2, 2, 2],
      // Maximum number of neighbors
      "maxNeighborNum"               : 60,
      // Fluid rest viscosity
      "fPartRestViscosity"           : 0.01,
      // Divergence free threshold
      "divFreeThreshold"             : 1e-4,
      // Incompressibility threshold
      "incompThreshold"              : 1e-4,
      // Surface Tension Coefficient
      "surfaceTensionCoefficient"    : 0.001,
      // Diffusion coefficient
      "diffusionCoefficientCf"       : 0.1,
      // Momentum exchange coefficient
      "momentumExchangeCoefficientCd": 0.5,
      // Rest densities of phases, list size denotes phase num
      "phaseRestDensity"             : [1000, 1000, 1000],
      // Rest viscosities of phases, list size denotes phase num
      "phaseRestViscosity"           : [0.01, 0.01, 0.01],
      // Rest colors of phases, list size denotes phase num
      "phaseRestColor"               : [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
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
  "solverType"   : 2,
  // Object export flag, when true, the object will be exported
  "exportFlag"   : false,
  // Object start velocity
  "velocityStart": [0, 0, 0],
  // Phase fraction of the object, list size denotes phase num and items sum up to 1
  "phaseFraction": [0.1, 0.2, 0.7]
}

```

When you use these configs, pls copy template files from `VT-Physics/ConfigTemplates/IMMSolver` to your target
directory and modify the parameters.

Related examples can be found in the `VT-Physics/Examples/IMM` directory.