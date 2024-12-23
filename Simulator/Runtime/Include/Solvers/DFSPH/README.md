## Divergence-Free Smoothed Particle Hydrodynamics

This is an implementation of the DFSPH method. The code is based on the paper
"Divergence-Free Smoothed Particle Hydrodynamics" by Jan Bender etc.  
**Paper:** <u>2015.SCA."Divergence-Free Smoothed Particle Hydrodynamics", Jan Bender etc.</u>  
**WebLink:** <u>https://dl.acm.org/doi/abs/10.1145/2786784.2786796</u>

### Solver Usage:

The Solver includes two main components:

1. **Solver Config and Export Config:** The configuration json file that holds the solver-level defined parameters.

```json
{
  "DFSPH"   : {
    "Required": {
      // Animation time in seconds
      "animationTime"            : 10,
      // Time step in seconds
      "timeStep"                 : 0.01,
      // Particle radius
      "particleRadius"           : 0.075,
      // Simulation space lower bound axis: [x,y,z]
      "simSpaceLB"               : [-1, -1, -1],
      // Simulation space size axis: [x,y,z]
      "simSpaceSize"             : [2, 2, 2],
      // Maximum number of neighbors
      "maxNeighborNum"           : 60,
      // Fluid particle rest density
      "fPartRestDensity"         : 1000,
      // Boundary particle rest density
      "bPartRestDensity"         : 1000,
      // Fluid rest viscosity
      "fPartRestViscosity"       : 0.01,
      // Divergence free threshold
      "divFreeThreshold"         : 1e-4,
      // Incompressibility threshold
      "incompThreshold"          : 1e-4,
      // Surface Tension Coefficient
      "surfaceTensionCoefficient": 0.001
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
  "solverType"   : 1,
  // Object export flag, when true, the object will be exported
  "exportFlag"   : false,
  // Object start velocity
  "velocityStart": [0, 0, 0],
  // Object start color
  "colorStart"   : [0, 1, 0]
}

```

When you use these configs, pls copy template files from `VT-Physics/ConfigTemplates/DFSPHSolver` to your target
directory and modify the parameters.

Related examples can be found in the `VT-Physics/Examples/DFSPH` directory.