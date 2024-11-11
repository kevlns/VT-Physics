## Uniform Grid Neighbor Search (UGNS)

This is an implementation of the Uniform Grid Neighbor Search method using CUDA.
**WebLink:** <u>https://developer.download.nvidia.cn/assets/cuda/files/particles.pdf</u>

### Module Usage:

The UGNS includes a main components:

1. **UGNS Config:** The configuration json file that holds the UGNS setup parameters.
```json
{
  // Simulation space lower bound axis: [x,y,z]
  "simSpaceLB"       : [-1, -1, -1],
  // Simulation space size axis: [x,y,z]
  "simSpaceSize"     : [2, 2, 2],
  // Total particle number
  "totalParticleNum" : 0,
  // Maximum number of neighbors
  "maxNeighborNum"   : 60,
  // Cuda Kernel Block Number
  "cuKernelBlockNum" : 1,
  // Cuda Kernel Thread Number
  "cuKernelThreadNum": 1,
  // Grid cell size, commonly set to the SPH support radius
  "gridCellSize"     : 0.1
}

```
When you use this config, pls refer to the file from `VT-Physics/ConfigTemplates/UGNS`.
