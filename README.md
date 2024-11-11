# VT-Physics

## 1. Introduction

An open-source physics engine for visual effects and research.

When conducting research tasks in the field of computer graphics, most papers
do not provide source code for actual testing and performance comparison. We developed
this library based on the actual research needs and practical engineering needs, and plan to
continuously improve the framework and implement as many classic techniques as possible.

### (1) Features

VT-Physics mainly focuses on the following aspects:

1. **Fluid Dynamics**: including particle-based, grid-based and hybrid fluid simulation
2. **Rigid-body Dynamics**
3. ...

More topics will be added in the future.

### (2) Implemented Techniques
1. "Position-Based Fluids" (**[2013-TOG. PBF](http://mmacklin.com/pbf_sig_preprint.pdf)**)
2. "Divergence-Free Smoothed Particle Hydrodynamics" (**[2015-SCA. DFSPH](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)**)
3. "An Implicitly Stable Mixture Model for Dynamic Multi-fluid Simulations" (**[2023-SIGAsia. IMM](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)**)
4. "Multiphase Viscoelastic Non-Newtonian Fluid Simulation" (**[2024-CGF. IMM-CT](https://dl.acm.org/doi/abs/10.1145/2786784.2786796)**)
5. ...

### (3) Recommended Build tool-chain

1. **CMake**: 3.10 or higher
2. **C++ & CUDA**:
    1. opt-1: Visual Studio 2019 with CUDA 11.6 or higher
    2. opt-2: Visual Studio 2022 with CUDA 12.6 or higher
3. **Platforms**: Windows / Linux.

## 2. Quick Start

### (1) Build

First, clone the project to your local directory:

```shell
git clone https://github.com/kevlns/VT-Physics.git
```

Then, run the bootstrap script to init the repo and install the dependencies:

```shell
cd VT-Physics
./bootstrap-vtphysics.bat   # for Windows
./bootstrap-vtphysics.sh    # for Linux
```

Finally, build the project with CMake:

```shell
cd VT-Physics
cmake -B build -S . -G "Visual Studio 16 2019"  # for CUDA 11.6+
cmake -B build -S . -G "Visual Studio 17 2022"  # for CUDA 12.6+
cmake --build build --config Debug
```

### (2) Run

We provide simple examples associated with each physical solver. You can run the examples with a little modification to
test the solver. The examples are located in the `VT-Physics/Examples` directory.

For each solver, we provide a `ReadME.md` file to introduce the solver and its usage, which you can find in the "VT-Physics/Simulator/Runtime/Include/Solvers" directory.