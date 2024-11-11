PROJECT(VPSolver-dfsph)

FIND_PACKAGE(CUDAToolkit)
FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)

FILE(GLOB_RECURSE DFSPH_FILES
        ${VP_RUNTIME_ROOT}/Include/Solvers/DFSPH/*.hpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/DFSPH/*.cpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/DFSPH/*.cuh
        ${VP_RUNTIME_ROOT}/Source/Solvers/DFSPH/*.cu
)

ADD_LIBRARY(vpdfsph STATIC ${DFSPH_FILES})
TARGET_LINK_LIBRARIES(vpdfsph PRIVATE vpcore vpugns vpda)
TARGET_INCLUDE_DIRECTORIES(vpdfsph PRIVATE ${VP_LIB_ROOT}/Include ${NLOHMANN_JSON_INCLUDE_DIRS})
ADD_DEPENDENCIES(vpdfsph vpcore vpugns)
ADD_COMPILE_OPTIONS("/utf-8")