PROJECT(VPSolver-immct)

FIND_PACKAGE(CUDAToolkit)
FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)

FILE(GLOB_RECURSE IMMCT_FILES
        ${VP_RUNTIME_ROOT}/Include/Solvers/IMM-CT/*.hpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/IMM-CT/*.cpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/IMM-CT/*.cuh
        ${VP_RUNTIME_ROOT}/Source/Solvers/IMM-CT/*.cu
)


ADD_LIBRARY(vpimmct STATIC ${IMMCT_FILES})
TARGET_LINK_LIBRARIES(vpimmct PRIVATE vpcore vpugns vpda)
TARGET_INCLUDE_DIRECTORIES(vpimmct PRIVATE ${VP_LIB_ROOT}/Include ${NLOHMANN_JSON_INCLUDE_DIRS})
ADD_DEPENDENCIES(vpimmct vpcore vpugns)
ADD_COMPILE_OPTIONS("/utf-8")