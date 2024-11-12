PROJECT(VPSolver-imm)

FIND_PACKAGE(CUDAToolkit)
FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)

FILE(GLOB_RECURSE IMM_FILES
        ${VP_RUNTIME_ROOT}/Include/Solvers/IMM/*.hpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/IMM/*.cpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/IMM/*.cuh
        ${VP_RUNTIME_ROOT}/Source/Solvers/IMM/*.cu
)

ADD_LIBRARY(vpimm STATIC ${IMM_FILES})
TARGET_LINK_LIBRARIES(vpimm PRIVATE vpcore vpugns vpda)
TARGET_INCLUDE_DIRECTORIES(vpimm PRIVATE ${VP_LIB_ROOT}/Include ${NLOHMANN_JSON_INCLUDE_DIRS})
ADD_DEPENDENCIES(vpimm vpcore vpugns)
ADD_COMPILE_OPTIONS("/utf-8")