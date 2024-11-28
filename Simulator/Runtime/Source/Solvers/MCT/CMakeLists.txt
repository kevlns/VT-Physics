PROJECT(VPSolver-immct)

FIND_PACKAGE(CUDAToolkit)
FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)

FILE(GLOB_RECURSE MCT_FILES
        ${VP_RUNTIME_ROOT}/Include/Solvers/MCT/*.hpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/MCT/*.cpp
        ${VP_RUNTIME_ROOT}/Source/Solvers/MCT/*.cuh
        ${VP_RUNTIME_ROOT}/Source/Solvers/MCT/*.cu
)

ADD_LIBRARY(vpmct STATIC ${MCT_FILES})
TARGET_LINK_LIBRARIES(vpmct PRIVATE vpcore vpugns vpda)
TARGET_INCLUDE_DIRECTORIES(vpmct PRIVATE ${VP_LIB_ROOT}/Include ${NLOHMANN_JSON_INCLUDE_DIRS})
ADD_DEPENDENCIES(vpimmct vpcore vpugns)
ADD_COMPILE_OPTIONS("/utf-8")