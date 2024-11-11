PROJECT(VP-Modules)

FIND_PACKAGE(CUDAToolkit)
FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)

INCLUDE_DIRECTORIES(
        ${VP_LIB_ROOT}/Include
        ${VP_RUNTIME_ROOT}/Include
        ${VP_CONFIG_ROOT}
        ${CUDAToolkit_INCLUDE_DIRS}
)
#===================================== Build Components.
ADD_SUBDIRECTORY(NeighborSearch)