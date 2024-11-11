PROJECT(VP-Runtime)

FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)
FIND_PACKAGE(CUDAToolkit)

INCLUDE_DIRECTORIES(
        Include
        ${VP_SOURCE_ROOT}
        ${VP_LIB_ROOT}/Include
        ${NLOHMANN_JSON_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

#===================================== Build Components.
ADD_SUBDIRECTORY(Source)