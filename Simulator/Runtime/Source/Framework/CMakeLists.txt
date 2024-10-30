PROJECT(VP-Core)

FIND_PACKAGE(spdlog CONFIG REQUIRED)
FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)
FIND_PACKAGE(CUDAToolkit)

FILE(GLOB_RECURSE RUNTIME_CORE_FILES
        ${VP_RUNTIME_ROOT}/Include/Core/*.hpp
        ${VP_RUNTIME_ROOT}/Include/Framework/*.hpp
        ${VP_RUNTIME_ROOT}/Source/Framework/*.cpp
)

ADD_LIBRARY(vpcore STATIC ${RUNTIME_CORE_FILES})
TARGET_LINK_LIBRARIES(vpcore PUBLIC vpjson vpmodel vplogger CUDA::cudart)
TARGET_INCLUDE_DIRECTORIES(vpcore PUBLIC
        ${VP_LIB_ROOT}/Include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${NLOHMANN_JSON_INCLUDE_DIRS})
ADD_DEPENDENCIES(vpcore vpjson vplogger)
