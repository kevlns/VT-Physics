PROJECT(VP-Manager)

FILE(GLOB_RECURSE RUNTIME_MANAGER_HEADER_FILES
        ${VP_RUNTIME_ROOT}/Include/Manager/*.hpp
)
FILE(GLOB_RECURSE RUNTIME_MANAGER_SRC_FILES
        *.cpp
)

ADD_LIBRARY(vpmanager STATIC ${RUNTIME_MANAGER_HEADER_FILES} ${RUNTIME_MANAGER_SRC_FILES})
TARGET_LINK_LIBRARIES(vpmanager PUBLIC
        vpcore
        vppbf
        vpdfsph
        vpimm
)
ADD_DEPENDENCIES(vpmanager vppbf vpdfsph vpimm)
