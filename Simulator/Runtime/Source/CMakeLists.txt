FIND_PACKAGE(CUDAToolkit)

INCLUDE_DIRECTORIES(
        ${VP_LIB_ROOT}/Include
        ${VP_CONFIG_ROOT}
        ${CUDAToolkit_INCLUDE_DIRS}
)

#===================================== Build Components.
ADD_SUBDIRECTORY(Framework)
ADD_SUBDIRECTORY(Solvers)
ADD_SUBDIRECTORY(Manager)
