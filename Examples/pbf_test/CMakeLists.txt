INCLUDE_DIRECTORIES(${VP_RUNTIME_ROOT}/Include ${VP_LIB_ROOT}/Include)

ADD_EXECUTABLE(pbfTest_main pbf_main.cpp)
TARGET_LINK_LIBRARIES(pbfTest_main PRIVATE vpmanager)