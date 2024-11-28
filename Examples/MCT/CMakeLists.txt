INCLUDE_DIRECTORIES(${VP_RUNTIME_ROOT}/Include ${VP_LIB_ROOT}/Include)

ADD_EXECUTABLE(Example.MCT.usage-1 Usage-1.cpp)
TARGET_LINK_LIBRARIES(Example.MCT.usage-1 PUBLIC vpmanager)

ADD_EXECUTABLE(Example.MCT.usage-2 Usage-2.cpp)
TARGET_LINK_LIBRARIES(Example.MCT.usage-2 PUBLIC vpmanager)