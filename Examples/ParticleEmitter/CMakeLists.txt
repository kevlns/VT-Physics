INCLUDE_DIRECTORIES(${VP_RUNTIME_ROOT}/Include ${VP_LIB_ROOT}/Include)

ADD_EXECUTABLE(Example.ParticleEmitter.usage-1 Usage-1.cpp)
TARGET_LINK_LIBRARIES(Example.ParticleEmitter.usage-1 PUBLIC vpmanager)