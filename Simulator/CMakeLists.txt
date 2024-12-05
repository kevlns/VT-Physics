
CONFIGURE_FILE(Config/RealPath.hpp.in ${VP_CONFIG_ROOT}/RealPath.h)

#===================================== Build Components.
ADD_SUBDIRECTORY(Library)

ADD_SUBDIRECTORY(Runtime)

ADD_SUBDIRECTORY(CommandLineTool)