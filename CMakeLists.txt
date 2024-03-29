cmake_minimum_required(VERSION 2.8.3)
project(la3dm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -o3")

find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("OpenMP enabled")
endif()

## Enable OpenMP
add_definitions(-DOPENMP)

## Enable Debug Info
#add_definitions(-DDEBUG)

#add_definitions(-DPREDICT)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        visualization_msgs
        nav_msgs
        pcl_ros
        roscpp
        tf_conversions
	cv_bridge
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
#  CATKIN_DEPENDS geometry_msgs pcl_ros roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
        include/bgkoctomap
        include/common
        include/dataset_devkit
        ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(bgkoctomap
#   src/${PROJECT_NAME}/bgkoctomap.cpp
# )

# Common source files
set(COMMON_SOURCE_FILES src/common/point3f)

# Files for BGK source
set(BGK_SOURCE_FILES
    src/bgkoctomap/bgkblock.cpp
    src/bgkoctomap/bgkoctomap.cpp
    src/bgkoctomap/bgkoctree.cpp)

#add_definitions(-DOCCUPANCY)
#add_executable(bgkoctomap_static_node
#    ${BGK_SOURCE_FILES}
#    ${COMMON_SOURCE_FILES}
#    src/bgkoctomap/bgkoctree_node.cpp
#    src/bgkoctomap/bgkoctomap_occupancy.cpp
#    src/bgkoctomap/bgkoctomap_static_node.cpp
#)

add_definitions(-DSEMANTICS)
add_executable(bgkoctomap_semantics_node
    ${BGK_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkoctomap/bgkoctree_node_semantics.cpp
    src/bgkoctomap/bgkoctomap_semantics.cpp
    src/bgkoctomap/bgkoctomap_traversability.cpp
    src/bgkoctomap/bgkoctomap_semantics_node.cpp
)

add_executable(cassie_node
    ${BGK_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkoctomap/bgkoctree_node_semantics.cpp
    src/bgkoctomap/bgkoctomap_semantics.cpp
    src/bgkoctomap/bgkoctomap_traversability.cpp
    src/bgkoctomap/cassie_node.cpp
)

add_executable(kitti_node
    ${BGK_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkoctomap/bgkoctree_node_semantics.cpp
    src/bgkoctomap/bgkoctomap_semantics.cpp
    src/bgkoctomap/bgkoctomap_traversability.cpp
    src/bgkoctomap/kitti_node.cpp
)

add_executable(tartanair_node
    ${BGK_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkoctomap/bgkoctree_node_semantics.cpp
    src/bgkoctomap/bgkoctomap_semantics.cpp
    src/bgkoctomap/bgkoctomap_traversability.cpp
    src/bgkoctomap/tartanair_node.cpp
)

add_executable(semantickitti_util
    ${COMMON_SOURCE_FILES}
    src/dataset_devkit/semantickitti_util.cpp
)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(BGKOctoMap_node BGKOctoMap_generate_messages_cpp)

## Specify libraries to link a library or executable target against
#target_link_libraries(bgkoctomap_static_node
#        ${catkin_LIBRARIES}
#)

target_link_libraries(bgkoctomap_semantics_node
        ${catkin_LIBRARIES}
)

target_link_libraries(cassie_node
        ${catkin_LIBRARIES}
)

target_link_libraries(kitti_node
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
)

target_link_libraries(tartanair_node
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
)

target_link_libraries(semantickitti_util
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS bgkoctomap BGKOctoMap_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_BGKOctoMap.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
