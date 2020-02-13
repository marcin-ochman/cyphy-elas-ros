cmake_minimum_required(VERSION 2.8.3)
project(elas_ros)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  image_geometry
  image_transport
  message_filters
  message_generation
  nav_msgs
  nodelet
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  stereo_msgs
  tf
  libelas
)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS signals)

add_message_files( DIRECTORY msg FILES ElasFrameData.msg)

add_definitions(-msse3)

include_directories(src ${libelas_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

generate_dynamic_reconfigure_options(cfg/ElasParameters.cfg)
add_library(elas_ros SHARED src/elas.cpp)

add_dependencies(elas_ros ${catkin_EXPORTED_TARGETS})
add_dependencies(elas_ros ${PROJECT_NAME}_gencfg)

target_link_libraries(elas_ros ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES})

install(TARGETS elas_ros
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

install(FILES launch/elas.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
