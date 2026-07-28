cmake_minimum_required(VERSION 3.8)
project(description)

find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-math7 REQUIRED)

add_library(DockSway SHARED plugins/dock_sway/DockSway.cc)
target_include_directories(DockSway PRIVATE plugins)
target_link_libraries(DockSway gz-sim8::gz-sim8 gz-math7::gz-math7)

install(
  DIRECTORY models worlds rviz foxglove
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS DockSway LIBRARY DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(DockSwayMotionTest plugins/dock_sway/test/DockSwayMotionTest.cc)
  target_include_directories(DockSwayMotionTest PRIVATE plugins)
  target_link_libraries(DockSwayMotionTest gz-math7::gz-math7)
endif()

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/hooks.dsv.in")

ament_package()
