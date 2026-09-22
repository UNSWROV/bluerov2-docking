cmake_minimum_required(VERSION 3.8)
project(perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  perception/led_mock_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME led_mock_publisher
)

install(PROGRAMS
  perception/aruco/aruco_relay.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME aruco_relay
)

install(PROGRAMS
  perception/aruco/aruco_fusion.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME aruco_fusion
)

install(PROGRAMS
  perception/aruco/dock_pose_filter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dock_pose_filter
)

install(PROGRAMS
  perception/aruco/dock_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dock_visualizer
)

install(PROGRAMS
  perception/nav_error_injector.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME nav_error_injector
)

install(PROGRAMS
  perception/oracle_dock_velocity.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME oracle_dock_velocity
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_transforms perception/utils/tests/test_transforms.py)
  ament_add_pytest_test(test_nav_error perception/utils/tests/test_nav_error.py)
  ament_add_pytest_test(test_differentiate perception/utils/tests/test_differentiate.py)
  ament_add_pytest_test(test_geometry perception/aruco/tests/test_geometry.py)
  ament_add_pytest_test(test_dock_layout perception/aruco/tests/test_dock_layout.py)
  ament_add_pytest_test(test_noise_model perception/aruco/tests/test_noise_model.py)
  ament_add_pytest_test(test_spatial_consensus perception/aruco/tests/test_spatial_consensus.py)
  ament_add_pytest_test(test_pose_fusion perception/aruco/tests/test_pose_fusion.py)
  ament_add_pytest_test(test_kalman perception/aruco/tests/test_kalman.py)
  ament_add_pytest_test(test_health perception/aruco/tests/test_health.py)
  ament_add_pytest_test(test_aruco_fusion_node
    perception/aruco/integration/test_aruco_fusion_node.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_dock_pose_filter_node
    perception/aruco/integration/test_dock_pose_filter_node.py
    TIMEOUT 60
  )
endif()

ament_package()
