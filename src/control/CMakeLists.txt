cmake_minimum_required(VERSION 3.8)
project(control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  control/coarse_approach_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME coarse_approach_node
)

install(PROGRAMS
  control/fine_align_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fine_align_node
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_pbvs control/tests/test_pbvs.py)
  ament_add_pytest_test(test_guidance control/tests/test_guidance.py)
  ament_add_pytest_test(test_health_gate control/tests/test_health_gate.py)
  ament_add_pytest_test(test_fine_guidance control/tests/test_fine_guidance.py)
  ament_add_pytest_test(test_velocity_loop control/tests/test_velocity_loop.py)
  ament_add_pytest_test(test_coarse_approach_node
    control/integration/test_coarse_approach_node.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_coarse_gate
    control/integration/test_coarse_gate.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_fine_align_node
    control/integration/test_fine_align_node.py
    TIMEOUT 60
  )
  ament_add_pytest_test(test_velocity_loop_node
    control/integration/test_velocity_loop_node.py
    TIMEOUT 90
  )
endif()

ament_package()
