cmake_minimum_required(VERSION 3.8)
project(local_learning)

find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(global_to_polar_cpp REQUIRED)
find_package(f1tenth_planning_custom_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(ament_index_cpp REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; import pathlib; print(pathlib.Path(torch.utils.cmake_prefix_path))"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

add_executable(local_learning_node src/local_learning_node.cpp)
target_compile_features(local_learning_node PRIVATE cxx_std_17)
target_link_libraries(local_learning_node "${TORCH_LIBRARIES}")
target_include_directories(local_learning_node PRIVATE ${TORCH_INCLUDE_DIRS})
ament_target_dependencies(local_learning_node
  rclcpp
  sensor_msgs
  global_to_polar_cpp
  f1tenth_planning_custom_msgs
  message_filters
  ament_index_cpp
)

install(TARGETS local_learning_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
