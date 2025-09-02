cmake_minimum_required(VERSION 3.10)
project(MNISTClassifier)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找LibTorch包
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(mnist_classifier src/main.cpp)

# 链接LibTorch和OpenCV库
target_link_libraries(mnist_classifier "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")

# 设置LibTorch库的运行时路径
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET mnist_classifier
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:mnist_classifier>)
endif (MSVC)