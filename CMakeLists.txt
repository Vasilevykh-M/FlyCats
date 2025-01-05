cmake_minimum_required(VERSION 3.10)
project(OpenGLExample)

set(CMAKE_CXX_STANDARD 11)

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# Добавляем stb_image
include_directories(/usr/include/stb)
find_path(ASSIMP_INCLUDE_DIRS NAMES assimp/scene.h PATHS /usr/include)
find_library(ASSIMP_LIBRARIES NAMES assimp PATHS /usr/lib/x86_64-linux-gnu)
include_directories(${ASSIMP_INCLUDE_DIRS})


add_executable(OpenGLExample main.cpp)

target_link_libraries(OpenGLExample glfw GLEW assimp ${OPENGL_LIBRARIES} ${ASSIMP_LIBRARIES})
