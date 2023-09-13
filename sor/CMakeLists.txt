cmake_minimum_required(VERSION 3.21.3)

project(Sor)

add_compile_options(-Wall -Wextra -Werror -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
add_library(${PROJECT_NAME} SHARED solver_py.c solver.c image.c)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES PREFIX "") # Disable "lib" prefix

find_package(Python COMPONENTS Development NumPy)

target_include_directories(${PROJECT_NAME} PUBLIC ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})


install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_SOURCE_DIR})
