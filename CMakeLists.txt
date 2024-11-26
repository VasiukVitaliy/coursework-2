project(Project1)


set(CMAKE_TOOLCHAIN_FILE "C:/Users/QSUS/Desktop/install/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

find_package(xlnt CONFIG REQUIRED)

find_package(Matplot REQUIRED)
add_executable(Project1 main.cpp)


target_link_libraries(Project1 PRIVATE xlnt::xlnt matplot)

