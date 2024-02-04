# CMakeLists.txt in the root of your project

cmake_minimum_required(VERSION 3.12)
project(YourProjectName VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add your own library
add_library(libGANN
    src/GANN.cpp
    include/GANN.hpp
)

target_include_directories(libGANN PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add dependencies
add_subdirectory(dependencies/generic-ga)
add_subdirectory(dependencies/Neural-Network)

# Add executable for testing
add_executable(testExe
    test/test.cpp
)

# Add compiler flags
target_compile_options(testExe PRIVATE -O2)
# Link your library and dependencies
target_link_libraries(testExe PRIVATE libGANN libGA libNeuralNet)

# Set the output directory for the test executable
set_target_properties(testExe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

# Copy necessary files for testing to the binary directory during build
configure_file(test/heart_disease.csv ${CMAKE_CURRENT_BINARY_DIR}/bin/heart_disease.csv COPYONLY)
configure_file(test/iris.csv ${CMAKE_CURRENT_BINARY_DIR}/bin/iris.csv COPYONLY)
configure_file(test/wine.csv ${CMAKE_CURRENT_BINARY_DIR}/bin/wine.csv COPYONLY)

# Optionally, you can add a custom target for running the tests
add_custom_target(run_tests
    COMMAND ./bin/testExe
    DEPENDS testExe
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Optionally, you can make 'make test' run the tests
enable_testing()
add_test(NAME MyTest COMMAND ./bin/testExe)

