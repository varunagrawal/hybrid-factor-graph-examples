cmake_minimum_required(VERSION 3.12)
enable_testing()
project(
  hfgex # Hybrid Factor Graph EXamples
  DESCRIPTION "Examples of Hybrid Factor Graphs"
  VERSION 1.0.0
  LANGUAGES CXX C)

add_compile_options(-faligned-new)

# Enforce c++20 standards
add_compile_options(-std=c++20) # CMake 3.1 and earlier
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_AUTHOR "Varun Agrawal")
set(PROJECT_AUTHOR_EMAIL "varunagrawal@gatech.edu")

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}"
                      "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find GTSAM components
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIR})

find_package(GTSAMCMakeTools)
include(GtsamBuildTypes)
include(GtsamTesting)
include(GtsamPrinting)

# Allow access to GTSAM 3rdparty libs
set(GTSAM_LIBS gtsam)

option(HFGEX_WITH_TBB "Use Intel Threaded Building Blocks (TBB) if available"
       OFF)
option(HFGEX_ENABLE_GPERFTOOLS "Enable profiling with GPerfTools" OFF)

if(HFGEX_WITH_TBB)
  find_package(TBB 4.4 COMPONENTS tbb tbbmalloc)
  if(TBB_FOUND)
    list(APPEND TBB_LIBRARIES TBB::tbb TBB::tbbmalloc)
  endif()
endif(HFGEX_WITH_TBB)

print_config("GTSAM Version" "${GTSAM_VERSION}")
print_config("yaml-cpp Version" "${yaml-cpp_VERSION}")
print_config("Use Intel TBB" "${TBB_VERSION}")
print_config("Build Type" "${CMAKE_BUILD_TYPE}")
print_config("GPerfTools Enabled" "${HFGEX_ENABLE_GPERFTOOLS}")

# For unit tests and scripts.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

print_build_options_for_target(${PROJECT_NAME})

# Add source code to build
add_subdirectory(src)
