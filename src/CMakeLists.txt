cmake_minimum_required(VERSION 3.12)
enable_testing()

# Add include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Get all script files
file(GLOB script_srcs "*.cpp")
file(GLOB script_headers "*.h")

# Variable to record libraries to link against
set(PROJECT_LIBS "gtsam;${Boost_PROGRAM_OPTIONS_LIBRARY}")

if(HFGEX_ENABLE_GPERFTOOLS)
  find_library(GPERFTOOLS_TCMALLOC NAMES tcmalloc)
  find_library(GPERFTOOLS_PROFILER NAMES profiler)
  message(STATUS "Linking ${GPERFTOOLS_TCMALLOC} & ${GPERFTOOLS_PROFILER}")

  set(PROJECT_LIBS
      "${PROJECT_LIBS};${GPERFTOOLS_TCMALLOC};${GPERFTOOLS_PROFILER}")
endif()

# Create executables
foreach(script_src IN ITEMS ${script_srcs})
  # Get script base name
  get_filename_component(script_name ${script_src} NAME_WE)

  # Add executable
  add_executable(${script_name} ${script_src} ${script_headers})

  target_link_libraries(${script_name} ${PROJECT_LIBS})

  # Apply user build flags from CMake cache variables:
  gtsam_apply_build_flags(${script_name})

  # Add target dependencies
  add_dependencies("examples" ${script_name})
  if(NOT MSVC AND NOT XCODE_VERSION)
    add_custom_target(
      ${script_name}.run
      ${EXECUTABLE_OUTPUT_PATH}${script_name}
      DEPENDS ${script_name})
  endif()

  # Add TOPSRCDIR
  set_property(
    SOURCE ${script_src}
    APPEND
    PROPERTY COMPILE_DEFINITIONS "TOPSRCDIR=\"${GTSAM_SOURCE_DIR}\"")

  # Configure target folder (for MSVC and Xcode)
  set_property(TARGET ${script_name} PROPERTY FOLDER "examples")
endforeach()
