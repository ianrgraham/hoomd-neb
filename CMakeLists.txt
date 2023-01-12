if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

# Name the plugin project
project(plugins LANGUAGES C CXX)

# Find the installed HOOMD
find_package(HOOMD 3.0.0 REQUIRED)

message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# Force installation to the HOOMD installation location
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)
endif()

set(COMPONENT_NAME neb_plugin)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(_${COMPONENT_NAME}_sources
    module.cc
    NEBEnergyMinimizer.cc
    NEBEnergyMinimizerGPU.cc
    )

set(_${COMPONENT_NAME}_cu_sources
    NEBEnergyMinimizerGPU.cu
    )

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
endif (ENABLE_HIP)

pybind11_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_cuda_sources} NO_EXTRAS)
# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME}
        PUBLIC HOOMD::_hoomd
        PUBLIC HOOMD::_md
        )

fix_cudart_rpath(_${COMPONENT_NAME})

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files
    __init__.py
    neb.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

copy_files_to_build("${files}" "${COMPONENT_NAME}" "*.py")

if (BUILD_TESTING)
    # add_subdirectory(test-py)
endif()
