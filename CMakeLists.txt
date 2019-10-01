cmake_minimum_required(VERSION 3.9.2)
PROJECT(ceddl LANGUAGES C)

# SET BUILD TYPE (RELEASE/DEBUG)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# SET STATIC/DYNAMIC RUNTIME
set(CEDDL_SHARED OFF CACHE BOOL "Whether to build a static or dynamic library")

# Microsoft Visual C++ - set static/dynamic runtime
if(MSVC)
    if(CEDDL_SHARED)
        message(STATUS "Setting dinamic runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
    else(CEDDL_SHARED)
        message(STATUS "Setting static runtime for MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach(flag_var)
	endif(CEDDL_SHARED)

	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add the given directories to those the compiler uses to search for include files
include_directories(src)


# SET SOURCES *******************************
SET(SOURCE_DIR "src")
SET(C_SOURCES
        src/cddl.c
        )
SET(SOURCES ${C_SOURCES}})

# [LIBRARY] CEDDL LIBRARY *********************
if(CEDDL_SHARED)
	add_library(ceddl SHARED "")
else(CEDDL_SHARED)
	add_library(ceddl STATIC "")
endif(CEDDL_SHARED)
target_sources(ceddl PRIVATE ${C_SOURCES})
target_include_directories(ceddl PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
	$<INSTALL_INTERFACE:include>
)

# Target properties
set_target_properties(eddl PROPERTIES OUTPUT_NAME_DEBUG "ceddld")
set_target_properties(eddl PROPERTIES OUTPUT_NAME_RELEASE "ceddl")
set_target_properties(eddl PROPERTIES POSITION_INDEPENDENT_CODE ON)

# [INSTALL] Install the CEDDL library
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
endif()
install(TARGETS ceddl
		EXPORT CEDDLTargets
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
install(EXPORT CEDDLTargets DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
		DESTINATION include/ceddl
        FILES_MATCHING PATTERN "*.h")


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CEDDLConfig.cmake.in ${CMAKE_BINARY_DIR}/cmake/CEDDLConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/cmake/CEDDLConfig.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}) 
