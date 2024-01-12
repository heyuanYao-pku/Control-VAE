PROJECT(ModifyODE)

cmake_minimum_required(VERSION 3.15)

add_compile_definitions(dDOUBLE)
add_compile_definitions(CCD_DOUBLE)
add_compile_definitions(ODE_LIB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(_USE_MATH_DEFINES)

if (WIN32)
add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dir})
set(CMAKE_BUILD_FILES_DIRECTORY ${dir})
set(CMAKE_BUILD_DIRECTORY ${dir})
set(CMAKE_BINARY_DIR  ${dir})
SET(EXECUTABLE_OUTPUT_PATH ${dir})
SET(LIBRARY_OUTPUT_PATH ${dir})
SET(CMAKE_CACHEFILE_DIR ${dir})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/eigen-3.3.8/Eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/eigen-3.3.8)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/glew32)
include_directories("include")
include_directories("include/ode")
include_directories("include/drawstuff")
include_directories("src")
include_directories("src/joints")
include_directories("OPCODE")

include_directories("Utils")
# include_directories("libccd/src")

aux_source_directory("src" DIR_SRCS)
aux_source_directory("src/joints" DIR_SRC_JOINTS)
aux_source_directory("OPCODE" DIR_OPCODE)
aux_source_directory("OPCODE/Ice" DIR_OPCODE_ICE)
aux_source_directory("Utils" DIR_UTILS)
aux_source_directory("Utils/test" DIR_UTILS_TEST)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../ThirdParty/EigenExtension EigenExtSrcs)

add_library(${PROJECT_NAME} STATIC ${DIR_SRCS} ${DIR_SRC_JOINTS} ${DIR_OPCODE} ${DIR_OPCODE_ICE})

if (WIN32)
set(
	DRAWSTUFF_SRCS
	include/drawstuff/drawstuff.h
	include/drawstuff/drawstuffWrapper.h
	include/drawstuff/version.h
	drawstuff/src/drawstuff.cpp
	drawstuff/src/drawstuffWrapper.cpp
	drawstuff/src/internal.h
)
set(DRAWSTUFF_RSRC drawstuff/src/resources.rc)
list(
	APPEND DRAWSTUFF_SRCS
	drawstuff/src/resource.h
	drawstuff/src/windows.cpp
	${DRAWSTUFF_RSRC}
)

find_package(OpenGL REQUIRED)
message("OpenGL include dir=${OPENGL_INCLUDE_DIR}")
message("OpenGL lib=${OPENGL_LIBRARIES}")

add_library(DrawStuff STATIC ${DRAWSTUFF_SRCS})
target_compile_definitions(DrawStuff PRIVATE -DDS_LIB)
# add_definitions(-DDEFAULT_PATH_TO_TEXTURES="${CMAKE_CURRENT_SOURCE_DIR}/drawstuff/textures")
target_link_libraries(DrawStuff PUBLIC ${PROJECT_NAME})
target_include_directories(DrawStuff PRIVATE ${OPENGL_INCLUDE_DIRS})
target_link_libraries(DrawStuff PRIVATE ${OPENGL_LIBRARIES})
target_link_libraries(DrawStuff PRIVATE glew32)
target_link_libraries(DrawStuff PRIVATE winmm)
else()
set(
	DRAWSTUFF_SRCS
	include/drawstuff/drawstuffWrapper.h
	drawstuff/src/drawstuffWrapper.cpp
)
add_library(DrawStuff STATIC ${DRAWSTUFF_SRCS})
endif()

ADD_LIBRARY(MotionUtils STATIC ${DIR_UTILS} ${EigenExtSrcs} ${DIR_UTILS_TEST})
