cmake_minimum_required(VERSION 2.8)
project(GASGD)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -pthread -DNDEBUG")

set(struct
    struct/Rating.h
    struct/MPIStructs.h
    struct/Object.h)

set(util
    util/Base.h
    util/FileUtil.h
    util/Conf.h
    util/RandomUtil.h
    util/Monitor.h)

set(threadpool
    boost/threadpool.hpp)

# find boost
find_package(Boost COMPONENTS thread program_options system filesystem)

if(NOT ${Boost_FOUND})
    SET(BOOST_ROOT ~/local) # default
    SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
    find_package(Boost COMPONENTS thread program_options system filesystem)
endif(NOT ${Boost_FOUND})

if(Boost_FOUND)
    message(STATUS "Boost found")
    include_directories( ${Boost_INCLUDE_DIR})
    link_libraries(${Boost_LIBRARIES})
endif(Boost_FOUND)

# find mpi
find_package(MPI REQUIRED)

if (MPI_FOUND)
    message(STATUS "MPI found")
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})
    link_libraries(${MPI_LIBRARIES})
endif()

# find TBB
# Add FindTBB directory to CMake's module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake/FindTBB/")

find_package(TBB REQUIRED)
if(TBB_FOUND)
    message(STATUS "TBB found: " ${TBB_INCLUDE_DIRS})
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ltbb -ltbbmalloc")
    include_directories(${TBB_INCLUDE_DIRS})
    link_libraries(${TBB_LIBRARIES})
endif(TBB_FOUND)

add_library(lib SHARED ${util} ${struct} ${threadpool})
set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)

add_executable(runGASGD GASGD.cpp Partitioner.h Data.h ASGD.h)
target_link_libraries(runGASGD lib)
target_link_libraries(runGASGD ${Boost_LIBRARIES})
target_link_libraries(runGASGD ${MPI_LIBRARIES})
target_link_libraries(runGASGD ${TBB_LIBRARIES})
