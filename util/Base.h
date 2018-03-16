#ifndef BASE_H
#define BASE_H

#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>

////////////////// Intel Threading Building Blocks //////////////////////
#include <tbb/tbb.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

using namespace tbb;

////////////////// Intel Threading Building Blocks //////////////////////


////////////////// Boost //////////////////////

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include "../boost/threadpool.hpp"

using namespace boost::threadpool;
namespace po = boost::program_options;

////////////////// Boost //////////////////////

using std::cout;
using std::cerr;
using std::endl;

using std::string;
using std::vector;
using std::map;
using std::unordered_map;
using std::unordered_set;

const int size_of_int = sizeof(int);
const int size_of_float = sizeof(float);
const int size_of_double = sizeof(double);

////////////////// type definition //////////////////////

typedef unsigned long long ull;
typedef double value_type;
#define VALUE_MPI_TYPE MPI_DOUBLE

typedef unsigned int int4rating;
#define INTR_MPI_TYPE MPI_UNSIGNED
int4rating int4rating_max = std::numeric_limits<unsigned int>::max();
////////////////// type definition //////////////////////

enum Step{Constant=0, Bold_driver=1};
enum Partition{Greedy=0, Item_Partition=1, User_Partition=2};

#endif //BASE_H