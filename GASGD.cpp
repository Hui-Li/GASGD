#include "util/Base.h"
#include "util/Conf.h"
#include "util/FileUtil.h"
#include "struct/MPIStructs.h"
#include "Partitioner.h"
#include "ASGD.h"

using namespace MPIStructs;

int main(int argc, char **argv) {

    if (!Conf::create(argc, argv)) {
        return 1;
    }

    // check whether MPI provides multiple threading
    int mpi_thread_provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
    if (mpi_thread_provided != MPI_THREAD_MULTIPLE) {
        cerr << "MPI multiple thread not provided!!! ("
             << mpi_thread_provided << " != " << MPI_THREAD_MULTIPLE << ")" << endl;
        exit(1);
    }

    // retrieve MPI task info
    MPI_Comm_rank(MPI_COMM_WORLD, &(Data::machine_id));
    MPI_Comm_size(MPI_COMM_WORLD, &(Conf::num_of_machine));

    FileUtil::readMetaData(Conf::meta_path, Conf::train_data_path, Conf::test_data_path, Data::user_num, Data::item_num, Data::train_rating_num, Data::test_rating_num);

    cout << boost::format{"machine %1%: user_num %2%, item_num %3%, train_rating_num %4%, test_rating_num %5%"} % Data::machine_id % Data::user_num %
            Data::item_num % Data::train_rating_num % Data::test_rating_num
         << endl;

    Data::num_of_workers = Conf::num_of_machine * Conf::num_of_thread;

    MPIStructs::init();

    // Step 1: partition data
    Partitioner partitioner;
    partitioner.greedy_partition();

    // Step 2: Distributed Asynchronous SGD
    ASGD asgd;
    asgd.train();

    MPIStructs::free();
    MPI_Finalize();

    return 0;
}