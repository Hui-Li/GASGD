#ifndef MPISTRUCTS_H
#define MPISTRUCTS_H

#include "../util/Base.h"

namespace MPIStructs{

    struct rating_assign {
        int4rating global_rating_id;
        int partition_id;
        int user_id;
        int item_id;
        value_type score;
    };

    struct tuple {
        int key;
        int value;
    };

    MPI_Datatype mpi_rating_assign_type;
    MPI_Datatype mpi_tuple_type;

    void init(){
        // ToDo: this should be automatically generated from the definition in MPIStructs.h
        /* create a type for struct rating_assign */
        const int nitems = 5;
        int blocklengths[5] = {1, 1, 1, 1, 1};
        MPI_Datatype types[5] = {INTR_MPI_TYPE, MPI_INT, MPI_INT, MPI_INT, VALUE_MPI_TYPE};

        MPI_Aint offsets[5];

        offsets[0] = offsetof(rating_assign, global_rating_id);
        offsets[1] = offsetof(rating_assign, partition_id);
        offsets[2] = offsetof(rating_assign, user_id);
        offsets[3] = offsetof(rating_assign, item_id);
        offsets[4] = offsetof(rating_assign, score);

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_rating_assign_type);
        MPI_Type_commit(&mpi_rating_assign_type);

        /* create a type for struct tuple */
        const int nitems2 = 2;
        int blocklengths2[2] = {1, 1};
        MPI_Datatype types2[2] = {MPI_INT, MPI_INT};

        MPI_Aint offsets2[2];
        offsets2[0] = offsetof(tuple, key);
        offsets2[1] = offsetof(tuple, value);

        MPI_Type_create_struct(nitems2, blocklengths2, offsets2, types2, &mpi_tuple_type);
        MPI_Type_commit(&mpi_tuple_type);

    }

    void free(){
        MPI_Type_free(&mpi_rating_assign_type);
        MPI_Type_free(&mpi_tuple_type);
    }
}
#endif //MPISTRUCTS_H
