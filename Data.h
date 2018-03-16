#ifndef MPI_DATA_H
#define MPI_DATA_H

#include "util/Base.h"
#include "struct/Rating.h"

class Data {
public:

    static int num_of_workers;
    static int4rating train_rating_num;
    static int4rating test_rating_num;
    static int user_num;
    static int item_num;
    static int machine_id;

    // training ratings which are assigned to this machine
    static vector<vector<Rating> > assigned_ratings;
    // number of ratings for the user/item in local assigned_ratings
//    static vector<unordered_map<int, int> > local_user_rating_nums;
//    static vector<unordered_map<int, int> > local_item_rating_nums;

    // testing ratings which are assigned to this machine
    static vector<Rating> test_ratings;

    // original user/item id -> master id
    static vector<unordered_map<int, int> > user_master_maps;
    static vector<unordered_map<int, int> > item_master_maps;

    // master id -> user/item id
    static unordered_map<int, unordered_set<int> > master_user_map;
    static unordered_map<int, unordered_set<int> > master_item_map;

    // original user/item id -> local id in new data layout
    static vector<unordered_map<int, int> > user_id_maps;
    static vector<unordered_map<int, int> > item_id_maps;

    // start position of each block
    // size: thread number + 1
    // [0] is 0, u_offset[thread number+1] - u_offset[thread number] is number of all user vectors in this thread block
    static vector<int> u_offset;
    static vector<int> i_offset;

};

int Data::num_of_workers = 0;
int4rating Data::train_rating_num = 0;
int4rating Data::test_rating_num = 0;
int Data::user_num = 0;
int Data::item_num = 0;
int Data::machine_id = 0;

vector<vector<Rating> > Data::assigned_ratings;
//vector<unordered_map<int, int> > Data::local_user_rating_nums;
//vector<unordered_map<int, int> > Data::local_item_rating_nums;
vector<Rating> Data::test_ratings;
vector<unordered_map<int, int> > Data::user_master_maps;
vector<unordered_map<int, int> > Data::item_master_maps;
unordered_map<int, unordered_set<int> > Data::master_user_map;
unordered_map<int, unordered_set<int> > Data::master_item_map;
vector<unordered_map<int, int> > Data::user_id_maps;
vector<unordered_map<int, int> > Data::item_id_maps;
vector<int> Data::u_offset;
vector<int> Data::i_offset;

#endif //MPI_DATA_H
