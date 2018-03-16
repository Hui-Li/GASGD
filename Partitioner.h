#ifndef PARTITIONER_H
#define PARTITIONER_H

#include <mutex>
#include "util/Base.h"
#include "struct/Object.h"
#include "struct/MPIStructs.h"
#include "util/Conf.h"
#include "struct/Rating.h"
#include "util/FileUtil.h"
#include "Data.h"
#include "util/Monitor.h"

using namespace MPIStructs;


class Partitioner {

private:

    int iteration;
    int4rating min_row_index;
    int4rating max_row_index;
    int4rating update_period_size;

    pool *thread_pool = nullptr;
    int *rec_sizes = nullptr;
    int *r_displs = nullptr;
    int4rating *partition_size = nullptr;
    std::mutex partition_size_mutex;

    vector<Object> users;
    vector<Object> items;

    rating_assign *rating_assignment_send = nullptr;
    rating_assign *rating_assign_receive = nullptr;

    int communication_send_size;
    int communication_receive_size;

    vector<Rating> local_ratings;
    concurrent_unordered_map<int, int> user_master;
    concurrent_unordered_map<int, int> item_master;

    void synchronize_master_info(concurrent_unordered_map<int, int> &masters,
                                 vector<Object> &obj, int4rating *partition_size) {

        int send_size = masters.size();

        // tell each node how much data is coming
        MPI_Allgather(&send_size, 1, MPI_INT, rec_sizes, 1, MPI_INT, MPI_COMM_WORLD);

        int rec_obj_master_count = 0;

        for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
            if (machine_id == 0) {
                r_displs[0] = 0;
            } else {
                r_displs[machine_id] = r_displs[machine_id - 1] + rec_sizes[machine_id - 1];
                rec_obj_master_count += rec_sizes[machine_id - 1];
            }
        }

        rec_obj_master_count += rec_sizes[Conf::num_of_machine - 1];

        tuple *send_data = new tuple[send_size];

        concurrent_unordered_map<int, int>::const_iterator master_it;
        int index = 0;
        for (master_it = masters.begin(); master_it != masters.end(); master_it++) {
            send_data[index].key = master_it->first;
            send_data[index].value = master_it->second;
            index++;
        }

        tuple *rec_data = new tuple[rec_obj_master_count];

        MPI_Allgatherv(send_data, send_size, mpi_tuple_type, rec_data, rec_sizes, r_displs, mpi_tuple_type, MPI_COMM_WORLD);

        for (int ind = 0; ind < rec_obj_master_count; ind++) {
            int obj_id = rec_data[ind].key;
            int master_id = rec_data[ind].value;

            // In case some objs/nodes get different master assignment, master with smallest load will be picked
            int prev_master_id = obj[obj_id].master_id;
            if (prev_master_id == -1) {
                obj[obj_id].master_id = master_id;
            } else {
                if (prev_master_id != master_id) {
                    if(partition_size[master_id] < partition_size[prev_master_id]){
                        obj[obj_id].master_id = master_id;
                    }
                }
            }
        }

        delete[] rec_data;
        delete[] send_data;
    }

    void remap_ids(vector<unordered_map<int, int> > &obj_id_maps, vector<unordered_map<int, int> > &obj_master_maps){

        // remap id according to its position in new data layout
        int id = 0;
        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            unordered_map<int, int> &obj_id_map = obj_id_maps[thread_index];
            unordered_map<int, int> &obj_master_map = obj_master_maps[thread_index];

            for (auto iter = obj_master_map.begin(); iter != obj_master_map.end(); iter++) {
                obj_id_map.insert(std::make_pair(iter->first, id));
                id++;
            }
        }
    }


    void format_data(vector<Object> &users, vector<Object> &items){

        // original user/item id -> local id in new data layout
        Data::user_id_maps.resize(Conf::num_of_thread);
        Data::item_id_maps.resize(Conf::num_of_thread);

        // start position of each block
        // size: thread number + 1
        // [0] is 0, u_offset[thread number+1] - u_offset[thread number] is number of all user vectors in this thread block
        Data::u_offset.resize(Conf::num_of_thread + 1, 0);
        Data::i_offset.resize(Conf::num_of_thread + 1, 0);

        // original user/item id -> master id
        Data::user_master_maps.resize(Conf::num_of_thread);
        Data::item_master_maps.resize(Conf::num_of_thread);

//        Data::local_user_rating_nums.resize(Conf::num_of_thread, unordered_map<int, int>());
//        Data::local_item_rating_nums.resize(Conf::num_of_thread, unordered_map<int, int>());

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {

            thread_pool->schedule(std::bind([&](const int thread_index) {
                vector<Rating> &ratings = Data::assigned_ratings[thread_index];
//                unordered_map<int, int> &user_rating_nums = Data::local_user_rating_nums[thread_index];
//                unordered_map<int, int> &item_rating_nums = Data::local_item_rating_nums[thread_index];
                unordered_set<int> user_ids;
                unordered_set<int> item_ids;
                for (Rating rating:ratings) {
                    user_ids.insert(rating.user_id);
                    item_ids.insert(rating.item_id);
//                    auto ptr1 = user_rating_nums.find(rating.user_id);
//                    if (ptr1 == user_rating_nums.end()) {
//                        user_rating_nums[rating.user_id] = 1;
//                    } else {
//                        user_rating_nums[rating.user_id] = user_rating_nums[rating.user_id] + 1;
//                    }
//                    auto ptr2 = item_rating_nums.find(rating.item_id);
//                    if (ptr2 == item_rating_nums.end()) {
//                        item_rating_nums[rating.item_id] = 1;
//                    } else {
//                        item_rating_nums[rating.item_id] = item_rating_nums[rating.item_id] + 1;
//                    }
                }

                for (int user_id:user_ids) {
                    unordered_set<int> &user_assign = users[user_id].assigned_partition_id;
                    if (user_assign.size() == 1) { // this vector does not need synchronization
                        Data::user_master_maps[thread_index][user_id] = Data::num_of_workers;
                    } else {
                        Data::user_master_maps[thread_index][user_id] = users[user_id].master_id;
                    }
                    Data::u_offset[thread_index + 1]++;
                }

                for (int item_id:item_ids) {
                    unordered_set<int> &item_assign = items[item_id].assigned_partition_id;
                    if (item_assign.size() == 1) { // this vector does not need synchronization
                        Data::item_master_maps[thread_index][item_id] = Data::num_of_workers;
                    } else {
                        Data::item_master_maps[thread_index][item_id] = items[item_id].master_id;
                    }
                    Data::i_offset[thread_index + 1]++;
                }
            }, thread_index));

        }

        thread_pool->wait();

        for (int user_id = 0; user_id < users.size(); user_id++) {
            int master_id = users[user_id].master_id;
            if (Data::master_user_map.find(master_id) == Data::master_user_map.end()) {
                Data::master_user_map.insert(std::make_pair(master_id, unordered_set<int>()));
            }
            Data::master_user_map[master_id].insert(user_id);
        }

        for (int item_id = 0; item_id < items.size(); item_id++) {
            int master_id = items[item_id].master_id;
            if (Data::master_item_map.find(master_id) == Data::master_item_map.end()) {
                Data::master_item_map.insert(std::make_pair(master_id, unordered_set<int>()));
            }
            Data::master_item_map[master_id].insert(item_id);
        }

        for (int i = 1; i <= Conf::num_of_thread; i++) {
            Data::u_offset[i] = Data::u_offset[i-1] + Data::u_offset[i];
            Data::i_offset[i] = Data::i_offset[i-1] + Data::i_offset[i];
        }

        remap_ids(Data::user_id_maps, Data::user_master_maps);
        remap_ids(Data::item_id_maps, Data::item_master_maps);

    }

    void partition(const int iter_index, const int thread_index){
        int4rating start = iter_index * update_period_size + min_row_index +
                    thread_index * iteration * update_period_size;
        int4rating end = std::min(start + update_period_size, max_row_index);

        // greedy heuristic
        for (int4rating global_id = start; global_id < end; global_id++) {

            // offset from the start of rating_assignment_send for this iteration
            int4rating offset = global_id - start + thread_index * update_period_size;

            // ratings only contains local copy
            Rating &rating = local_ratings[global_id - min_row_index];

            int user_id = rating.user_id;
            int item_id = rating.item_id;

            rating_assignment_send[offset].global_rating_id = global_id;
            rating_assignment_send[offset].user_id = user_id;
            rating_assignment_send[offset].item_id = item_id;
            rating_assignment_send[offset].score = rating.score;

            Object &user = users[user_id];
            Object &item = items[item_id];

            std::lock_guard<std::mutex> user_lock(*(user.mt));
            std::lock_guard<std::mutex> item_lock(*(item.mt));

            int &user_master_id = user.master_id;
            int &item_master_id = item.master_id;
            unordered_set<int> &u_assign = user.assigned_partition_id;
            unordered_set<int> &i_assign = item.assigned_partition_id;

            // case 1
            if (u_assign.size() == 0 && i_assign.size() == 0) {
                std::lock_guard<std::mutex> partition_size_lock(partition_size_mutex);
                int4rating smallest_size = int4rating_max;
                int smallest_partition_id = -1;
                for (int id = 0; id < Data::num_of_workers; id++) {
                    if (partition_size[id] < smallest_size) {
                        smallest_size = partition_size[id];
                        smallest_partition_id = id;
                    }
                }
                user_master_id = smallest_partition_id;
                item_master_id = smallest_partition_id;
                u_assign.insert(smallest_partition_id);
                i_assign.insert(smallest_partition_id);

                rating_assignment_send[offset].partition_id = smallest_partition_id;
                user_master.insert(std::make_pair(user_id, smallest_partition_id));
                item_master.insert(std::make_pair(item_id, smallest_partition_id));

                partition_size[smallest_partition_id]++;

            } else if (u_assign.size() != 0 && i_assign.size() == 0) { // case 2
                std::lock_guard<std::mutex> partition_size_lock(partition_size_mutex);
                int4rating smallest_size = int4rating_max;
                int smallest_partition_id = -1;
                for (auto ptr = u_assign.begin(); ptr != u_assign.end(); ptr++) {
                    if (partition_size[*ptr] < smallest_size) {
                        smallest_size = partition_size[*ptr];
                        smallest_partition_id = *ptr;
                    }
                }
                item_master_id = smallest_partition_id;
                i_assign.insert(smallest_partition_id);
                rating_assignment_send[offset].partition_id = smallest_partition_id;
                item_master.insert(std::make_pair(item_id, smallest_partition_id));

                partition_size[smallest_partition_id]++;

            } else if (u_assign.size() == 0 && i_assign.size() != 0) { // case 2
                std::lock_guard<std::mutex> partition_size_lock(partition_size_mutex);
                int4rating smallest_size = int4rating_max;
                int smallest_partition_id = -1;
                for (auto ptr = i_assign.begin(); ptr != i_assign.end(); ptr++) {
                    if (partition_size[*ptr] < smallest_size) {
                        smallest_size = partition_size[*ptr];
                        smallest_partition_id = *ptr;
                    }
                }
                user_master_id = smallest_partition_id;
                u_assign.insert(smallest_partition_id);
                rating_assignment_send[offset].partition_id = smallest_partition_id;
                user_master.insert(std::make_pair(user_id, smallest_partition_id));

                partition_size[smallest_partition_id]++;

            } else {

                vector<int> common_data;
                set_intersection(u_assign.begin(), u_assign.end(), i_assign.begin(), i_assign.end(),
                                 std::back_inserter(common_data));

                if (common_data.size() != 0) { // case 3

                    std::lock_guard<std::mutex> partition_size_lock(partition_size_mutex);

                    int4rating smallest_size = int4rating_max;
                    int smallest_partition_id = -1;

                    for (int partition_id:common_data) {

                        if (partition_size[partition_id] < smallest_size) {
                            smallest_size = partition_size[partition_id];
                            smallest_partition_id = partition_id;
                        }
                    }

                    rating_assignment_send[offset].partition_id = smallest_partition_id;

                    partition_size[smallest_partition_id]++;

                } else { // case 4

                    switch(Conf::partition){
                        case Partition::Greedy:{
                            std::lock_guard<std::mutex> partition_size_lock(partition_size_mutex);

                            unordered_set<int> union_set = user.assigned_partition_id;
                            union_set.insert(item.assigned_partition_id.begin(), item.assigned_partition_id.end());

                            int4rating smallest_size = int4rating_max;
                            int smallest_partition_id = -1;

                            for (int partition_id:union_set) {
                                if (partition_size[partition_id] < smallest_size) {
                                    smallest_size = partition_size[partition_id];
                                    smallest_partition_id = partition_id;
                                }
                            }

                            // one of them already has it, but we do not check
                            u_assign.insert(smallest_partition_id);
                            i_assign.insert(smallest_partition_id);
                            rating_assignment_send[offset].partition_id = smallest_partition_id;

                            partition_size[smallest_partition_id]++;
                            break;
                        }
                        case Partition::Item_Partition: {
                            std::lock_guard<std::mutex> partition_size_lock(partition_size_mutex);
                            assert(i_assign.size()==1);
                            // there is only one element
                            rating_assignment_send[offset].partition_id = *(i_assign.begin());
                            partition_size[*(i_assign.begin())]++;
                            break;
                        }
                        case Partition::User_Partition :{
                            std::lock_guard<std::mutex> partition_size_lock(partition_size_mutex);
                            assert(u_assign.size()==1);
                            // there is only one element
                            rating_assignment_send[offset].partition_id = *(u_assign.begin());
                            partition_size[*(u_assign.begin())]++;
                            break;
                        }
                        default:{
                            cerr << "logitical error in Conf::partition " << Conf::partition << endl;
                            exit(1);
                        }
                    }

                }
            }
        }
    }

public:

    Partitioner() {

        // ratings which are assigned to this machine
        Data::assigned_ratings.resize(Conf::num_of_thread);

        // calculate how many number of rows is to be used locally
        // num_rows_per_part * Data::num_of_workers may be larger than train_rating_num
        const int4rating num_rows_per_part = Data::train_rating_num / Data::num_of_workers + ((Data::train_rating_num % Data::num_of_workers > 0) ? 1 : 0);

        update_period_size = num_rows_per_part * Conf::g_period;

        // read data which is needed by all threads in this machine
        min_row_index = Data::machine_id * Conf::num_of_thread * num_rows_per_part;
        max_row_index = std::min(min_row_index + Conf::num_of_thread * num_rows_per_part, Data::train_rating_num);

        if (!FileUtil::readDataLocally(Conf::train_data_path, local_ratings, min_row_index, max_row_index, Data::train_rating_num,
                                       Data::user_num)) {
            cerr << "error in reading training file" << endl;
            exit(1);
        }

        iteration = num_rows_per_part / update_period_size + ((num_rows_per_part % update_period_size > 0) ? 1 : 0);

        thread_pool = new pool(Conf::num_of_thread);
        rec_sizes = new int[Conf::num_of_machine];
        r_displs = new int[Conf::num_of_machine];
        partition_size = new int4rating[Data::num_of_workers];
        std::fill(partition_size, partition_size + Data::num_of_workers, 0);

        users.resize(Data::user_num, Object());
        items.resize(Data::item_num, Object());

        for (int user_id = 0; user_id < Data::user_num; user_id++) {
            users[user_id].id = user_id;
        }

        for (int item_id = 0; item_id < Data::item_num; item_id++) {
            items[item_id].id = item_id;
        }

        communication_send_size = update_period_size * Conf::num_of_thread;
        communication_receive_size = communication_send_size * Conf::num_of_machine;

        // size is the total size over all threads of this machine
        rating_assignment_send = new rating_assign[communication_send_size];
        rating_assign_receive = new rating_assign[communication_receive_size];
    }

    ~Partitioner() {
        delete[] r_displs;
        delete[] rec_sizes;
        delete[] partition_size;
        delete thread_pool;

        delete[] rating_assignment_send;
        delete[] rating_assign_receive;
    }

    void greedy_partition() {

        Monitor timer;
        timer.start();

        unordered_set<int4rating> received_rating_ids;

        // updater thread
        for (int iter_index = 0; iter_index < iteration; iter_index++) {

            user_master.clear();
            item_master.clear();

            // machine_id and thread_index start with 0
            for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {

                thread_pool->schedule(std::bind(
                        [&](const int _iter_index, const int _thread_index) {
                            partition(_iter_index, _thread_index);

                        }, iter_index, thread_index));

            }

            thread_pool->wait();
            MPI_Barrier(MPI_COMM_WORLD);

            // synchronization

            // step 1: synchronize ratings
            // it is possible that communication_send_size is larger than max(int). In that case, tune g_period.
            MPI_Allgather(rating_assignment_send, communication_send_size, mpi_rating_assign_type,
                          rating_assign_receive,
                          communication_send_size, mpi_rating_assign_type, MPI_COMM_WORLD);

            // communication_receive_size can be larger than real_size (only for last iteration),
            // the repeated ones are from last iteration because rating_assignment_send will not be cleaned before sending.
            for (int index = 0; index < communication_receive_size; index++) {

                rating_assign &ra = rating_assign_receive[index];
                if (ra.global_rating_id < 0 || ra.global_rating_id >= Data::train_rating_num ||
                    received_rating_ids.find(ra.global_rating_id) != received_rating_ids.end()) {
                    continue;
                }

                received_rating_ids.insert(ra.global_rating_id);

                if (ra.partition_id >= Data::machine_id * Conf::num_of_thread &&
                    ra.partition_id < (Data::machine_id + 1) * Conf::num_of_thread) {
                    Data::assigned_ratings[ra.partition_id - Data::machine_id * Conf::num_of_thread].push_back(Rating(ra.global_rating_id, ra.user_id, ra.item_id, ra.score));
                }

                // exclude local copy
                if (ra.global_rating_id < max_row_index && ra.global_rating_id >= min_row_index) {
                    continue;
                }

//                concurrent_unordered_map<int, Object>::iterator finder = users.find(ra.user_id);
//                if (finder == users.end()) {
//                    users.insert(std::make_pair(ra.user_id, Object(ra.user_id)));
//                }
//
//                concurrent_unordered_map<int, Object>::iterator finder2 = items.find(ra.item_id);
//                if (finder2 == items.end()) {
//                    items.insert(std::make_pair(ra.item_id, Object(ra.item_id)));
//                }

                users[ra.user_id].assigned_partition_id.insert(ra.partition_id);
                items[ra.item_id].assigned_partition_id.insert(ra.partition_id);

                partition_size[ra.partition_id]++;

            }

            MPI_Barrier(MPI_COMM_WORLD);

            // step 2: synchronize master information
            synchronize_master_info(user_master, users, partition_size);
            synchronize_master_info(item_master, items, partition_size);

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (Data::machine_id==0) {

            for (int i = 0; i < Data::num_of_workers; i++) {
                cout << boost::format{"Partition %1%: number of ratings %2%"} % i % partition_size[i] << endl;
            }

//        for (int i = 0; i < Data::num_of_workers; i++) {
//            cout << boost::format{"Machine %1%, Thread %2%, Partition size %3%"} % Data::machine_id % i % partition_size[i] << endl;
//        }
//
//        for (int i = 0; i < Conf::num_of_thread; i++) {
//            cout << boost::format{"Machine %1%, Thread %2%, Assigned Ratings %3%"} % Data::machine_id % i % Data::assigned_ratings[i].size() << endl;
//        }
//        cout << "-----------" << endl;
        }

        format_data(users, items);

        timer.stop();

        cout << boost::format{"machine %1%: partitioning time %2% secs"} % Data::machine_id % timer.getElapsedTime() << endl;

    }

};


#endif //PARTITIONER_H
