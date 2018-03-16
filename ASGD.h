#ifndef ASGD_H
#define ASGD_H

#include "util/Base.h"
#include "struct/Rating.h"
#include "util/Conf.h"
#include "util/RandomUtil.h"
#include "Data.h"
#include "util/Monitor.h"

class ASGD {

private:

    pool *thread_pool = nullptr;
    value_type *user_vecs; // the 0-th dimension is for weight
    value_type *item_vecs;
    int rank; // k+1
    vector<vector<int4rating> > rating_indices;

    //// for computation phase
    // size for machine block
    int *u_send_machine_sizes = nullptr;
    int *i_send_machine_sizes = nullptr;
    int *u_rec_machine_sizes = nullptr;
    int *i_rec_machine_sizes = nullptr;

    // start indices for thread
    int *u_send_displ_thread = nullptr;
    int *i_send_displ_thread = nullptr;
    int *u_rec_displ_thread = nullptr;
    int *i_rec_displ_thread = nullptr;

    // size for thread block
    int *u_id_send_size_array = nullptr;
    int *u_id_rec_size_array = nullptr;
    int *i_id_send_size_array = nullptr;
    int *i_id_rec_size_array = nullptr;

    // start indices for machine
    int *u_send_displ_machine = nullptr;
    int *i_send_displ_machine = nullptr;
    int *u_rec_displ_machine = nullptr;
    int *i_rec_displ_machine = nullptr;
    //// for computation phase

    //// for copy_master_to_workers
    int *rec_sizes = nullptr;
    int *r_displs = nullptr;
    //// for copy_master_to_workers

    //// for prediction on test data
    value_type *final_user_vecs = nullptr;
    value_type *final_item_vecs = nullptr;


    inline value_type inner_product(value_type *a, value_type *b, int size) {
        value_type result = 0;
        for (int i = 0; i < size; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    inline value_type sqr_norm(value_type *a, int size) {
        value_type result = 0;
        for (int i = 0; i < size; i++) {
            result += a[i] * a[i];
        }
        return result;
    }

    // initialize latent vectors
    inline void init_vec() {
        user_vecs = new value_type[Data::u_offset[Conf::num_of_thread] * rank];
        item_vecs = new value_type[Data::i_offset[Conf::num_of_thread] * rank];

        for(int i = 0; i < Data::u_offset[Conf::num_of_thread] * rank; i++) {
            user_vecs[i] = RandomUtil::uniform_real();
        }

        for (int i = 0; i < Data::i_offset[Conf::num_of_thread] * rank; i++) {
            item_vecs[i] = RandomUtil::uniform_real();
        }

        final_user_vecs = new value_type[Data::user_num * (Conf::k)];
        final_item_vecs = new value_type[Data::item_num * (Conf::k)];


//        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
//            thread_pool->schedule(std::bind([&](const int thread_index) {
//                // the 0-th dimension is for weight
//                value_type *sub_user_vecs = &user_vecs[Data::u_offset[thread_index] * rank];
//                value_type *sub_item_vecs = &item_vecs[Data::i_offset[thread_index] * rank];
//                int user_num = Data::u_offset[thread_index + 1] - Data::u_offset[thread_index];
//                int item_num = Data::i_offset[thread_index + 1] - Data::i_offset[thread_index];
//
//                // The starting points were chosen by taking i.i.d. samples from the Uniform(-0.5,0.5) distribution
//                for (int user_ind = 0; user_ind < user_num; user_ind++) {
//                    sub_user_vecs[user_ind * rank] = 0.0;
//                    for (int dim = user_ind * rank + 1; dim < (user_ind + 1) * rank; dim++) {
//                        sub_user_vecs[dim] = RandomUtil::uniform_real();
//                    }
//                }
//
//                for (int item_ind = 0; item_ind < item_num; item_ind++) {
//                    sub_item_vecs[item_ind * rank] = 0.0;
//                    for (int dim = item_ind * rank + 1; dim < (item_ind + 1) * rank; dim++) {
//                        sub_item_vecs[dim] = RandomUtil::uniform_real();
//                    }
//                }
//            }, thread_index));
//        }

//        thread_pool->wait();
    }

    // randomly shuffle ratings
    inline void shuffle_ratings() {

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            vector<int4rating > &rating_index = rating_indices[thread_index];
            std::random_shuffle(rating_index.begin(), rating_index.end());
        }

//        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
//            thread_pool->schedule(std::bind([&](const int thread_index) {
//                vector<int> &rating_index = rating_indices[thread_index];
//                std::random_shuffle(rating_index.begin(), rating_index.end());
//            }, thread_index));
//        }
//
//        thread_pool->wait();
    }

    void compute_train_statistics(value_type &global_rmse, value_type &global_loss, value_type &global_reg) {

        vector<value_type> losses(Conf::num_of_thread, 0);
        vector<value_type> regs(Conf::num_of_thread, 0);

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            thread_pool->schedule(std::bind([&](const int thread_index) {
                vector<Rating> &ratings = Data::assigned_ratings[thread_index];
                unordered_map<int, int> &user_id_map = Data::user_id_maps[thread_index];
                unordered_map<int, int> &item_id_map = Data::item_id_maps[thread_index];

                for (auto rating:ratings) {

                    int local_user_id = user_id_map[rating.user_id];
                    int local_item_id = item_id_map[rating.item_id];

                    value_type *user_vec = &user_vecs[local_user_id * rank];
                    value_type *item_vec = &item_vecs[local_item_id * rank];

                    value_type error = rating.score - inner_product(user_vec + 1, item_vec + 1, Conf::k);
                    losses[thread_index] += error * error;
                    regs[thread_index] += sqr_norm(user_vec + 1, Conf::k) + sqr_norm(item_vec + 1, Conf::k);
                }

            }, thread_index));
        }

        thread_pool->wait();

        value_type local_loss = 0;

        for (value_type value:losses) {
            local_loss += value;
        }

        value_type local_reg = 0;

        for (value_type value:regs) {
            local_reg += value;
        }

        global_loss = 0;
        global_reg = 0;

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Reduce(&local_loss, &global_loss, 1, VALUE_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_reg, &global_reg, 1, VALUE_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

        global_rmse = sqrt(global_loss / Data::train_rating_num);
    }

    void computation_phase(const int step, int4rating *size_of_folder, vector<concurrent_unordered_set<int> > &user_id_to_send,
                           vector<concurrent_unordered_set<int> > &item_id_to_send) {

        // computation phase
        for(int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            thread_pool->schedule(std::bind([&](const int thread_index) {

                vector<Rating> &ratings = Data::assigned_ratings[thread_index];
                vector<int4rating> &rating_index = rating_indices[thread_index];

                unordered_map<int, int> &user_id_map = Data::user_id_maps[thread_index];
                unordered_map<int, int> &item_id_map = Data::item_id_maps[thread_index];

                concurrent_unordered_set<int> &sub_user_id_to_send = user_id_to_send[thread_index];
                concurrent_unordered_set<int> &sub_item_id_to_send = item_id_to_send[thread_index];
                sub_user_id_to_send.clear();
                sub_item_id_to_send.clear();

                unordered_map<int, int> &user_master_map = Data::user_master_maps[thread_index];
                unordered_map<int, int> &item_master_map = Data::item_master_maps[thread_index];

                int4rating start_ind = step * size_of_folder[thread_index];
                int4rating end_ind;
                if (step != Conf::folder - 1) {
                    end_ind = start_ind + size_of_folder[thread_index];
                } else {
                    end_ind = rating_index.size();
                }

                value_type *last_user_vec = new value_type[rank];

                for (int4rating rating_ind = start_ind; rating_ind < end_ind; rating_ind++) {

                    Rating &rating = ratings[rating_ind];

                    if (user_master_map[rating.user_id] != Data::num_of_workers) {
                        sub_user_id_to_send.insert(rating.user_id);
                    }

                    if (item_master_map[rating.item_id] != Data::num_of_workers) {
                        sub_item_id_to_send.insert(rating.item_id);
                    }

                    int local_user_id = user_id_map[rating.user_id];
                    int local_item_id = item_id_map[rating.item_id];

                    value_type *user_vec = &user_vecs[local_user_id * rank];
                    value_type *item_vec = &item_vecs[local_item_id * rank];

                    value_type error = rating.score - inner_product(user_vec + 1, item_vec + 1, Conf::k);

                    std::copy(user_vec, user_vec + rank, last_user_vec);

                    for (int ind = 1; ind < rank; ind++) {
                        user_vec[ind] = last_user_vec[ind] + 2 * Conf::learning_rate * (error * item_vec[ind] - Conf::lambda * last_user_vec[ind]);
                        item_vec[ind] = item_vec[ind] + 2 * Conf::learning_rate * (error * last_user_vec[ind] - Conf::lambda * item_vec[ind]);
                    }

                    // weight
                    user_vec[0] += 1.0;
                    item_vec[0] += 1.0;
                }

                delete[] last_user_vec;

            }, thread_index));
        }

        thread_pool->wait();
    }

    void init_send_data(vector<vector<vector<int> > > &send_user_ids,
                        vector<vector<vector<int> > > &send_item_ids, int *send_user_ids_array,
                        int *send_item_ids_array, int *u_id_send_size_array, int *i_id_send_size_array,
                        value_type *send_user_latent_vec, value_type *send_item_latent_vec, int *u_send_displ_thread,
                        int *i_send_displ_thread) {

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            thread_pool->schedule(std::bind([&](const int thread_index) {

                unordered_map<int, int> &user_master_map = Data::user_master_maps[thread_index];
                unordered_map<int, int> &item_master_map = Data::item_master_maps[thread_index];

                unordered_map<int, int> &user_id_map = Data::user_id_maps[thread_index];
                unordered_map<int, int> &item_id_map = Data::item_id_maps[thread_index];

                for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {

                    int u_start = u_send_displ_thread[machine_id * Conf::num_of_thread + thread_index];

                    vector<int> &u_ids = send_user_ids[machine_id][thread_index];
                    std::copy(u_ids.begin(), u_ids.end(), send_user_ids_array + u_start);

                    int offset = 0;
                    for (int user_id:u_ids) {
                        value_type *send_user_vec = &user_vecs[user_id_map[user_id] * rank];
                        std::copy(send_user_vec, send_user_vec + rank,
                                  send_user_latent_vec + (u_start + offset) * rank);
                        offset++;
                    }

                    int i_start = i_send_displ_thread[machine_id * Conf::num_of_thread + thread_index];

                    vector<int> &i_ids = send_item_ids[machine_id][thread_index];
                    std::copy(i_ids.begin(), i_ids.end(), send_item_ids_array + i_start);

                    offset = 0;
                    for (int item_id:i_ids) {
                        value_type *send_item_vec = &item_vecs[item_id_map[item_id] * rank];
                        std::copy(send_item_vec, send_item_vec + rank,
                                  send_item_latent_vec + (i_start + offset) * rank);
                        offset++;
                    }
                }
            }, thread_index));
        }

        thread_pool->wait();
    }

    void update_local_master_copies(int *obj_rec_displ_thread, const int obj_rec_id_size,
                                    int *rec_obj_ids_array, value_type *rec_obj_latent_vec,
                                    vector<unordered_map<int, int> > &obj_id_maps, value_type *obj_vecs) {

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            thread_pool->schedule(std::bind([&](const int thread_index) {

                unordered_map<int, value_type> weight;

                // first pass: calculate weights
                for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
                    int index = obj_rec_displ_thread[machine_id * Conf::num_of_thread + thread_index];
                    int size;

                    if ((machine_id * Conf::num_of_thread + thread_index) == (Data::num_of_workers - 1)) {
                        size = obj_rec_id_size - index;
                    } else {
                        size = obj_rec_displ_thread[machine_id * Conf::num_of_thread + thread_index + 1] - index;
                    }

                    int *sub_rec_obj_ids = &rec_obj_ids_array[index];
                    value_type *sub_rec_obj_latent_vec = &rec_obj_latent_vec[index * rank];

                    for (int i = 0; i < size; i++) {
                        int obj_id = sub_rec_obj_ids[i];
                        auto finder = weight.find(obj_id);
                        if (finder == weight.end()) {
                            weight[obj_id] = sub_rec_obj_latent_vec[0];
                        } else {
                            weight[obj_id] += sub_rec_obj_latent_vec[0];
                        }
                    }

                }

                // second pass: weighted sum
                unordered_map<int, int> &obj_id_map = obj_id_maps[thread_index];

                // set local master copies to zero
                // At the same time, weights are set to zero so there is no need to reset weight after update
                for (auto ptr = weight.begin(); ptr != weight.end(); ptr++) {
                    int local_obj_id = obj_id_map[ptr->first];

                    value_type *sub_obj_vec = &obj_vecs[local_obj_id * rank];

                    for (int i = 0; i < rank; i++) {
                        sub_obj_vec[i] = 0;
                    }
                }

                for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
                    int index = obj_rec_displ_thread[machine_id * Conf::num_of_thread + thread_index];
                    int size;

                    if ((machine_id * Conf::num_of_thread + thread_index) == (Data::num_of_workers - 1)) {
                        size = obj_rec_id_size - index;
                    } else {
                        size = obj_rec_displ_thread[machine_id * Conf::num_of_thread + thread_index + 1] - index;
                    }

                    int *sub_rec_obj_ids = &rec_obj_ids_array[index];
                    value_type *sub_rec_obj_latent_vec = &rec_obj_latent_vec[index * rank];

                    for (int i = 0; i < size; i++) {
                        int obj_id = sub_rec_obj_ids[i];
                        int local_obj_id = obj_id_map[obj_id];

                        value_type *sub_obj_vec = &obj_vecs[local_obj_id * rank];

                        for (int i = 1; i < rank; i++) {
                            sub_obj_vec[i] += sub_rec_obj_latent_vec[0] * sub_rec_obj_latent_vec[i];
                        }
                    }
                }

                for (auto ptr = weight.begin(); ptr != weight.end(); ptr++) {
                    int local_obj_id = obj_id_map[ptr->first];
                    value_type *sub_obj_vec = &obj_vecs[local_obj_id * rank];
                    for (int i = 1; i < rank; i++) {
                        sub_obj_vec[i] /= ptr->second;
                    }
                }

            }, thread_index));
        }

        thread_pool->wait();
    }

    void copy_master_to_workers(const int rec_ids_array_size, int *rec_ids_array,
                                vector<unordered_map<int, int> > &obj_id_maps, value_type *obj_vecs,
                                vector<int> &offset) {
        // obj ids of which objs will be updated by the updated master vectors.
        // obj ids can be obtained from rec_ids_array (rec_user_ids_array and rec_item_ids_array)
        std::unordered_set<int> *obj_ids_set = new std::unordered_set<int>();
        for (int i = 0; i < rec_ids_array_size; i++) {
            obj_ids_set->insert(rec_ids_array[i]);
        }

        int send_id_size = obj_ids_set->size();

        // tell each node how much data is coming
        MPI_Allgather(&send_id_size, 1, MPI_INT, rec_sizes, 1, MPI_INT, MPI_COMM_WORLD);

        int rec_id_size = 0;

        for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
            if (machine_id == 0) {
                r_displs[0] = 0;
            } else {
                r_displs[machine_id] = r_displs[machine_id - 1] + rec_sizes[machine_id - 1];
                rec_id_size += rec_sizes[machine_id - 1];
            }
        }
        rec_id_size += rec_sizes[Conf::num_of_machine - 1];

        int *send_ids = new int[send_id_size];
        std::copy(obj_ids_set->begin(), obj_ids_set->end(), send_ids);
        delete obj_ids_set;

        int *rec_ids = new int[rec_id_size];

        // first send ids
        MPI_Allgatherv(send_ids, send_id_size, MPI_INT, rec_ids, rec_sizes, r_displs, MPI_INT, MPI_COMM_WORLD);

        // Then send master vectors
        for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
            r_displs[machine_id] *= rank;
            rec_sizes[machine_id] *= rank;
        }

        value_type *send_latent_vecs = new value_type[send_id_size * rank];
        value_type *rec_latent_vecs = new value_type[rec_id_size * rank];

        int index = 0;
        for (int i = 0; i < send_id_size; i++) {
            int send_id = send_ids[i];
            int contain_thread_id = -1;
            int local_id = -1;

            for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
                unordered_map<int, int> &obj_id_map = obj_id_maps[thread_index];
                auto finder = obj_id_map.find(send_id);
                if (finder != obj_id_map.end()) {
                    local_id = finder->second;
                    contain_thread_id = thread_index;
                    break; // master only exists in one thread block
                }
            }

            if (contain_thread_id == -1 || local_id == -1) {
                cerr << "logical error in copy_master_to_workers!" << endl;
                exit(1);
            }

            value_type *latent_vec = &obj_vecs[local_id * rank];

            std::copy(latent_vec, latent_vec + rank, send_latent_vecs + index * rank);
            index++;
        }

        delete[] send_ids;

        // broadcast masters which are updated during this folder to all nodes, no matter whether this node has slave vector or not
        MPI_Allgatherv(send_latent_vecs, send_id_size * rank, VALUE_MPI_TYPE, rec_latent_vecs, rec_sizes, r_displs,
                       VALUE_MPI_TYPE, MPI_COMM_WORLD);

        delete[] send_latent_vecs;

        // copy master to worker
        for (int i = 0; i < rec_id_size; i++) {
            int obj_id = rec_ids[i];

            for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {

                unordered_map<int, int> &obj_id_map = obj_id_maps[thread_index];
                auto finder = obj_id_map.find(obj_id);
                if (finder != obj_id_map.end()) {
                    int local_id = finder->second;
                    value_type *latent_vec = &obj_vecs[local_id * rank];
                    value_type *master_vec = &rec_latent_vecs[i * rank];
                    std::copy(master_vec, master_vec + rank, latent_vec);
                }
            }
        }

        delete[] rec_ids;
        delete[] rec_latent_vecs;
    }

    void communicate_phase(vector<concurrent_unordered_set<int> > &user_id_to_send,
                           vector<concurrent_unordered_set<int> > &item_id_to_send) {

        // machine block -> thread block -> ids
        vector<vector<vector<int> > > send_user_ids(Conf::num_of_machine,
                                                    vector<vector<int> >(Conf::num_of_thread, vector<int>()));
        vector<vector<vector<int> > > send_item_ids(Conf::num_of_machine,
                                                    vector<vector<int> >(Conf::num_of_thread, vector<int>()));

        std::fill(u_id_send_size_array, u_id_send_size_array + Data::num_of_workers, 0);
        std::fill(i_id_send_size_array, i_id_send_size_array + Data::num_of_workers, 0);

        int u_send_id_size = 0;
        int i_send_id_size = 0;

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            concurrent_unordered_set<int> &sub_user_id_to_send = user_id_to_send[thread_index];
            unordered_map<int, int> &user_master_map = Data::user_master_maps[thread_index];

            for (int user_id:sub_user_id_to_send) {
                int master_id = user_master_map[user_id];
                int machine_id = master_id / Conf::num_of_thread;
                int local_master_id = master_id % Conf::num_of_thread;
                send_user_ids[machine_id][local_master_id].push_back(user_id);
                u_send_id_size++;

                u_id_send_size_array[master_id]++;
            }

            concurrent_unordered_set<int> &sub_item_id_to_send = item_id_to_send[thread_index];
            unordered_map<int, int> &item_master_map = Data::item_master_maps[thread_index];

            for (int item_id:sub_item_id_to_send) {
                int master_id = item_master_map[item_id];
                int machine_id = master_id / Conf::num_of_thread;
                int local_master_id = master_id % Conf::num_of_thread;
                send_item_ids[machine_id][local_master_id].push_back(item_id);
                i_send_id_size++;

                i_id_send_size_array[master_id]++;
            }
        }

        // Tell each node how much data is coming
        MPI_Alltoall(u_id_send_size_array, Conf::num_of_thread, MPI_INT, u_id_rec_size_array, Conf::num_of_thread,
                     MPI_INT, MPI_COMM_WORLD);
        MPI_Alltoall(i_id_send_size_array, Conf::num_of_thread, MPI_INT, i_id_rec_size_array, Conf::num_of_thread,
                     MPI_INT, MPI_COMM_WORLD);

        int u_rec_id_size = 0;
        int i_rec_id_size = 0;

        for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
            u_send_machine_sizes[machine_id] = 0;
            i_send_machine_sizes[machine_id] = 0;
            u_rec_machine_sizes[machine_id] = 0;
            i_rec_machine_sizes[machine_id] = 0;

            for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
                int index = machine_id * Conf::num_of_thread + thread_index;
                u_send_machine_sizes[machine_id] += u_id_send_size_array[index];
                i_send_machine_sizes[machine_id] += i_id_send_size_array[index];

                u_rec_machine_sizes[machine_id] += u_id_rec_size_array[index];
                i_rec_machine_sizes[machine_id] += i_id_rec_size_array[index];

                if (index == 0) {
                    u_send_displ_thread[index] = 0;
                    i_send_displ_thread[index] = 0;
                    u_rec_displ_thread[index] = 0;
                    i_rec_displ_thread[index] = 0;
                } else {
                    u_send_displ_thread[index] = u_send_displ_thread[index - 1] + u_id_send_size_array[index-1];
                    i_send_displ_thread[index] = i_send_displ_thread[index - 1] + i_id_send_size_array[index-1];
                    u_rec_displ_thread[index] = u_rec_displ_thread[index - 1] + u_id_rec_size_array[index-1];
                    i_rec_displ_thread[index] = i_rec_displ_thread[index - 1] + i_id_rec_size_array[index-1];
                }
            }

            u_rec_id_size += u_rec_machine_sizes[machine_id];
            i_rec_id_size += i_rec_machine_sizes[machine_id];

            if (machine_id == 0) {
                u_send_displ_machine[0] = 0;
                i_send_displ_machine[0] = 0;
                u_rec_displ_machine[0] = 0;
                i_rec_displ_machine[0] = 0;
            } else {
                u_send_displ_machine[machine_id] = u_send_displ_machine[machine_id - 1] + u_send_machine_sizes[machine_id - 1];
                i_send_displ_machine[machine_id] = i_send_displ_machine[machine_id - 1] + i_send_machine_sizes[machine_id - 1];
                u_rec_displ_machine[machine_id] = u_rec_displ_machine[machine_id - 1] + u_rec_machine_sizes[machine_id - 1];
                i_rec_displ_machine[machine_id] = i_rec_displ_machine[machine_id - 1] + i_rec_machine_sizes[machine_id - 1];
            }
        }

        // now send ids corresponding to each vector and then send latent vectors
        int *send_user_ids_array = new int[u_send_id_size];
        int *send_item_ids_array = new int[i_send_id_size];
        int *rec_user_ids_array = new int[u_rec_id_size];
        int *rec_item_ids_array = new int[i_rec_id_size];

        value_type *send_user_latent_vec = new value_type[u_send_id_size * rank];
        value_type *send_item_latent_vec = new value_type[i_send_id_size * rank];
        value_type *rec_user_latent_vec = new value_type[u_rec_id_size * rank];
        value_type *rec_item_latent_vec = new value_type[i_rec_id_size * rank];

        init_send_data(send_user_ids, send_item_ids, send_user_ids_array, send_item_ids_array, u_id_send_size_array,
                       i_id_send_size_array, send_user_latent_vec, send_item_latent_vec, u_send_displ_thread,
                       i_send_displ_thread);

        // send ids
        MPI_Alltoallv(send_user_ids_array, u_send_machine_sizes, u_send_displ_machine, MPI_INT, rec_user_ids_array,
                      u_rec_machine_sizes, u_rec_displ_machine, MPI_INT, MPI_COMM_WORLD);
        MPI_Alltoallv(send_item_ids_array, i_send_machine_sizes, i_send_displ_machine, MPI_INT, rec_item_ids_array,
                      i_rec_machine_sizes, i_rec_displ_machine, MPI_INT, MPI_COMM_WORLD);

        // send latent vectors
        for (int i = 0; i < Conf::num_of_machine; i++) {
            u_send_displ_machine[i] *= rank;
            i_send_displ_machine[i] *= rank;
            u_rec_displ_machine[i] *= rank;
            i_rec_displ_machine[i] *= rank;
            u_send_machine_sizes[i] *= rank;
            i_send_machine_sizes[i] *= rank;
            u_rec_machine_sizes[i] *= rank;
            i_rec_machine_sizes[i] *= rank;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Alltoallv(send_user_latent_vec, u_send_machine_sizes, u_send_displ_machine, VALUE_MPI_TYPE,
                      rec_user_latent_vec,
                      u_rec_machine_sizes, u_rec_displ_machine, VALUE_MPI_TYPE, MPI_COMM_WORLD);
        MPI_Alltoallv(send_item_latent_vec, i_send_machine_sizes, i_send_displ_machine, VALUE_MPI_TYPE,
                      rec_item_latent_vec,
                      i_rec_machine_sizes, i_rec_displ_machine, VALUE_MPI_TYPE, MPI_COMM_WORLD);

        // update master copies
        update_local_master_copies(u_rec_displ_thread, u_rec_id_size, rec_user_ids_array, rec_user_latent_vec, Data::user_id_maps, user_vecs);
        update_local_master_copies(i_rec_displ_thread, i_rec_id_size, rec_item_ids_array, rec_item_latent_vec, Data::item_id_maps, item_vecs);

        MPI_Barrier(MPI_COMM_WORLD);

        // copy master to workers
        copy_master_to_workers(u_rec_id_size, rec_user_ids_array, Data::user_id_maps, user_vecs, Data::u_offset);
        copy_master_to_workers(i_rec_id_size, rec_item_ids_array, Data::item_id_maps, item_vecs, Data::i_offset);

        delete[] send_user_ids_array;
        delete[] send_item_ids_array;
        delete[] rec_user_ids_array;
        delete[] rec_item_ids_array;

        delete[] send_user_latent_vec;
        delete[] send_item_latent_vec;
        delete[] rec_user_latent_vec;
        delete[] rec_item_latent_vec;
    }

    void final_master_broadcast(value_type *final_obj_vecs, value_type *local_obj_vecs, vector<int> &offset,
                                vector<unordered_map<int, int> > &obj_id_maps,
                                unordered_map<int, unordered_set<int> > &master_obj_map) {

        // send master vector to all nodes
        vector<int> obj_ids;
        vector<value_type> send_obj_master_vecs;

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {

            int master_id = Data::machine_id * Conf::num_of_thread + thread_index;
            unordered_set<int> &set = master_obj_map[master_id];
            unordered_map<int, int> &obj_id_map = obj_id_maps[thread_index];

            for (int obj_id:set) {

                obj_ids.push_back(obj_id);

                auto finder = obj_id_map.find(obj_id);
                if (finder != obj_id_map.end()) {
                    int local_obj_id = finder->second;
                    value_type *latent_obj_vec = local_obj_vecs + local_obj_id * rank + 1;
                    for (int i = 0; i < Conf::k; i++) {
                        send_obj_master_vecs.push_back(latent_obj_vec[i]);
                    }
                } else {
                    cerr << "logical error in final_master_broadcast!" << endl;
                    exit(1);
                }
            }
        }

        int send_id_size = obj_ids.size();

        // tell each node how much data is coming
        MPI_Allgather(&send_id_size, 1, MPI_INT, rec_sizes, 1, MPI_INT, MPI_COMM_WORLD);

        int rec_id_size = 0;

        for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
            if (machine_id == 0) {
                r_displs[0] = 0;
            } else {
                r_displs[machine_id] = r_displs[machine_id - 1] + rec_sizes[machine_id - 1];
                rec_id_size += rec_sizes[machine_id - 1];
            }
        }
        rec_id_size += rec_sizes[Conf::num_of_machine - 1];

        int *rec_ids = new int[rec_id_size];

        // first send ids
        MPI_Allgatherv(&(obj_ids[0]), send_id_size, MPI_INT, rec_ids, rec_sizes, r_displs, MPI_INT, MPI_COMM_WORLD);

        // Then send master vectors
        for (int machine_id = 0; machine_id < Conf::num_of_machine; machine_id++) {
            r_displs[machine_id] *= Conf::k;
            rec_sizes[machine_id] *= Conf::k;
        }

        value_type *rec_latent_vecs = new value_type[rec_id_size * Conf::k];

        // broadcast masters to all nodes
        MPI_Allgatherv(&(send_obj_master_vecs[0]), send_id_size * Conf::k, VALUE_MPI_TYPE, rec_latent_vecs, rec_sizes, r_displs,
                       VALUE_MPI_TYPE, MPI_COMM_WORLD);

        for (int i = 0; i < rec_id_size; i++) {
            int obj_id = rec_ids[i];
            std::copy(rec_latent_vecs + i * Conf::k, rec_latent_vecs + (i + 1) * Conf::k,
                      final_obj_vecs + obj_id * Conf::k);
        }

        delete[] rec_ids;
        delete[] rec_latent_vecs;
    }

    void init() {

        RandomUtil::init_seed();

        // the 0-th dimension is for weight
        this->rank = Conf::k + 1;
        thread_pool = new pool(Conf::num_of_thread);

        //// for computation phase
        // size for machine block
        u_send_machine_sizes = new int[Conf::num_of_machine];
        i_send_machine_sizes = new int[Conf::num_of_machine];
        u_rec_machine_sizes = new int[Conf::num_of_machine];
        i_rec_machine_sizes = new int[Conf::num_of_machine];

        // start indices for thread
        u_send_displ_thread = new int[Data::num_of_workers];
        i_send_displ_thread = new int[Data::num_of_workers];
        u_rec_displ_thread = new int[Data::num_of_workers];
        i_rec_displ_thread = new int[Data::num_of_workers];

        // size for thread block
        u_id_send_size_array = new int[Data::num_of_workers];
        u_id_rec_size_array = new int[Data::num_of_workers];
        i_id_send_size_array = new int[Data::num_of_workers];
        i_id_rec_size_array = new int[Data::num_of_workers];

        // start indices for machine
        u_send_displ_machine = new int[Conf::num_of_machine];
        i_send_displ_machine = new int[Conf::num_of_machine];
        u_rec_displ_machine = new int[Conf::num_of_machine];
        i_rec_displ_machine = new int[Conf::num_of_machine];
        //// for computation phase

        //// for copy_master_to_workers
        rec_sizes = new int[Conf::num_of_machine];
        r_displs = new int[Conf::num_of_machine];
        //// for copy_master_to_workers

        // read testing data
        int4rating test_num_rows_per_part = Data::test_rating_num / Data::num_of_workers + ((Data::test_rating_num % Data::num_of_workers > 0) ? 1 : 0);

        int4rating test_min_row_index = Data::machine_id * Conf::num_of_thread * test_num_rows_per_part;
        int4rating test_max_row_index = std::min(test_min_row_index + Conf::num_of_thread * test_num_rows_per_part, Data::test_rating_num);

        if (!FileUtil::readDataLocally(Conf::test_data_path, Data::test_ratings, test_min_row_index, test_max_row_index, Data::test_rating_num,
                                       Data::user_num)) {
            cerr << "error in reading testing file" << endl;
            exit(1);
        }
    }

    void predict(value_type &test_loss, value_type &test_rmse){

        // broadcast master vectors to all nodes
        final_master_broadcast(final_user_vecs, user_vecs, Data::u_offset, Data::user_id_maps, Data::master_user_map);
        final_master_broadcast(final_item_vecs, item_vecs, Data::i_offset, Data::item_id_maps, Data::master_item_map);

        MPI_Barrier(MPI_COMM_WORLD);

//        int user_id = 15;
//        int user_index = Data::user_id_maps[0][user_id];
//        cout << user_index << endl;
//        for(int i=0;i<Conf::k;i++){
//            cout << i << ": " << *(final_user_vecs+user_id*Conf::k+i) << "," << *(user_vecs+ user_index*rank + 1 + i) << endl;
//        }
//        exit(1);

        int4rating test_workload = Data::test_ratings.size() / Conf::num_of_thread + ((Data::test_ratings.size() % Conf::num_of_thread == 0)?0:1);

        vector<value_type> test_losses(Conf::num_of_thread, 0);

        for (int thread_index = 0; thread_index < Conf::num_of_thread; thread_index++) {
            thread_pool->schedule(std::bind([&](const int thread_index) {
                int4rating start = test_workload * thread_index;
                int4rating end = ((start + test_workload) < Data::test_ratings.size())? (start + test_workload) : Data::test_ratings.size();

                for (int4rating i = start; i < end; i++) {
                    Rating &rating = Data::test_ratings[i];
                    value_type *user_vec = &final_user_vecs[rating.user_id * Conf::k];
                    value_type *item_vec = &final_item_vecs[rating.item_id * Conf::k];
                    value_type error = rating.score - inner_product(user_vec, item_vec, Conf::k);
                    test_losses[thread_index] += error * error;
                }
            }, thread_index));
        }

        thread_pool->wait();

        value_type local_test_loss = 0;

        for (value_type value:test_losses) {
            local_test_loss += value;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Reduce(&local_test_loss, &test_loss, 1, VALUE_MPI_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);

        test_rmse = sqrt(test_loss / Data::test_rating_num);

    }

public:

    ~ASGD() {

        delete[] user_vecs;
        delete[] item_vecs;

        delete[] u_id_send_size_array;
        delete[] u_id_rec_size_array;
        delete[] i_id_send_size_array;
        delete[] i_id_rec_size_array;

        delete[] u_send_machine_sizes;
        delete[] i_send_machine_sizes;
        delete[] u_rec_machine_sizes;
        delete[] i_rec_machine_sizes;

        delete[] u_send_displ_machine;
        delete[] i_send_displ_machine;
        delete[] u_rec_displ_machine;
        delete[] i_rec_displ_machine;

        delete[] u_send_displ_thread;
        delete[] i_send_displ_thread;
        delete[] u_rec_displ_thread;
        delete[] i_rec_displ_thread;

        delete[] rec_sizes;
        delete[] r_displs;

        delete[] final_user_vecs;
        delete[] final_item_vecs;
        delete thread_pool;
    }

    void train() {

        init();

        int4rating *size_of_folder = new int4rating[Conf::num_of_thread];

        rating_indices.resize(Conf::num_of_thread);

        for (int i = 0; i < Conf::num_of_thread; i++) {
            vector<int4rating> &rating_index = rating_indices[i];
            rating_index.resize(Data::assigned_ratings[i].size());
            std::iota(std::begin(rating_index), std::end(rating_index), 0);

            if (rating_index.size() % Conf::folder == 0) {
                size_of_folder[i] = rating_index.size() / Conf::folder;
            } else {
                size_of_folder[i] = rating_index.size() / Conf::folder + 1;
            }
        }

        // initialize latent vector
        init_vec();

        vector<concurrent_unordered_set<int> > user_id_to_send(Conf::num_of_thread);
        vector<concurrent_unordered_set<int> > item_id_to_send(Conf::num_of_thread);

        value_type global_rmse, global_obj, global_loss, global_reg, test_loss, test_rmse;
        compute_train_statistics(global_rmse, global_loss, global_reg);
        global_obj = global_loss + Conf::lambda * global_reg;
        predict(test_loss, test_rmse);

        if (Data::machine_id == 0) {
            cout << "=====================================================" << endl;
            cout << "epoch: " << 0 << endl;
            cout << "elapsed time: " << 0 << " secs" << endl;
            cout << "current training RMSE: " << global_rmse << endl;
            cout << "current training obj: " << global_obj << endl;
            cout << "current training LOSS: " << global_loss << endl;
            cout << "current training reg: " << global_reg << endl;
            cout << "current test LOSS: " << test_loss << endl;
            cout << "current test RMSE: " << test_rmse << endl;
            cout << "=====================================================" << endl;
        }

        Monitor timer;
        value_type total_time = 0;
        value_type prev_obj = std::numeric_limits<value_type>::max();

        for (int epoch = 1; epoch <= Conf::max_iter; epoch++) {

            timer.start();

            // randomly shuffle ratings
            shuffle_ratings();

            for (int step = 0; step < Conf::folder; step++) {

                MPI_Barrier(MPI_COMM_WORLD);

                // computation phase
                computation_phase(step, size_of_folder, user_id_to_send, item_id_to_send);
                MPI_Barrier(MPI_COMM_WORLD);

                // global message transmission phase
                communicate_phase(user_id_to_send, item_id_to_send);

                // barrier synchronization
                MPI_Barrier(MPI_COMM_WORLD);
            }

            compute_train_statistics(global_rmse, global_loss, global_reg);

            global_obj = global_loss + Conf::lambda * global_reg;

            if(global_obj <= prev_obj){
                Conf::learning_rate *= 1.05;
            } else {
                Conf::learning_rate *= 0.5;
            }
            prev_obj = global_obj;

            timer.stop();
            total_time+=timer.getElapsedTime();

            predict(test_loss, test_rmse);

            if (Data::machine_id == 0) {
                cout << "=====================================================" << endl;
                cout << "epoch: " << epoch << endl;
                cout << "elapsed time: " << total_time << " secs" << endl;
                cout << "current training RMSE: " << global_rmse << endl;
                cout << "current training obj: " << global_obj << endl;
                cout << "current training LOSS: " << global_loss << endl;
                cout << "current training reg: " << global_reg << endl;
                cout << "current test LOSS: " << test_loss << endl;
                cout << "current test RMSE: " << test_rmse << endl;
                cout << "=====================================================" << endl;
            }

        }

        delete[] size_of_folder;

        cout << "total training time: " << total_time << " secs" << endl;

    }

};


#endif //ASGD_H
