#ifndef FILEUTIL_H
#define FILEUTIL_H

#include "Base.h"
#include "../struct/Rating.h"

using std::fstream;
using std::ifstream;
using std::ofstream;

namespace FileUtil {

    void readMetaData(string meta_path, string &train_data_path, string &test_data_path, int &user_num, int &item_num, int4rating &train_rating_num,
                      int4rating &test_rating_num) {
        ifstream data_file(meta_path + "/meta");

        string line;
        std::getline(data_file, line);
        vector<string> par;
        boost::split(par, line, boost::is_any_of(" "));

        user_num = strtoull(par[0].c_str(), nullptr, 0);
        item_num = strtoull(par[1].c_str(), nullptr, 0);

        std::getline(data_file, line);
        boost::split(par, line, boost::is_any_of(" "));
        train_rating_num = strtoull(par[0].c_str(), nullptr, 0);
        train_data_path = meta_path + "/" + par[1];

        std::getline(data_file, line);
        boost::split(par, line, boost::is_any_of(" "));
        test_rating_num = strtoull(par[0].c_str(), nullptr, 0);
        test_data_path = meta_path + "/" + par[1];

        data_file.close();
    }

    // data is divided among nodes/threads
    bool readDataLocally(string data_path, vector<Rating> &ratings, const int4rating min_row_index,
                         const int4rating max_row_index, const int4rating total_rating_num, const int row_num) {
        ifstream data_file(data_path);

        ull begin_skip = min_row_index * size_of_double;

        // scores
        data_file.seekg(begin_skip, std::ios_base::cur);

        ull size = (max_row_index - min_row_index) * size_of_double;
        int *score_rows = scalable_allocator<int>().allocate(size);
        if (!data_file.read(reinterpret_cast<char *>(score_rows), size)) {
            cerr << "Error in reading rating values from file!" << endl;
            scalable_allocator<int>().deallocate(score_rows, size);
            return false;
        }

        double *score_ptr = reinterpret_cast<double *>(score_rows);

        // row_index
        begin_skip = total_rating_num * size_of_double;
        data_file.seekg(begin_skip, std::ios_base::beg);
        ull size2 = size_of_int * (row_num + 1);
        int *row_nums = scalable_allocator<int>().allocate(size2);
        if (!data_file.read(reinterpret_cast<char *>(row_nums), size2)) {
            cerr << "Error in reading row index from file!" << endl;
            scalable_allocator<int>().deallocate(score_rows, size);
            scalable_allocator<int>().deallocate(row_nums, size2);
            return false;
        }

        // col_index
        begin_skip = min_row_index * size_of_int;
        data_file.seekg(begin_skip, std::ios_base::cur);

        ull size3 = size_of_int * (max_row_index - min_row_index);

        int *col_indices = scalable_allocator<int>().allocate(size3);
        if (!data_file.read(reinterpret_cast<char *>(col_indices), size3)) {
            cerr << "Error in reading col index from file!" << endl;
            scalable_allocator<int>().deallocate(score_rows, size);
            scalable_allocator<int>().deallocate(row_nums, size2);
            scalable_allocator<int>().deallocate(col_indices, size3);
            return false;
        }

        // format data
        ratings.resize(max_row_index - min_row_index);
        int index = 0;
        int global_id_start = 0;
        int rating_num = 0;
        bool finish = false;
        for (int row_index = 1; row_index < row_num + 1; row_index++) {

            // accumulation includes row of row_index = row_nums[row_index];
            if (row_nums[row_index] < min_row_index) {
                continue;
            } else if (row_nums[row_index - 1] < min_row_index && row_nums[row_index] >= min_row_index) {
                global_id_start = min_row_index;
                rating_num =
                        ((max_row_index - row_nums[row_index - 1]) < (row_nums[row_index] - row_nums[row_index - 1])) ? (max_row_index - row_nums[row_index - 1]) : (row_nums[row_index] - row_nums[row_index - 1]);
            } else if (row_nums[row_index - 1] >= min_row_index && row_nums[row_index - 1] < max_row_index) {
                global_id_start = row_nums[row_index - 1];
                rating_num =  ((max_row_index - row_nums[row_index - 1])<(row_nums[row_index] - row_nums[row_index - 1]))?(max_row_index - row_nums[row_index - 1]) : (row_nums[row_index] - row_nums[row_index - 1]);
            } else {
                cerr << "Logical error!" << endl;
                return false;
            }

            for (int offset = 0; offset < rating_num; offset++) {
                ratings[index].global_id = global_id_start + offset;
                ratings[index].user_id = row_index - 1;
                ratings[index].item_id = col_indices[index];
                ratings[index].score = score_ptr[index];

                index++;
                if (index >= max_row_index - min_row_index) {
                    finish = true;
                    break;
                }
            }

            if (finish) {
                break;
            }
        }

        data_file.close();

        scalable_allocator<int>().deallocate(score_rows, size);
        scalable_allocator<int>().deallocate(row_nums, size2);
        scalable_allocator<int>().deallocate(col_indices, size3);

        return true;
    }

}
#endif //FILEUTIL_H