#ifndef CONF_H
#define CONF_H

#include "Base.h"

class Conf {
public:

    static int k; // rank
    static value_type learning_rate;
    static value_type lambda;
    static int folder;
    static int max_iter;
    static int num_of_machine;
    static int num_of_thread;
    static string meta_path;
    static string train_data_path;
    static string test_data_path;
    static value_type g_period;
    static int step;
    static int partition;

    static bool create(int argc, char **argv) {

        po::options_description desc("Allowed options");
        desc.add_options()
                ("help", "produce help message")
                ("k", po::value<int>(&(Conf::k))->default_value(50), "dimensionality of latent vector")
                ("step", po::value<int>(&(Conf::step))->default_value(1), "0: constant step size, 1: bold driver")
                ("lr", po::value<value_type>(&(Conf::learning_rate))->default_value(0.002), "learning rate")
                ("lambda", po::value<value_type>(&(Conf::lambda))->default_value(0.05), "regularization weight")
                ("folder", po::value<int>(&(Conf::folder))->default_value(1), "how many times the machines communicate during each epoch")
                ("max_iter", po::value<int>(&(Conf::max_iter))->default_value(30), "how many iterations to be performed by the sgd algorithm")
                ("node", po::value<int>(&(Conf::num_of_machine))->default_value(1), "number of machines")
                ("thread", po::value<int>(&(Conf::num_of_thread))->default_value(4), "number of thread per machine")
                ("partition", po::value<int>(&(Conf::partition))->default_value(2), "0: Greedy, 1: Item Partition, 2: User Partition")
                ("g_period", po::value<value_type>(&(Conf::g_period))->default_value(0.01), "synchronization window for graph partitioning")
                ("path", po::value<string>(&(Conf::meta_path)), "file path of data");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return false;
        } else {
            return true;
        }
    }
};

int Conf::k = 0;
value_type Conf::learning_rate = 0;
value_type Conf::lambda = 0;
int Conf::folder = 0;
int Conf::max_iter = 0;
int Conf::num_of_machine = 0;
int Conf::num_of_thread = 0;
int Conf::step = 0;
int Conf::partition = 0;

value_type Conf::g_period = 0.0;
string Conf::meta_path = "";
string Conf::train_data_path = "";
string Conf::test_data_path = "";

#endif //CONF_H
