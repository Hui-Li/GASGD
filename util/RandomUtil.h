#ifndef RANDOMUTIL_H
#define RANDOMUTIL_H

#include "Base.h"
#include <random>

namespace RandomUtil {

    std::random_device rd;
    std::mt19937 gen(rd());

    void init_seed(){
        srand(time(NULL));
    }

    inline value_type uniform_real() {
        std::uniform_real_distribution<value_type> distribution(-0.5, 0.5);
        return distribution(gen);
    }
};
#endif //RANDOMUTIL_H
