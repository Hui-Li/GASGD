#ifndef OBJECT_H
#define OBJECT_H

#include "../util/Base.h"

class Object {
public:

    int id;
    unordered_set<int> assigned_partition_id;
    int master_id;
    std::mutex *mt = nullptr;

    Object(){
        master_id = -1;
        mt = new std::mutex();
    };

    Object(int id) {
        this->id = id;
        Object();
    }

    ~Object() {
        delete mt;
    }

    // https://stackoverflow.com/questions/9320533/boost-pointer-to-a-mutex-will-that-work-boostmutex-and-stdvector-noncopy

    Object(const Object &other) : id(other.id), master_id(other.master_id), mt(new std::mutex) {}

    Object& operator=(const Object &other) {
        id = other.id;
        master_id = other.master_id;
        assigned_partition_id = other.assigned_partition_id;
    }

    bool operator==(const Object& b) const {
        return this->id==b.id;
    }

};

#endif //OBJECT_H
