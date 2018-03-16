#ifndef RATING_H
#define RATING_H

class Rating {
public:
    int4rating global_id;
    int user_id;
    int item_id;
    value_type score;

    Rating(){};

    Rating(int4rating global_id, int user_id, int item_id, value_type score) {
        this->global_id = global_id;
        this->user_id = user_id;
        this->item_id = item_id;
        this->score = score;
    }

    Rating(int user_id, int item_id, value_type score) {
        this->user_id = user_id;
        this->item_id = item_id;
        this->score = score;
    }

    bool operator==(const Rating& b) const {
        return this->global_id == b.global_id;
    }
};

#endif //RATING_H
