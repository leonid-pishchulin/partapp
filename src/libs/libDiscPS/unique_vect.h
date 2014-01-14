#ifndef _UNIQUE_VECT_HPP_
#define _UNIQUE_VECT_HPP_

#include <vector>
#include <algorithm>
#include <iostream>

#include <libBoostMath/boost_math.h>

struct CompareDoubleVector {

  CompareDoubleVector() {}

  bool operator()(const boost_math::double_vector &v1, const boost_math::double_vector &v2) {
    assert(v1.size() == v2.size());

    for (uint idx = 0; idx < v1.size(); ++idx) {
      if (v1(idx) < v2(idx))
        return true;
      else if (v1(idx) > v2(idx))
        return false;

    }

    return false;
  }

};

bool is_equal_vect(const boost_math::double_vector &v1, const boost_math::double_vector &v2);
void print_vect(const std::vector<boost_math::double_vector> &V);
void check_get_unique_elements(const std::vector<boost_math::double_vector> &V, const std::vector<boost_math::double_vector> &V2);
void get_unique_elements(const std::vector<boost_math::double_vector> &_V, std::vector<boost_math::double_vector> &V2);



#endif
