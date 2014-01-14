#include "unique_vect.h"

  
bool is_equal_vect(const boost_math::double_vector &v1, const boost_math::double_vector &v2) {
  assert(v1.size() == v2.size());

  for (uint idx = 0; idx < v1.size(); ++idx)
    if (v1(idx) != v2(idx))
      return false;

  return true;
}

void print_vect(const std::vector<boost_math::double_vector> &V)
{
  for (uint idx = 0; idx < V.size(); ++idx) {
    for (uint idx2 = 0; idx2 < V[idx].size(); ++idx2)
      std::cout << V[idx](idx2) << " ";

    std::cout << std::endl;
  }
}

void check_get_unique_elements(const std::vector<boost_math::double_vector> &V, 
                               const std::vector<boost_math::double_vector> &V2)
{
  std::cout << "check_get_unique_elements" << std::endl;

  for (uint idx = 0; idx < V2.size(); ++idx) {

    bool found = false;

    for (uint idx2 = 0; idx2 < V.size(); ++idx2) {
      if (is_equal_vect(V2[idx], V[idx2])) {
        found = true;
        break;
      }
    }
    
    assert(found);
  }
}


void get_unique_elements(const std::vector<boost_math::double_vector> &_V, std::vector<boost_math::double_vector> &V2)
{
  std::vector<boost_math::double_vector> V(_V);

  CompareDoubleVector comp;
  std::sort(V.begin(), V.end(), comp);

  assert(V2.size() == 0);

  uint idx = 0;
  while (idx < V.size()) {

    V2.push_back(V[idx]);

    ++idx;

    while (idx < V.size() && is_equal_vect(V[idx], V[idx - 1])) 
      ++idx;
  }
  
  check_get_unique_elements(_V, V2);
}


