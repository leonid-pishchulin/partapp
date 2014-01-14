/*
 * AdaBoost code (copyright by Christian Wojek)
 *
 * This code is granted free of charge for non-commercial research and
 * education purposes. However you must obtain a license from the author
 * to use it for commercial purposes.

 * The software and derivatives of the software must not be distributed
 * without prior permission of the author.
 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

/*
 *
 *  Implementation of Discrete Adaboost as described in Viola & Jones 
 * in Robost Real-Time Face Detection, IJCV 2004
 * 
 */ 

#include <iostream>
#include <fstream>

#include <vector>
#include "BoostingData.h"

#include <libMisc/misc.hpp>

struct DecisionStump {
  int f;						// Feature number
  float p;					// Sign for < vs. > in the stump node
  float theta;				// Threshold
  float alpha;				// Weak learner's weight
	
  float J_wse;				// Error measure
  float value;				// Keeps the value during inference	
};

struct singleDimensionDataOrder{
  int id;
  float value;
	
  singleDimensionDataOrder(int i, float v) {id = i; value = v;};
};

inline bool operator<(const singleDimensionDataOrder &a,const singleDimensionDataOrder &b) {
  return (a.value < b.value);	
}

class AdaBoostClassifier{
  friend std::ostream& operator<<(std::ostream& output, AdaBoostClassifier& ab);
  friend std::istream& operator>>(std::istream& input, AdaBoostClassifier& ab);
 public:
  AdaBoostClassifier();	
  virtual void trainClassifier(Data* TrainingData, unsigned int M, std::ostream *outStream = 0);
  virtual ~AdaBoostClassifier() {};
  virtual void printEnsemble();
  virtual std::vector<DecisionStump> getAllStumps();	

  virtual float evaluateFeaturePoint(const FeatureVector &v, bool normalize = false, unsigned int rounds = 0) const;
  template<typename T> float evaluateFeaturePoint(const T* v, int dataLen, bool normalize = false, unsigned int rounds = 0) const;
  inline float evaluateFeaturePoint(const std::vector<float>& v, bool normalize, unsigned int rounds = 0) const;

  virtual std::vector<float> evaluateStumps(const FeatureVector &v, unsigned int rounds = 0) const;
  virtual std::vector<float> getAlpha() const;
  virtual void setAlpha(std::vector<float> alpha);

  virtual void loadClassifier(std::string filename);
  virtual void saveClassifier(std::string filename);
  virtual void performRounds(unsigned int numRounds, Data* TrainingData = 0, std::ostream *outStream = 0);
  virtual bool validModel() {return stumps.size() > 0;};
  virtual void clearTrainingWeights() {weights.clear(); weights.reserve(0);}
  /*
   * Sets the update mode
   * 0 = as proposed by Viola and Jones
   * 1 = exponential as proposed by Hastie et. al
   */
  virtual void setUpdateMode(unsigned int mode) { updateMode = mode; };

  virtual unsigned int getNumberOfRounds() {
    return stumps.size();
  }

  inline int getD() {return m_D;}
	
 private:
  DecisionStump getBestWeakLearner(int d);
  std::vector<DecisionStump> stumps;
		
  //Data points and their weights
  int m_D;

  Data* m_v;
  std::vector<float> weights;
  float sum_alpha;
  unsigned int updateMode;
};

/*
 * Some functions to store our classiers conviently in streams
 */
std::ostream& operator<<(std::ostream& output, AdaBoostClassifier& jb);
std::istream& operator>>(std::istream& input, AdaBoostClassifier& jb);

/*
 * Inlined method for fast evaluation
 */
template<typename T> float AdaBoostClassifier::evaluateFeaturePoint(const T* v, int dataLen, bool normalize, unsigned int rounds) const {
  assert(false);
  unsigned int activeStumps = stumps.size();	
  if(rounds != 0)
    activeStumps = rounds;		
	
  // Evaluate all features
  float s = 0;
  for(unsigned int i = 0; i < activeStumps; i++) {
    const float &val = v[stumps[i].f];	
    //stumps[i].value = (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : 0;
    //stumps[i].value = (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : -stumps[i].alpha;
    //s += stumps[i].value;
    s += (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : -stumps[i].alpha;
  }
	
  if (normalize)
    s /= sum_alpha;
	
  return s;	
}

float AdaBoostClassifier::evaluateFeaturePoint(const std::vector<float>& v, bool normalize, unsigned int rounds) const {

  //assert(v.size() == (uint)m_D);
  
  if (v.size() != (uint)m_D) {
    std::cout << "v.size(): " << v.size() << std::endl;
    std::cout << "m_D: " << m_D << std::endl;
    assert(v.size() == (uint)m_D);
  }
  
  unsigned int activeStumps = stumps.size();	
  if(rounds != 0)
    activeStumps = rounds;		
	
  // Evaluate all features
  float s = 0;
  for(unsigned int i = 0; i < activeStumps; i++) {
    const float &val = v[stumps[i].f];	
    //stumps[i].value = (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : 0;
    //stumps[i].value = (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : -stumps[i].alpha;
    //s += stumps[i].value;
    s += (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : -stumps[i].alpha;
    //std::cout << i << " " << s << " " << stumps[i].alpha << std::endl; //getchar();
  }
	
  if (normalize)
    s /= sum_alpha;
	
  return s;	
}

#endif /*ADABOOST_H_*/
