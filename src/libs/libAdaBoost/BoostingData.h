/*
 * AdaBoost code (copyright by Christian Wojek)
 *
 * This code is granted free of charge for non-commercial research and
 * education purposes. However you must obtain a license from the author
 * to use it for commercial purposes.

 * The software and derivatives of the software must not be distributed
 * without prior permission of the author.
 */

#ifndef BOOSTINGDATA_H_
#define BOOSTINGDATA_H_

//#include <libFeatures/featurevector.hh>
#include "featurevector.hh"

#include <vector>



class Data{
public:
	Data(int D, int C);
	virtual ~Data();
	
	/*
	 *  Be aware that addresses might change if further elements are added to &s!!!!
	 * So only construct dataSet and then run Boosting training immediately
	 * 
	 */
	virtual void addSample(FeatureVector* s);
	virtual void addSampleVector(std::vector<FeatureVector> &s);
	
	virtual std::vector<FeatureVector*>* getAllSamples(); 
	virtual std::vector<FeatureVector*>* getSamplesOfClass(int c);
	
	inline int getC() {return m_C;};
	inline int getN() {return m_N;};
	inline int getD() {return m_D;};
	
	void clear();
	
private:
	int m_D;				//Overall number of features D
	int m_N;				//Overall number of training samples N
	int m_C;				//Overall number of classes C
	std::vector<FeatureVector*> m_samples;
	std::vector< std::vector<FeatureVector*> > m_samples_of_class; 	
};


#endif /*BOOSTINGDATA_H_*/
