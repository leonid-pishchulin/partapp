/*
 * AdaBoost code (copyright by Christian Wojek)
 *
 * This code is granted free of charge for non-commercial research and
 * education purposes. However you must obtain a license from the author
 * to use it for commercial purposes.

 * The software and derivatives of the software must not be distributed
 * without prior permission of the author.
 */

#include <libAdaBoost/BoostingData.h>
#include "featurevector.hh"

Data::Data(int D, int C) : m_samples_of_class(C + 1) {
	m_D = D;
	m_C = C;	
	m_N = 0;
}

void Data::clear() {
	m_samples.clear();
	m_N = 0;
}

Data::~Data() {
	clear();	
}

void Data::addSample(FeatureVector* s) {
	assert(s->numDims() == m_D);		
	m_samples.push_back(s);
	
	if (s->getTargetClass() == -1)
		m_samples_of_class[m_C].push_back(s);
	else {
		assert(s->getTargetClass() < m_C);
		m_samples_of_class[s->getTargetClass()].push_back(s);
	}	
	m_N++;	
} 

void Data::addSampleVector(std::vector<FeatureVector> &s) {	
	for(unsigned int i = 0; i < s.size(); i++) 
		addSample(&s[i]);
}

std::vector<FeatureVector*>* Data::getAllSamples() {
	return &m_samples;
}

std::vector<FeatureVector*>* Data::getSamplesOfClass(int c) {
	assert(c >= -1 && c < m_C);
	if (c==-1)
		return &(m_samples_of_class[m_C]);
	else
		return &(m_samples_of_class[c]);
}
