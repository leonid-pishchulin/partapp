/*
 * AdaBoost code (copyright by Christian Wojek)
 *
 * This code is granted free of charge for non-commercial research and
 * education purposes. However you must obtain a license from the author
 * to use it for commercial purposes.

 * The software and derivatives of the software must not be distributed
 * without prior permission of the author.
 */

#include <libAdaBoost/AdaBoost.h>
#include <iostream>
#include <fstream> 
#include <math.h>
#include <iomanip>
#include <algorithm>

#define BIGINT	100000

using namespace std;

void AdaBoostClassifier::trainClassifier(Data* TrainingData, unsigned int M, std::ostream *outStream) {
  
  	m_v = TrainingData;	

	vector<FeatureVector*>* samples = m_v->getAllSamples();

        assert(samples->size() > 0);
        m_D = (*samples)[0]->numDims();

	if (m_v->getC() != 2) {
		if (outStream)
			(*outStream)  << "ERROR: Discrete AdaBoost only supports two class classification problems!" << endl;
	}
	
	if (outStream) {
		(*outStream) << "===========================================================" << endl;
		(*outStream) << "Starting to learn an ensemble classifier..." << endl;
		(*outStream) << "Boosting rounds: " << M << endl;
		(*outStream) << "Number of classes: " << m_v->getC() << endl;
		(*outStream) << "Data dimensionality: " << m_v->getD() << endl;
		for (int i = 0; i < m_v->getC(); i++)
			(*outStream) << "# Training instances for class " << i << ": " << m_v->getSamplesOfClass(i)->size() << endl;
		(*outStream) << "===========================================================" << endl;
	}
		
	//Initialize weights 
	weights.clear();
	weights.resize(m_v->getN());
	
	int m = m_v->getSamplesOfClass(0)->size();
	int l = m_v->getSamplesOfClass(1)->size();

	for(int i = 0; i < m_v->getN(); i++) {
		if ( (*samples)[i]->getTargetClass() == 0)
			weights[i] = 1.0f / 2 / m;	
		else
			weights[i] = 1.0f / 2 / l;
	}
	
	//Initialize ensemble
	stumps.clear();
		
	performRounds(M, 0, outStream);
}

void AdaBoostClassifier::performRounds(unsigned int M, Data* TrainingData,std::ostream *outStream) {
	if(TrainingData)
		m_v = TrainingData;
	
	if (!m_v || weights.size() == 0) {
		cout << "No training data available or invalid weights!" << endl;
		return;
	}
		
	for(unsigned int m = 0; m < M; m++) {				
		if (outStream)
			(*outStream) << "Performing boosting round number: " << m << endl;
		
		//Normalize the weights
		float sum = 0;		
		for(unsigned int i = 0; i < weights.size(); i++) {
			sum += weights[i];
		}
		for(unsigned int i = 0; i < weights.size(); i++) {
			weights[i] /= sum;
			//cout << "weight " << i << " : " << weights[i] << " value: " <<  (*m_v->getAllSamples())[i]->getData(1) << endl;
		}
		
		DecisionStump addedStump;		
		for(int d = 0; d < m_v->getD(); d++) {			
			DecisionStump bestStumpForDimension = getBestWeakLearner(d);
			
			//cout << "d: " << d << " " << bestStumpForDimension.J_wse << endl;
			
			//If we do better than with previous stumps keep stump for dimension d
			if (bestStumpForDimension.J_wse < addedStump.J_wse || d==0) {
				addedStump = bestStumpForDimension;			
			}			
		}
				
				
		//update weights[i]
		vector<FeatureVector*>* samples = m_v->getAllSamples();
		
		float beta_t = addedStump.J_wse / (1 - addedStump.J_wse); 
				
		for(int i = 0; i < m_v->getN(); i++) {
			
			const float h = (*samples)[i]->getData(addedStump.f) * addedStump.p < addedStump.theta * addedStump.p ? 1 : 0;
			const float e_i = (*samples)[i]->getTargetClass() == h ? 0 : 1; 
			
			if (updateMode == 0) 				
				if (e_i == 0)				
					weights[i] = weights[i] * beta_t;
			
			if(updateMode == 1)
				weights[i] = weights[i] * expf(addedStump.alpha * e_i);
			
		}	
		
		//if all training samples are classified correctly we can stop
		if (addedStump.J_wse == 0) {
			addedStump.alpha = BIGINT;
			stumps.push_back(addedStump);
			break;
		} else {
			//add the stump
			stumps.push_back(addedStump);
		}
						
	}	
	
	sum_alpha = 0;
	for(unsigned int i = 0; i < stumps.size(); i++)
		sum_alpha += stumps[i].alpha;
}

DecisionStump AdaBoostClassifier::getBestWeakLearner(int d) {
	DecisionStump r;
	r.J_wse = 1;
	
	const vector<FeatureVector*>* samples = m_v->getAllSamples();
	 
	vector<singleDimensionDataOrder> data;
	data.reserve(m_v->getN());
	for(int i = 0; i < m_v->getN(); i++) {
		data.push_back(singleDimensionDataOrder(i, (*samples)[i]->getData(d) ));
	}
	sort(data.begin(), data.end());
	
	float S_p = 0;
	float S_m = 0;
	float T_p = 0;
	float T_m = 0;
	
	//Get overall positive and negative weight
	for(unsigned int i = 0; i < data.size(); i++) {
		//cout << (*samples)[data[i].id]->getData(d) << " " << data[i].id << " " << (*samples)[data[i].id]->getTargetClass() << " weight:" << weights[i] << endl;
		if ((*samples)[data[i].id]->getTargetClass() == 1) 
			T_p += weights[data[i].id];
		else
			T_m += weights[data[i].id];
	}

	float rememberLower = data[0].value;
	for(unsigned int i = 0; i < data.size(); i++) {
		
		//Cumulate weights lower than thresholds
		if(i > 0) {
			if ((*samples)[data[i-1].id]->getTargetClass() == 1) 
				S_p += weights[data[i-1].id];
			else
				S_m += weights[data[i-1].id];

		
			if (data[i].value == data[i-1].value)
				continue;
			else
				rememberLower = data[i-1].value;
		}
			
		// " < " case		
		if (i == 0 || S_m + T_p - S_p < r.J_wse) {
			r.f = d;
			r.p = 1;			
			r.theta = (data[i].value + rememberLower) / 2;
			r.J_wse =  S_m + T_p - S_p;
			r.value = 0;
			r.alpha = logf((1 - r.J_wse)/ r.J_wse);			
		}
	
			// " > " case		
		if (S_p + T_m - S_m < r.J_wse) {
			r.f = d;
			r.p = -1;	
			r.theta = (data[i].value + rememberLower) / 2;						
			r.J_wse = S_p + T_m - S_m;
			r.value = 0;
			r.alpha = logf((1 - r.J_wse)/ r.J_wse);													
		}
		
		
		
	}
		
	return r;		
}

AdaBoostClassifier::AdaBoostClassifier(){
        m_D = -1;
	m_v = 0;
	updateMode = 0;
}

float AdaBoostClassifier::evaluateFeaturePoint(const FeatureVector &v, bool normalize, unsigned int rounds) const {
	unsigned int activeStumps = stumps.size();	
	if(rounds != 0)
	  activeStumps = rounds;		

	// Evaluate all features
	float s = 0;
	for(unsigned int i = 0; i < activeStumps; i++) {
		const float &val = v.at(stumps[i].f);	
		//stumps[i].value = (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : 0;
                //stumps[i].value = (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : -stumps[i].alpha;
		//s += stumps[i].value;
                s += (val * stumps[i].p  < stumps[i].theta * stumps[i].p) ? stumps[i].alpha : -stumps[i].alpha;
	}
	
	if (normalize)
	  s /= sum_alpha;
	
	return s;	
}

vector<float> AdaBoostClassifier::evaluateStumps(const FeatureVector &v, unsigned int rounds) const {
  unsigned int activeStumps = stumps.size();	
	if(rounds != 0)
		activeStumps = rounds;		
	
	vector<float> evalStumps(activeStumps);
	// Evaluate all stumps
	float s = 0;
	for(unsigned int i = 0; i < activeStumps; i++) {
	  const float &val = v.at(stumps[i].f);	
	  if (val * stumps[i].p  < stumps[i].theta * stumps[i].p)
	    evalStumps[i] = 1.0;///sum_alpha;
	  else
	    evalStumps[i] = -1.0;///sum_alpha;
	}
	return evalStumps;	
}

void AdaBoostClassifier::setAlpha(std::vector<float> alpha){
  unsigned int activeStumps = stumps.size();	
  
  assert(activeStumps == alpha.size());
  
  for(unsigned int i = 0; i < activeStumps; i++)
    stumps[i].alpha = alpha[i];
  
  sum_alpha = 0;
  for(unsigned int i = 0; i < activeStumps; i++){
    sum_alpha += (stumps[i].alpha >= 0) ? stumps[i].alpha : -stumps[i].alpha;
  }
}

vector<float> AdaBoostClassifier::getAlpha() const {

  unsigned int activeStumps = stumps.size();	
  vector<float> alpha(activeStumps);
  for(unsigned int i = 0; i < activeStumps; i++)
    alpha[i] = stumps[i].alpha;
  return alpha;
}

void AdaBoostClassifier::printEnsemble() {
	for(unsigned int c = 0; c < stumps.size(); c++) {
		cout << "Feature number: " << stumps[c].f << endl;
		cout << "Theta: " << stumps[c].theta << endl;
		cout << "p: " << stumps[c].p << endl;
		cout << "alpha: " << stumps[c].alpha << endl;
		cout << "J_wse: " << stumps[c].J_wse << endl;
		cout << "---------------------------------------------" << endl;			
	}	
	cout << "================================================================" << endl;
}

vector<DecisionStump> AdaBoostClassifier::getAllStumps() {
	return stumps;
}

ostream& operator<<(ostream& output, AdaBoostClassifier& jb) {
	try {		          
                output << jb.m_D << endl;
                output << setprecision(15) << jb.stumps.size() << "\t" << endl;
		
		for(unsigned int i = 0; i < jb.stumps.size(); i++) {
			output << jb.stumps[i].f << "\t" << jb.stumps[i].p << "\t" << jb.stumps[i].alpha << "\t" << jb.stumps[i].J_wse << "\t" << jb.stumps[i].theta << "\t" << endl;			
		}
		
	} catch(iostream::failure) {
		cerr << "Could not write to stream...";
		throw;		
	} 
	return output;
}

istream& operator>>(istream& input, AdaBoostClassifier& jb) {
	//Reset Data to make sure training data and classifier match
	jb.m_v = 0;
	jb.stumps.clear();
	
	try {				
                input >> jb.m_D;
          
		//Restore number of rounds, i.e. number of stumps
		unsigned int ns;
		input >> ns;
		jb.stumps.resize(ns);	
		
		jb.sum_alpha = 0;
		//Restore stumps
		for(unsigned int i = 0; i < ns; i++) {
			DecisionStump r;			
			input >> r.f >> r.p >> r.alpha >> r.J_wse >> r.theta;			
			jb.stumps[i] = r;
			jb.sum_alpha += r.alpha;
		}
		
	} catch(iostream::failure) {
		cerr << "Could not read from stream..." << endl;
		throw;		
	} 
	return input;
}

void AdaBoostClassifier::loadClassifier(std::string filename) {
	ifstream f(filename.c_str());
	f >> (*this);
}

void AdaBoostClassifier::saveClassifier(std::string filename) {
	if (!stumps.size()) return;
	
	ofstream f(filename.c_str());
	f << (*this);
}
