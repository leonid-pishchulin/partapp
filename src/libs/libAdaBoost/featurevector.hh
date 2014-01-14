/*********************************************************************/
/*                                                                   */
/* FILE         featurevector.hh                                     */
/* AUTHORS      Bastian Leibe                                        */
/* EMAIL        leibe@inf.ethz.ch                                    */
/*                                                                   */
/* CONTENT      Define a general feature vector class derived from a */
/*              std::vector<float> and provide some basic functions.      */
/*                                                                   */
/* BEGIN        Wed Jul 24 2002                                      */
/* LAST CHANGE  Wed Jul 24 2002                                      */
/*                                                                   */
/*********************************************************************/

#ifndef LEIBE_FEATUREVECTOR_HH
#define LEIBE_FEATUREVECTOR_HH

#ifdef _USE_PERSONAL_NAMESPACES
//namespace Leibe {
#endif

//#define NDEBUG

/****************/
/*   Includes   */
/****************/
#include <vector>
#include <string>
#include <cassert>
#include <fstream>


/*************************/
/*   Class Definitions   */
/*************************/

/*===================================================================*/
/*                         Class FeatureVector                       */
/*===================================================================*/
/* Define a general feature vector class */
class FeatureVector
{
    friend std::ostream& operator<<(std::ostream& output, FeatureVector& fv);
    friend std::istream& operator>>(std::istream& input, FeatureVector& fv);
public:
    FeatureVector();
    FeatureVector( int nDims );
    template<typename T>FeatureVector( std::vector<T> data, int targetClass = -1, float instanceWeight = 1.0f );
    template<typename T> FeatureVector( T* data, int dataLen, int targetClass = -1, float instanceWeight = 1.0f );
    FeatureVector( const FeatureVector &other );
     ~FeatureVector();

    FeatureVector& operator=( FeatureVector other );
    FeatureVector& operator=( std::vector<float> other);
    FeatureVector& operator=( std::vector<double> other);

protected:
     void  copyFromOther( const FeatureVector &other );

     void  initBins();

     int   calcTotalNumBins() const;

public:
  /*******************************/
  /*   Content Access Functions  */
  /*******************************/
   bool  isValid() const;

  int   numDims() const     { assert( m_bSizeDefined ); return m_nDims; }

   void  setNumDims  ( int nDims );

  inline float  at ( int x ) const          { return m_vBins[idx(x)]; }
   float& at ( int x )                { return m_vBins[idx(x)]; }

  inline void  setValue( int x, float val )  { m_vBins[idx(x)] = val; }
  inline void  setTargetClass (int targetClass) {m_targetClass = targetClass;}
  inline void  setInstanceWeight (float instanceWeight) {m_instanceWeight = instanceWeight;}

   void  setData( std::vector<float> data, int targetClass = -1, float instanceWeight = 1.0f );
   void  setData( std::vector<double> data, int targetClass = -1, float instanceWeight = 1.0f);
  template<typename T> void  setData( T* data, int dataLen, int targetClass = -1, float instanceWeight = 1.0f);

   const std::vector<float>& getData() const { return m_vBins; }
   const float& getData(int index) const {return m_vBins[index];}

   int getTargetClass() const 	{return m_targetClass;}
   float getInstanceWeight() const {return m_instanceWeight;}

  /*Low level access*/
   std::vector<float>& getData() {return m_vBins; }
  //Call corretDim if you directly manipulated the vector
  inline void correctDim() {m_nDims = m_vBins.size(); m_nTotalNumBins = m_nDims; m_bSizeDefined = true;}

   void  clear();

   void  print();
   void  printContent();

public:
  /******************************/
  /*   FeatureVector File I/O   */
  /******************************/
   bool  save( std::string filename );
   bool  load( std::string filename, bool verbose=false );

   void  writeHeader( std::ofstream &ofile );
   void  writeData( std::ofstream &ofile ) const;
   bool  readHeader( std::ifstream &ifile, bool &isAscii,
														bool verbose=false );
   bool  readData( std::ifstream &ifile, bool isAscii=true );

public:
  /**********************/
  /*   Vector Algebra   */
  /**********************/
  /*--------------------------*/
  /* Vector-Vector operations */
  /*--------------------------*/
   void  addVector   ( const FeatureVector &other );
   void  subVector   ( const FeatureVector &other );

  FeatureVector& operator+=( const FeatureVector &other );
  FeatureVector& operator-=( const FeatureVector &other );

  friend FeatureVector operator+( const FeatureVector &a,
                                  const FeatureVector &b );
  friend FeatureVector operator-( const FeatureVector &a,
                                  const FeatureVector &b );

  float         dot  ( const FeatureVector &other );
  FeatureVector cross( const FeatureVector& other );

  /*--------------------------*/
  /* Vector-Scalar operations */
  /*--------------------------*/
   void  multFactor  ( float factor );
  FeatureVector& operator+=( float x );
  FeatureVector& operator-=( float x );
  FeatureVector& operator*=( float x );
  FeatureVector& operator/=( float x );

  friend FeatureVector operator+( const FeatureVector& a, float x );
  friend FeatureVector operator-( const FeatureVector& a, float x );
  friend FeatureVector operator*( const FeatureVector& a, float x );
  friend FeatureVector operator/( const FeatureVector& a, float x );

public:
  /**********************************/
  /*   FeatureVector Manipulation   */
  /**********************************/
   void  normalizeVector ( float newSum );
   void  normalizeEntries( std::vector<float> vBinFactors );
  void subtractMean();
  void normalizeEnergy();
  void normalizeEnergy2();
  void normalizeZeroMeanUnitVar();
  void normalizeZeroMeanUnitStdDev();
  void normalizeZeroMeanUnitStdDev2();

public:
  /********************************/
  /*   FeatureVector Statistics   */
  /********************************/
  float getSum() const;
  void  getMinMax( float &min, float &max ) const;

  friend void computeFeatureStatistics( std::vector<FeatureVector> vFeatureVectors,
																				std::vector<float> &vMeans,
																				std::vector<float> &vVariances );

public:
  /*************************************/
  /*   Histogram Comparison Measures   */
  /*************************************/
  float compSSD             ( const FeatureVector &other ) const;
  float compCorrelation     ( const FeatureVector &other ) const;
  float compIntersection    ( const FeatureVector &other,
                              bool bNormalizeResult=true ) const;
  float compBhattacharyya   ( const FeatureVector &other,
                              bool bNormalizeInputs=true ) const;

protected:
     int   idx( int x ) const;

    int            m_nDims;
    bool           m_bSizeDefined;

    int            m_nTotalNumBins;
    std::vector<float>  m_vBins;     	// contains all bins, access via idx function
    int			   m_targetClass;	// target class for multi-class problems
    float		   m_instanceWeight;
};


/****************************/
/*   Associated Functions   */
/****************************/

bool  readKeyWord( char* ifile, std::string KeyWords );
bool readKeyWord( std::ifstream &ifile, std::string KeyWords );
std::vector<std::string> extractWords( std::string WordList );

void computeFeatureStatistics( std::vector<FeatureVector> vFeatureVectors,
                               std::vector<float> &vMeans,
                               std::vector<float> &vVariances );

std::ostream& operator<<(std::ostream& output, FeatureVector& fv);
std::istream& operator>>(std::istream& input, FeatureVector& fv);


/****************************/
/*      Inline Methods      */
/****************************/

template<typename T>FeatureVector::FeatureVector( std::vector<T> data, int targetClass, float instanceWeight  )
{
  m_nDims = data.size();
  m_bSizeDefined = true;

  m_vBins.clear();
  m_vBins.reserve(data.size());
  for( int i=0; i<(int)data.size(); i++ )
    m_vBins.push_back( static_cast<float>(data[i]) );
  m_nTotalNumBins = data.size();
  m_targetClass = targetClass;
  m_instanceWeight = instanceWeight;
}

template<typename T>FeatureVector::FeatureVector( T* data, int dataLen, int targetClass, float instanceWeight )
{
  setData(data, dataLen, targetClass, instanceWeight);
}

template<typename T> void FeatureVector::setData( T* data, int dataLen, int targetClass, float instanceWeight) {
	m_nDims = dataLen;
	m_bSizeDefined = true;

	m_vBins.clear();
	m_vBins.reserve(dataLen);
	for( int i=0; i < dataLen; i++ )
		m_vBins.push_back( static_cast<float>(data[i]) );
	m_nTotalNumBins = dataLen;
	m_targetClass = targetClass;
	m_instanceWeight = instanceWeight;
}

#ifdef _USE_PERSONAL_NAMESPACES
//}
#endif

#endif // LEIBE_FEATUREVECTOR_HH
