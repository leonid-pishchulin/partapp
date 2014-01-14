/** 
    This file is part of the implementation of the human pose estimation model as described in the paper:
    
    L. Pishchulin, M. Andriluka, P. Gehler and B. Schiele
    Strong Appearance and Expressive Spatial Models for Human Pose Estimation
    IEEE Conference on Computer Vision and Pattern Recognition (ICCV'13), Sydney, Australia, December 2013

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  
*/

#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <vector>
#include <iostream>
#include <fstream>

#include <libAnnotation/annorect.h>


class Annotation
{
public:
  Annotation() : m_nImageWidth(0), m_nImageHeight(0), m_bIsTiffStream(false) {};
  Annotation(const std::string& name): m_nImageWidth(0), m_nImageHeight(0), m_bIsTiffStream(false)  {m_sName=name;};
  //Annotation(const Annotation& other);
  ~Annotation(){};

  /** MA: moved to public because it is super annoying to use accessors all the time */

  std::vector<AnnoRect> m_vRects;

private:
  // new 
  int m_nImageWidth;
  int m_nImageHeight;

  std::string m_sPath;
  std::string m_sName;
  std::string m_sDim;
  int m_dFrameNr;
  bool m_bIsTiffStream;
  
  // synthetic/real data 
  // 0 - real, 1 - reshaped
  int m_dataType;
  
public:
  int imageWidth() {return m_nImageWidth;}
  int imageHeight() {return m_nImageHeight;}
  void setImageWidth(int nImageWidth) { m_nImageWidth = nImageWidth; }
  void setImageHeight(int nImageHeight) { m_nImageHeight = nImageHeight; }

  unsigned size() const {return m_vRects.size();};
  void clear() {m_vRects.clear();};

  void sortByScore();

  const AnnoRect& annoRect(unsigned i) const {return m_vRects[i];};
  AnnoRect& annoRect(unsigned i) {return m_vRects[i];};

  const AnnoRect& operator[]  (unsigned i) const {return m_vRects[i];};
  AnnoRect& operator[] (unsigned i) {return m_vRects[i];};
  
  void addAnnoRect(const AnnoRect& rect) {m_vRects.push_back(rect);};
  void removeAnnoRect(unsigned pos) {m_vRects.erase(m_vRects.begin()+pos);};

  const std::string& imageName() const {return m_sName;};  
  void setImageName(const std::string& name) {m_sName=name;};
  const std::string& imagePath() const {return m_sPath;};
  void setPath(const std::string& path) {m_sPath=path;};
  
  std::string fileName() {
  	std::string::size_type pos = m_sName.rfind("/");  	
  	if (pos != std::string::npos)
  		return m_sName.substr(pos+1);
  	else
  		return std::string("");
  }
    
  const std::string& imageDim() const {return m_sDim;};
  void setDim(const std::string& dim) {m_sDim=dim;};
  
  //Stream extensions
  int frameNr() const {return m_dFrameNr;}
  void setFrameNr(const int frameNr) {m_dFrameNr = frameNr; m_bIsTiffStream = true;}
  bool isStream() const {return m_bIsTiffStream;}
  
  //--- IO ---//
  void printXML() const;
  void printIDL() const;
  
  void writeXML(std::ofstream&, bool bSaveRelativeToHome) const;
  void writeIDL(std::ofstream&, bool bSaveRelativeToHome) const;

  void parseXML(const std::string&);
  void parseIDL(const std::string& annostring, const std::string& sPath="");
  
  void setDataType(const int dataType){m_dataType = dataType;};
  const int dataType() const {return m_dataType;} 
};


#endif
