/** 
    This file is part of the implementation of the human pose estimation model as described in the paper:
    
    Leonid Pishchulin, Micha Andriluka, Peter Gehler and Bernt Schiele
    Strong Appearance and Expressive Spatial Models for Human Pose Estimation
    IEEE International Conference on Computer Vision (ICCV'13), Sydney, Australia, December 2013

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  
*/

#ifndef ANNORECT_H
#define ANNORECT_H

#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

#include <cstdlib>

enum FixDimType{ FIX_OBJWIDTH = 0, FIX_OBJHEIGHT = 1};

struct AnnoPoint {
  AnnoPoint(): id(-1),x(0),y(0),is_visible(true) {}
  AnnoPoint(int _id, int _x, int _y):id(_id), x(_x), y(_y), is_visible(true) {}
  AnnoPoint(int _id, int _x, int _y, bool _is_visible):id(_id), x(_x), y(_y), is_visible(_is_visible) {}

  int id;
  int x;
  int y;

  bool is_visible;
};


class AnnoRect
{
 public:
  AnnoRect():m_x1(-1),m_x2(-1),m_y1(-1),m_y2(-1),
    m_dScore(-1),m_nSilhouetteID(-1),m_dScale(-1),
    m_nObjPosX(-1), m_nObjPosY(-1), 
    m_nObjectId(-1), m_dMotionPhase(-1) {};
    
    AnnoRect(int x1, int y1, int x2, int y2, double score=-1, int sil=-1, float scale = -1):m_x1(x1),m_x2(x2),m_y1(y1),m_y2(y2),
    m_dScore(score),m_nSilhouetteID(sil), m_dScale(scale), 
      m_nObjPosX(-1), m_nObjPosY(-1), 
      m_nObjectId(-1), m_dMotionPhase(-1){};

    //explicit AnnoRect(const AnnoRect& other);
  ~AnnoRect(){};

  // MA: made public 
  int m_x1, m_x2, m_y1, m_y2;

protected:
  double m_dScore;
  int m_nSilhouetteID;

  std::vector<int> m_vArticulations;
  std::vector<int> m_vViewPoints;

  float m_dScale;

  int m_nObjPosX;
  int m_nObjPosY;
public:
  std::vector<AnnoPoint> m_vAnnoPoints; 
  int m_nObjectId;
  float m_dMotionPhase;

  void setCoords(int x1, int y1, int x2, int y2) {m_x1=x1; m_y1=y1; m_x2=x2; m_y2=y2;};
  void setX1(int x1) {m_x1=x1;};
  void setX2(int x2) {m_x2=x2;};
  void setY1(int y1) {m_y1=y1;};
  void setY2(int y2) {m_y2=y2;};
  int x1() const {return m_x1;};
  int x2() const {return m_x2;};
  int y1() const {return m_y1;};
  int y2() const {return m_y2;};
  int centerX() const {return (m_x1 + m_x2) / 2;};
  int centerY() const {return (m_y1 + m_y2) / 2;};
  
  int left() const {return std::min(m_x1, m_x2);};
  int right() const {return std::max(m_x1, m_x2);};
  int top() const {return std::min(m_y1, m_y2);};
  int bottom() const {return std::max(m_y1, m_y2);};  
  
  int w() const {return abs(m_x2-m_x1);};
  int h() const {return abs(m_y2-m_y1);};
  
  int getObjPosX() const {return m_nObjPosX;};
  int getObjPosY() const {return m_nObjPosY;};

  void setObjPos(int x, int y) {m_nObjPosX = x; m_nObjPosY = y;}

  void setScale(float newScale) {m_dScale = newScale;}; 
  float scale() const {return m_dScale;};
  float logScale() const {return log(m_dScale);};
  
  void rescale(double factor) {m_x1=int(m_x1*factor); m_y1=int(m_y1*factor); m_x2=int(m_x2*factor); m_y2=int(m_y2*factor); m_dScale *= factor;};
  
  void setScore(double val) {m_dScore=val;};
  double score() const  {return m_dScore;};

  void setSilhouetteID(int id) {m_nSilhouetteID=id;};
  int silhouetteID()const  {return m_nSilhouetteID;};

  void clearArticulations() {m_vArticulations.clear();};
  void addArticulation(int id) {m_vArticulations.push_back(id);};
  unsigned noArticulations() const {return m_vArticulations.size();};
  int articulation(int i) const {return m_vArticulations[i];};
  int& articulation(int i) {return m_vArticulations[i];};

  void clearViewPoints() {m_vViewPoints.clear();};
  void addViewPoint(int id)    {m_vViewPoints.push_back(id);};
  unsigned noViewPoints() const {return m_vViewPoints.size();};
  int viewPoint(int i) const {return m_vViewPoints[i];};
  int& viewPoint(int i) {return m_vViewPoints[i];};

  const AnnoPoint& getAnnoPoint(int nAnnoPointIdx) const;
  void addAnnoPoint(AnnoPoint ap) {m_vAnnoPoints.push_back(ap);};
  void clearAnnoPoints() {m_vAnnoPoints.clear();};

  std::vector<AnnoPoint>& getAnnoPoints() {return m_vAnnoPoints;};
  const std::vector<AnnoPoint>& getAnnoPoints() const {return m_vAnnoPoints;};

  const AnnoPoint* get_annopoint_by_id(int id) const;

  void sortCoords();

  //--- IO --//
  void printXML() const;
  void printIDL() const;

  void writeXML(std::ofstream&) const;
  void writeIDL(std::ofstream&) const;

  void parseXML(const std::string&);
  void parseIDL(const std::string&);

  //--- Compare ---//
  double compCover   ( const AnnoRect& other) const;
  double compRelDist ( const AnnoRect& other) const;
  double compRelDist ( const AnnoRect& other, float dAspectRatio, FixDimType eFixObjDim=FIX_OBJHEIGHT ) const;
  bool   isMatching  ( const AnnoRect& other, double dTDist, double dTCover, double dTOverlap) const;
  //bool   isMatching  ( const AnnoRect& other, double dTDist, double dTCover, double dTOverlap,
  //                                            float dAspectRatio, FixDimType eFixObjDim=FIX_OBJHEIGHT );
  
};

struct compAnnoRectByScore
{
	bool operator()(const AnnoRect& r1, const AnnoRect& r2)
	{
		return r1.score()>r2.score();
	}
};



#endif
