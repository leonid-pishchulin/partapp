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

#include <libAnnotation/annorect.h>
#include <libAnnotation/xmlhelpers.h>
#include <cmath>
#include <assert.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////
///
///
///   AnnoRect
///
///
//////////////////////////////////////////////////////////////////////////////


const AnnoPoint& AnnoRect::getAnnoPoint(int nAnnoPointIdx) const
{
  assert(nAnnoPointIdx >= 0 && (unsigned)nAnnoPointIdx < m_vAnnoPoints.size());
  assert(nAnnoPointIdx + 1 == m_vAnnoPoints[nAnnoPointIdx].id || 
         nAnnoPointIdx == m_vAnnoPoints[nAnnoPointIdx].id);
  return m_vAnnoPoints[nAnnoPointIdx];
}

void AnnoRect::parseXML(const string& rectString)
{
  //cerr << "AnnoRect::parse()"<< endl;
  vector<string> tmp;
  tmp = getElements("x1", rectString);
  if (tmp.size()>0)
    m_x1 = getElementDataInt("x1", tmp[0]);
  tmp = getElements("x2", rectString);
  if (tmp.size()>0)
    m_x2 = getElementDataInt("x2", tmp[0]);
  tmp = getElements("y1", rectString);
  if (tmp.size()>0)
    m_y1 = getElementDataInt("y1", tmp[0]);
  tmp = getElements("y2", rectString);
  if (tmp.size()>0)
    m_y2 = getElementDataInt("y2", tmp[0]);

  tmp = getElements("score", rectString);
  if (tmp.size()>0) {
    m_dScore = getElementDataFloat("score", tmp[0]);
  }

  tmp = getElements("scale", rectString);
  if (tmp.size()>0) {
    m_dScale = getElementDataFloat("scale", tmp[0]);
  }

  tmp = getElements("silhouette", rectString);
  if (tmp.size()>0)
  {
    tmp = getElements("id", tmp[0]);
    for(unsigned i=0; i<tmp.size(); i++)
      m_nSilhouetteID = getElementDataInt("id", tmp[0]);
  }

  tmp = getElements("articulation", rectString);
  if (tmp.size()>0)
  {
    tmp = getElements("id", tmp[0]);
    for(unsigned i=0; i<tmp.size(); i++)
      m_vArticulations.push_back(getElementDataInt("id", tmp[i]));
  }

  tmp = getElements("viewpoint", rectString);
  if (tmp.size()>0)
  {
    tmp = getElements("id", tmp[0]);
    for(unsigned i=0; i<tmp.size(); i++)
      m_vViewPoints.push_back(getElementDataInt("id", tmp[i]));
  }

  vector <string> tmp2;

  tmp = getElements("annopoints", rectString);
  if (tmp.size()>0) {
    tmp = getElements("point", tmp[0]);
    for(unsigned i=0; i<tmp.size(); i++) {
      AnnoPoint p;

      tmp2 = getElements("id", tmp[i]);
      if (tmp2.size() > 0)
        p.id = getElementDataInt("id", tmp2[0]);

      tmp2 = getElements("x", tmp[i]);
      if (tmp2.size() > 0) 
        p.x = getElementDataInt("x", tmp2[0]);

      tmp2 = getElements("y", tmp[i]);
      if (tmp2.size() > 0)
        p.y = getElementDataInt("y", tmp2[0]);

      tmp2 = getElements("is_visible", tmp[i]);
      if (tmp2.size() > 0)
        p.is_visible = static_cast<bool>(getElementDataInt("is_visible", tmp2[0]));


//       if (!(p.id > 0 && p.x > 0 && p.y > 0))
//         cout << p.id << " " << p.x << " " << p.y << endl;

//      assert(p.id >= 0 && p.x >= 0 && p.y >= 0);
      assert(p.id >= 0);

      m_vAnnoPoints.push_back(p);
    }
  }

  tmp = getElements("objpos", rectString);
  if (tmp.size() > 0) {
    tmp2 = getElements("x", tmp[0]);
    if (tmp2.size() > 0)
      m_nObjPosX = getElementDataInt("x", tmp2[0]);

    tmp2 = getElements("y", tmp[0]);
    if (tmp2.size() > 0)
      m_nObjPosY = getElementDataInt("y", tmp2[0]);
  }

  tmp = getElements("object_id", rectString);
  if (tmp.size()>0)
    m_nObjectId = getElementDataInt("object_id", tmp[0]);

  tmp = getElements("motion_phase", rectString);
  if (tmp.size()>0) 
    m_dMotionPhase = getElementDataFloat("motion_phase", tmp[0]);

  //printXML();

  if (m_x1 > m_x2) {
    int tmp = m_x2;
    m_x2 = m_x1;
    m_x1 = tmp;
  }

  if (m_y1 > m_y2) {
    int tmp = m_y2;
    m_y2 = m_y1;
    m_y1 = tmp;
  }
    
  assert(m_x1 <= m_x2 && m_y1 <= m_y2);
}

void AnnoRect::parseIDL(const string& rectString)
{
  string::size_type start=1, end;
  end = rectString.find(",", start);
  m_x1 = atoi(rectString.substr(start, end-start).c_str());
  //cout << m_x1 << endl;
  start=end+1;
  end = rectString.find(",", start);
  m_y1 = atoi(rectString.substr(start, end-start).c_str());
  //cout << m_y1 << endl;
  start=end+1;
  end = rectString.find(",", start);
  m_x2 = atoi(rectString.substr(start, end-start).c_str());
  //cout << m_x2 << endl;
  start=end+1;
  end = rectString.find("):", start);
  m_y2 = atoi(rectString.substr(start, end-start).c_str());
  //cout << m_y2 << endl;

  start = end+2;
  end = rectString.find("/", start);
  m_dScore = atof(rectString.substr(start, end-start).c_str());
  //cout << m_dScore << endl;

  if (end==string::npos)
    m_nSilhouetteID=-1;
  else
  {
    start = end+1;
    m_nSilhouetteID = atoi(rectString.substr(start,string::npos).c_str());
  }
  //cout << m_nSilhouetteID << endl;

  //printXML();
}

void AnnoRect::writeXML(ofstream& out) const
{
  out << "    <annorect>\n";
  out << "        <x1>" << m_x1 << "</x1>\n";
  out << "        <y1>" << m_y1 << "</y1>\n";
  out << "        <x2>" << m_x2 << "</x2>\n";
  out << "        <y2>" << m_y2 << "</y2>\n";
  if (m_dScore!=-1)
    out << "        <score>" << m_dScore << "</score>\n";
  if (m_nSilhouetteID!=-1)
  {
    out << "        <silhouette>\n";
    out << "            <id>"<< m_nSilhouetteID << "</id>\n";
    out << "        </silhouette>\n";
  }
  if (m_vArticulations.size()>0)
  {
    out << "        <articulation>\n";
    vector<int>::const_iterator it;
    for(it=m_vArticulations.begin(); it!=m_vArticulations.end(); it++)
      out << "            <id>"<< *it << "</id>\n";
    out << "        </articulation>\n";
  }
  if (m_vViewPoints.size()>0)
  {
    out << "        <viewpoint>\n";
    vector<int>::const_iterator it;
    for(it=m_vViewPoints.begin(); it!=m_vViewPoints.end(); it++)
      out << "            <id>"<< *it << "</id>\n";
    out << "        </viewpoint>\n";
  }

  if (m_vAnnoPoints.size() > 0) {
    out << "        <annopoints>\n";
    vector<AnnoPoint>::const_iterator it;
    for (it=m_vAnnoPoints.begin(); it!=m_vAnnoPoints.end(); it++) {
      out << "          <point>\n";
      out << "          <id>" << (*it).id << "</id>\n";
      out << "          <x>" << (*it).x << "</x>\n";
      out << "          <y>" << (*it).y << "</y>\n";
      out << "          <is_visible>" << (*it).is_visible << "</is_visible>\n";
      out << "          </point>\n";
    }
    out << "        </annopoints>\n";
  }

  if (m_nObjPosX > 0 && m_nObjPosY > 0) {
    out << "        <objpos>\n";
    out << "        <x>" << m_nObjPosX << "</x>\n";
    out << "        <y>" << m_nObjPosY << "</y>\n";
    out << "        </objpos>\n";
  }

  if (m_nObjectId >= 0) {
    out << "        <object_id>" << m_nObjectId << "</object_id>\n";
  }

  if (m_dMotionPhase >= 0) {
    out << "        <motion_phase>" << m_dMotionPhase << "</motion_phase>\n";
  }

  out << "    </annorect>\n";
}


void AnnoRect::printXML() const
{
  cout << "    <annorect>\n";
  cout << "        <x1>" << m_x1 << "</x1>\n";
  cout << "        <y1>" << m_y1 << "</y1>\n";
  cout << "        <x2>" << m_x2 << "</x2>\n";
  cout << "        <y2>" << m_y2 << "</y2>\n";
  if (m_dScore!=-1)
    cout << "        <score>" << m_dScore << "</score>\n";
  if (m_nSilhouetteID!=-1)
  {
    cout << "        <silhouette>\n";
    cout << "            <id>"<< m_nSilhouetteID << "</id>\n";
    cout << "        </silhouette>\n";
  }
  if (m_vArticulations.size()>0)
  {
    cout << "        <articulation>\n";
    vector<int>::const_iterator it;
    for(it=m_vArticulations.begin(); it!=m_vArticulations.end(); it++)
      cout << "            <id>"<< *it << "</id>\n";
    cout << "        </articulation>\n";
  }
  if (m_vViewPoints.size()>0)
  {
    cout << "        <viewpoint>\n";
    vector<int>::const_iterator it;
    for(it=m_vViewPoints.begin(); it!=m_vViewPoints.end(); it++)
      cout << "            <id>"<< *it << "</id>\n";
    cout << "        </viewpoint>\n";
  }

  if (m_nObjectId >= 0) {
    cout << "        <object_id>" << m_nObjectId << "</objpos>\n";
  }

  if (m_dMotionPhase >= 0) {
    cout << "        <motion_phase>" << m_dMotionPhase << "</motion_phase>\n";
  }

  cout << "    </annorect>\n";
}

void AnnoRect::writeIDL(ofstream& out) const
{
  if (m_nSilhouetteID==-1)
    out << "(" << m_x1 <<", " << m_y1 << ", " << m_x2 << ", " << m_y2 << "):" << m_dScore;
  else
    out << "(" << m_x1 <<", " << m_y1 << ", " << m_x2 << ", " << m_y2 << "):" << m_dScore <<"/" << m_nSilhouetteID;
}

void AnnoRect::printIDL() const
{
  if (m_nSilhouetteID==-1)
    cout << "(" << m_x1 <<", " << m_y1 << ", " << m_x2 << ", " << m_y2 << "):" << m_dScore;
  else
    cout << "(" << m_x1 <<", " << m_y1 << ", " << m_x2 << ", " << m_y2 << "):" << m_dScore <<"/" << m_nSilhouetteID;
}

void AnnoRect::sortCoords()
{
  int tmp;
  if (m_x1>m_x2)
  {
    tmp = m_x1;
    m_x1= m_x2;
    m_x2= tmp;
  }
  if (m_y1>m_y2)
  {
    tmp = m_y1;
    m_y1= m_y2;
    m_y2= tmp;
  }
}

double AnnoRect::compCover( const AnnoRect& other ) const
{
  AnnoRect r1 = *this;
  AnnoRect r2 = other;
  r1.sortCoords();
  r2.sortCoords();

  int nWidth  = r1.x2() - r1.x1();
  int nHeight = r1.y2() - r1.y1();
  int iWidth  = max(0,min(max(0,r2.x2()-r1.x1()),nWidth )-max(0,r2.x1()-r1.x1()));
  int iHeight = max(0,min(max(0,r2.y2()-r1.y1()),nHeight)-max(0,r2.y1()-r1.y1()));
  return ((double)iWidth * (double)iHeight)/((double)nWidth * (double)nHeight);
}

double AnnoRect::compRelDist( const AnnoRect& other, float dAspectRatio, FixDimType eFixObjDim ) const
{
  double dWidth, dHeight;

  switch( eFixObjDim )
  {
  case FIX_OBJWIDTH:
    dWidth  = m_x2 - m_x1;
    dHeight = dWidth / dAspectRatio;
    break;

  case FIX_OBJHEIGHT:
    dHeight = m_y2 - m_y1;
    dWidth  = dHeight * dAspectRatio;
    break;

  default:
    cerr << "Error in ImgDescrList::compRelDist(): "
    << "Unknown type for parameter ('which obj dimension to fix?'): "
    << eFixObjDim << "!" << endl;
    return -1.0;
  }

  double xdist = (double)(m_x1+m_x2-other.x1()-other.x2()) / dWidth;
  double ydist = (double)(m_y1+m_y2-other.y1()-other.y2()) / dHeight;
  return sqrt(xdist*xdist + ydist*ydist);
}


double AnnoRect::compRelDist( const AnnoRect& other ) const
{
  double dWidth  = m_x2 - m_x1;
  double dHeight = m_y2 - m_y1;
  double xdist   = (double)(m_x1 + m_x2 - other.x1() - other.x2()) / dWidth;
  double ydist   = (double)(m_y1 + m_y2 - other.y1() - other.y2()) / dHeight;
  return sqrt(xdist*xdist + ydist*ydist);
}


// bool AnnoRect::isMatching( const AnnoRect& other, double dTDist, double dTCover, double dTOverlap, float dAspectRatio, FixDimType eFixObjDim )
// {
//   double dWidth, dHeight;
//
//   switch( eFixObjDim ) {
//   case FIX_OBJWIDTH:
//     dWidth  = m_x2 - m_x1;
//     dHeight = dWidth / dAspectRatio;
//     break;
//
//   case FIX_OBJHEIGHT:
//     dHeight = m_y2 - m_y1;
//     dWidth  = dHeight * dAspectRatio;
//     break;
//
//   default:
//     cerr << "Error in ImgDescrList::compRelDist(): "
//          << "Unknown type for parameter ('which obj dimension to fix?'): "
//          << eFixObjDim << "!" << endl;
//     return -1.0;
//   }
//
//   double xdist = (double)(m_x1+m_x2-other.x1()-other.x2()) / dWidth;
//   double ydist = (double)(m_y1+m_y2-other.y1()-other.y2()) / dHeight;
//   return sqrt(xdist*xdist + ydist*ydist);
// }


bool AnnoRect::isMatching( const AnnoRect& other, double dTDist, double dTCover, double dTOverlap) const
{
  return ( (compRelDist(other) <= dTDist) &&
           (compCover(other) >= dTCover) &&
           (other.compCover(*this) >= dTOverlap) );
}

const AnnoPoint* AnnoRect::get_annopoint_by_id(int id) const
{

  for (uint ptidx = 0; ptidx < m_vAnnoPoints.size(); ++ptidx) {
    if (m_vAnnoPoints[ptidx].id == id) {
      return &(m_vAnnoPoints[ptidx]);
    }
  }

  return NULL;
}
