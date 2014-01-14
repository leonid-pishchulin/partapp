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

#include <cassert>
#include <algorithm>


#include <libAnnotation/annotation.h>
#include <libAnnotation/xmlhelpers.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////
///
///
///   Annotation
///
///
//////////////////////////////////////////////////////////////////////////////

// Annotation::Annotation(const Annotation& other)
// {
//   m_sPath = other.m_sPath;
//   m_sName = other.m_sName;
//   m_sDim =  other.m_sDim;
//   m_vRects.clear();
//   for (unsigned i =0; i<other.size(); i++)
//   {
//     this->addAnnoRect(other.annoRect(i));
//   }
// }


string convertRelativeToHome(const string &fname) 
{
  /** convert file path to be relative to home directory (if possible) */
  string strHome = getenv("HOME");
  int home_len = strHome.length();

  assert(home_len > 0);
  
  if ((int)fname.length() > home_len) 
    if (fname.substr(0, home_len) == strHome) {
      string strNewName = fname.substr(home_len);
      if (strNewName[0] != '/')
        strNewName = "/" + strNewName;

      strNewName = "~" + strNewName;

      //cout << "new image name: " << strNewName << endl;
      return strNewName;
    }

  return fname;
}

void Annotation::printXML() const
{
  cout << "<annotation>\n";
  cout << "    <image>\n";
  cout << "        <name>" << m_sName << "</name>\n";
  if (m_sPath.length()>0)
    cout << "        <path>" << m_sPath << "</path>\n";
  if (m_sDim.length()>0)
    cout << "        <dimensions>" << m_sDim << "</dimensions>\n";
  if (m_bIsTiffStream)
  	 cout << "        <frameNr>" << m_dFrameNr << "</frameNr>\n";
  cout << "    </image>\n";
  for(vector<AnnoRect>::const_iterator it=m_vRects.begin(); it!=m_vRects.end(); it++)
    it->printXML();
  cout << "</annotation>\n";
}

void Annotation::writeXML(ofstream& f, bool bSaveRelativeToHome) const
{
  f << "<annotation>\n";
  f << "    <image>\n";
  
  if (bSaveRelativeToHome)
    f << "        <name>" << convertRelativeToHome(m_sName) << "</name>\n";
  else
    f << "        <name>" << m_sName << "</name>\n";

  if (m_sPath.length()>0)
    f << "        <path>" << m_sPath << "</path>\n";

  if (m_sDim.length()>0)
    f << "        <dimensions>" << m_sDim << "</dimensions>\n";

  if (m_nImageWidth > 0 && m_nImageHeight > 0) {
    f << "        <image_width>" << m_nImageWidth << "</image_width>\n";
    f << "        <image_height>" << m_nImageHeight << "</image_height>\n";
  }
    
  if (m_bIsTiffStream)
    f  << "        <frameNr>" << m_dFrameNr << "</frameNr>\n";
  f << "    </image>\n";
  for(vector<AnnoRect>::const_iterator it=m_vRects.begin(); it!=m_vRects.end(); it++)
    it->writeXML(f);
  f << "</annotation>\n";
}

void Annotation::printIDL() const
{
  
  cout << "\"" << m_sName;
  if (m_bIsTiffStream)
  	cout << "@" << m_dFrameNr; 
  cout << "\"";
  if (m_vRects.size()>0)
    cout << ": ";
  
  for(vector<AnnoRect>::const_iterator it=m_vRects.begin(); it!=m_vRects.end(); it++)
  {
    it->printIDL();
    if (it+1==m_vRects.end())
      cout << ", ";
  }
}

void Annotation::writeIDL(ofstream& f, bool bSaveRelativeToHome) const
{
  if (bSaveRelativeToHome)
    f << "\"" << convertRelativeToHome(m_sName);
  else
    f << "\"" << m_sName;

  if (m_bIsTiffStream)
  	f << "@" << m_dFrameNr; 
  f << "\"";
  
  if (m_vRects.size()>0)
    f << ": ";
    
  for(vector<AnnoRect>::const_iterator it=m_vRects.begin(); it!=m_vRects.end(); it++)
  {
    it->writeIDL(f);
    if (it+1!=m_vRects.end())
      f << ", ";
  }
}

void Annotation::parseXML(const string& annoString)
{
  vector<string> tmp;

  //--- get image infor ---//
  vector<string> image = getElements("image", annoString);
  if (image.size()>0)
  {
    tmp = getElements("name", image[0]);
    if (tmp.size()>0)
      m_sName = getElementDataString("name", tmp[0]);

    tmp = getElements("path", image[0]);
    if (tmp.size()>0)
      m_sPath = getElementDataString("path", tmp[0]);

    tmp = getElements("dimensions", image[0]);
    if (tmp.size()>0)
      m_sDim = getElementDataString("dimensions", tmp[0]);

    tmp = getElements("image_width", image[0]);
    if (tmp.size() > 0)
      m_nImageWidth = getElementDataInt("image_width", tmp[0]);

    tmp = getElements("image_height", image[0]);
    if (tmp.size() > 0)
      m_nImageHeight = getElementDataInt("image_height", tmp[0]);

    tmp = getElements("frameNr", image[0]);
    if (tmp.size()>0) {
      std::string fn = getElementDataString("frameNr", tmp[0]);
      m_dFrameNr = atoi(fn.c_str());
      m_bIsTiffStream = true;
    }
  }

  //--- chop in AnnoRects ---//
  vector<string> rectStrings = getElements("annorect", annoString);
  vector<string>::const_iterator it;
  for(it=rectStrings.begin(); it!=rectStrings.end(); it++)
  {
    AnnoRect r;
    r.parseXML(*it);
    m_vRects.push_back(r);
  }
}

void Annotation::parseIDL(const string& annoString, const string& sPath)
{
  vector<string> tmp;

  //--- get image name ---//
  string::size_type pos=0, start, end;
  pos = annoString.find("@");
  if (pos == string::npos) {
  	m_bIsTiffStream = false;
  	pos = annoString.find("\":");
  	
  	if (pos==string::npos)
    	pos = annoString.rfind("\"");
  	
  	m_sName = annoString.substr(1,pos-1);  
  } else {
  	m_bIsTiffStream = true;
  	m_sName = annoString.substr(1,pos-1);
  	
  	if (m_sName[m_sName.size() - 1] == '/')
  		m_sName = m_sName.substr(0, m_sName.size() - 1);
  	
	string::size_type epos = annoString.find("\":", pos);  	
  	if (epos == string::npos)
    	epos = annoString.rfind("\"");
  	
  	std::string frame = annoString.substr(pos + 1,  epos - 1);
  	m_dFrameNr = atoi(frame.c_str());
  	
  	pos = epos; 	
  }  	
  
  assert(m_sName.length()>0);
  if (m_sName[0]!='/' && m_sName[0]!='~')
    m_sName = sPath+m_sName;

  //cout << "Reading Image: "<< m_sName << endl;
  
  //--- chop in AnnoRects ---//
  string rectString;
  vector<string> rectStrings;
  while(pos!=string::npos)
  {
    start = annoString.find("(", pos);
    if (start==string::npos)
      break;
    pos = annoString.find(")", start);
    end = annoString.find(",", pos+1);

    rectString = annoString.substr(start,end-start);
    rectStrings.push_back(rectString);
    //cout << "Start: " << start << " End: " << end << endl;
    //cout << "RectString: " << rectString << endl;
  }
  
  vector<string>::const_iterator it;
  for(it=rectStrings.begin(); it!=rectStrings.end(); it++)
  {
    AnnoRect r;
    r.parseIDL(*it);
    m_vRects.push_back(r);
  }
}

void Annotation::sortByScore()
{
  std::stable_sort(m_vRects.begin(), m_vRects.end(), compAnnoRectByScore());
}
