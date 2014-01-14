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

#include <cstdlib>

#include <libAnnotation/xmlhelpers.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////
///
///
///   XML helpers
///
///
//////////////////////////////////////////////////////////////////////////////

string getElementDataString(const string& name, const string& doc)
{
  
  //cout << "  getElementDataInt()"<< endl;
  //cout << "    TagName: "<< name <<" Doc: " << doc << endl;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  unsigned start = tagStart.length();
  unsigned end = doc.find(tagEnd, start);
  
  string dataString = doc.substr(start,end-start);
  //cout << "    Data: " << dataString << "-> " << atoi(dataString.c_str()) << endl;
  
  return dataString;
  
}

int getElementDataInt(const string& name, const string& doc)
{
  
  //cout << "  getElementDataInt()"<< endl;
  //cout << "    TagName: "<< name <<" Doc: " << doc << endl;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  unsigned start = tagStart.length();
  unsigned end = doc.find(tagEnd, start);
  
  string dataString = doc.substr(start,end-start);
  //cout << "    Data: " << dataString << "-> " << atoi(dataString.c_str()) << endl;
  
  return atoi(dataString.c_str());
  
}

float getElementDataFloat(const string& name, const string& doc)
{
  
  //cout << "  getElementDataInt()"<< endl;
  //cout << "    TagName: "<< name <<" Doc: " << doc << endl;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  unsigned start = tagStart.length();
  unsigned end = doc.find(tagEnd, start);
  
  string dataString = doc.substr(start,end-start);
  //cout << "    Data: " << dataString << "-> " << atof(dataString.c_str()) << endl;
  
  return atof(dataString.c_str());
  
}

vector<string> getElements(const string& name, const string& doc)
{
  
  //cout << "  getElements(" << name << ")"<< endl;
  //cout << "    Doc: " << doc << endl << "    End Doc" << endl;
  
  string::size_type start, end, pos=0;
  string elementString;
  vector<string> elementStrings;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  while (pos!=string::npos)
  {
    
    pos = doc.find(tagStart, pos);
    if (pos==string::npos) break;
    start = pos;
    
    pos = doc.find(tagEnd, pos+1);
    if (pos==string::npos) break;
    end = pos;
    
    elementString = doc.substr(start,end-start+tagEnd.length());
    elementStrings.push_back(elementString);
    
    //cout << "    New Element found: " << endl;
    //cout << elementString << endl;
    //cout << "    End new Element!" << endl;
    
  }  
  
  return elementStrings;
  
}


