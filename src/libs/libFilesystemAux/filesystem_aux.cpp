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

#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>

#include <cassert>
#include <iostream>

#include "filesystem_aux.h"

namespace filesys {
  bool check_file(QString qsFilename) 
  {
    QFileInfo fi(qsFilename);
    return fi.exists() && fi.isFile();
  }
  
  bool rename_file(QString qsFilename, QString qsNewFilename){
    QFile file(qsFilename);
    file.open(QIODevice::ReadOnly | QIODevice::Text);
    /*
    std::cout << "rename" << std::endl;
    getchar();
    QDir qdir = QDir(qsDir);
    bool res = qdir.rename(qsFilename, qsNewFilename);
    */
    bool res = file.rename(qsNewFilename);
    file.close();
    return res;
    
  }

  bool remove_file(QString qsFilename){
    return QFile::remove(qsFilename);
  }
  
  bool read_file_txt(QString qsFilename, QString &qsMsg)
  {
    QFile file(qsFilename);
    if (not file.open(QIODevice::ReadOnly | QIODevice::Text))
      return false;
    
    QTextStream in(&file);
    
    while(not in.atEnd()) 
      qsMsg = in.readLine();

    file.close();

    return true;
  }
  
  bool write_file_txt(QString qsFilename, QString qsMsg)
  {
    QFile file(qsFilename);
    if (not file.open(QIODevice::WriteOnly | QIODevice::Text))
      return false;
    QTextStream out(&file);
    out << qsMsg << "\n";
    file.close(); 
    
    return true;
  }

  bool check_dir(QString qsDir)
  {
    assert(!qsDir.isEmpty());
    QDir dir(qsDir);
    return dir.exists();
  }

  bool create_dir(QString qsDir) 
  {
    QFileInfo fi(qsDir);
  
    /* absolute path is necessary since mkpath works weird otherwise (2 dirs are created instead of one) */
    if (fi.isRelative())
      qsDir = fi.absoluteFilePath();

    if (check_dir(qsDir))
      return true;

    QDir dir(qsDir);
    //cout << "creating " << qsDir.toStdString() << endl;
    return dir.mkpath(qsDir);

  }

  void split_filename(QString qsFilename, QString &qsPath, QString &qsName)
  {
    QFileInfo fi(qsFilename);

    qsPath = fi.path();
    qsName = fi.fileName();
  }

  /**
     split full path into components

     qsPath does not end with '/'
     qsExt does not start with '.'
   */

  void split_filename_ext(QString qsFilename, QString &qsPath, QString &qsBaseName, QString &qsExt)
  {
    QFileInfo fi(qsFilename);

    qsPath = fi.path();
    qsBaseName = fi.baseName();
    qsExt = fi.completeSuffix();
  }

  QString add_suffix(QString qsFilename, QString qsSuffix)
  {
    QString qsPath, qsBaseName, qsExt;
    split_filename_ext(qsFilename, qsPath, qsBaseName, qsExt);
    return qsPath + "/" + qsBaseName + qsSuffix + "." + qsExt;
  }

  void make_absolute(QString &qsFilename) {
    QFileInfo fi(qsFilename);
    if (fi.isRelative())
      qsFilename = fi.absoluteFilePath();
  }

  bool copy_file(QString oldFilePath, QString newFilePath)
  {
    //same file, no need to copy
    if(oldFilePath.compare(newFilePath) == 0)
      return true;

    //load both files
    QFile oldFile(oldFilePath);
    QFile newFile(newFilePath);
    bool openOld = oldFile.open( QIODevice::ReadOnly );
    bool openNew = newFile.open( QIODevice::WriteOnly );

    //if either file fails to open bail
    if(!openOld || !openNew) { return false; }

    //copy contents
    uint BUFFER_SIZE = 16000;
    char* buffer = new char[BUFFER_SIZE];
    while(!oldFile.atEnd())
      {
	long len = oldFile.read( buffer, BUFFER_SIZE );
	newFile.write( buffer, len );
      }

    //deallocate buffer
    delete[] buffer;
    buffer = NULL;
    return true;
  }
  
  bool getFileNames(QString qsDir, QString qsFilter, std::vector<QString> &fileNames){
    
    QDir export_folder(qsDir);
    export_folder.setNameFilters(QStringList()<< qsFilter);
    QStringList qsFileList = export_folder.entryList();
        
    if (qsFileList.size() == 0)
      return false;
    for (int idx = 0; idx < qsFileList.size(); ++idx)
      fileNames.push_back(qsFileList.at(idx));
    return true;
  }

}
