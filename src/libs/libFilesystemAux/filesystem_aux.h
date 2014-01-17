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

#ifndef _FILESUSTEM_AUX_H_
#define _FILESUSTEM_AUX_H_

#include <QString>
#include <vector>

namespace filesys {

  bool check_file(QString qsFilename);
  bool write_file_txt(QString qsFilename, QString qsMsg);
  bool read_file_txt(QString qsFilename, QString &qsMsg);
  bool remove_file(QString qsFilename);
  bool rename_file(QString qsFilename, QString qsNewFilename);
  bool copy_file(QString oldFilePath, QString newFilePath);

  bool check_dir(QString qsDir);
  bool create_dir(QString qsDir);
  void split_filename(QString qsFilename, QString &qsPath, QString &qsName);
  void split_filename_ext(QString qsFilename, QString &qsPath, QString &qsBaseName, QString &qsExt);
  bool getFileNames(QString qsDir, QString qsFilter, std::vector<QString> &fileNames);
  
  inline QString get_basename(QString qsFilename) {
    QString qsPath, qsBaseName, qsExt;
    split_filename_ext(qsFilename, qsPath, qsBaseName, qsExt);
    return qsBaseName;
  }

  void make_absolute(QString &qsFilename);


  QString add_suffix(QString qsFilename, QString qsSuffix);
}

#endif
