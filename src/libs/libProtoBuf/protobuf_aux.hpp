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

#ifndef _PROTOBUF_AUX_H_
#define _PROTOBUF_AUX_H_

#include <QString>
#include <fstream>
#include <iostream>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <libFilesystemAux/filesystem_aux.h>

/**
   load and save protocol buffer messages in text format
*/

template <typename PB_CLASS>
bool write_message_binary(QString qsFilename, PB_CLASS &msg)
{
  using namespace std;
  fstream out(qsFilename.toStdString().c_str(), ios::out | ios::trunc | ios::binary);
  bool bRes = msg.SerializeToOstream(&out);
  assert(bRes);
  return bRes;
}

template <typename PB_CLASS>
bool parse_message_binary(QString qsFilename, PB_CLASS &msg)
{
  using namespace std;
  assert(filesys::check_file(qsFilename));

  fstream in(qsFilename.toStdString().c_str(), ios::in | ios::binary);
  bool bRes = msg.ParseFromIstream(&in);
  assert(bRes);
  return bRes;
}

template <typename PB_CLASS>
void parse_message_from_text_file(QString qsFilename, PB_CLASS &msg)
{
  std::fstream fstr_in(qsFilename.toAscii().data(), std::ios::in);
  google::protobuf::io::ZeroCopyInputStream *zc_in = new google::protobuf::io::IstreamInputStream(&fstr_in);

  if (!filesys::check_file(qsFilename)) {
    std::cout << "file not found: " << qsFilename.toStdString() << std::endl;
    assert(false);
  }

  bool bRes = google::protobuf::TextFormat::Parse(zc_in, &msg);  
  assert(bRes && "error while parsing protobuf file");

  delete zc_in;
}

template <typename PB_CLASS>
void print_message_to_text_file(QString qsFilename, const PB_CLASS &msg)
{
  std::fstream fstr_out(qsFilename.toAscii().data(), std::ios::out | std::ios::trunc);
  google::protobuf::io::ZeroCopyOutputStream *zc_out = new google::protobuf::io::OstreamOutputStream(&fstr_out);

  bool bRes = google::protobuf::TextFormat::Print(msg, zc_out);
  assert(bRes && "error while saving protobuf file");

  delete zc_out;  
}


#endif
