/*
* "Shape Context" descriptor code (by Krystian Mikolajczyk <http://personal.ee.surrey.ac.uk/Personal/K.Mikolajczyk> ) based on SIFT (by David Lowe)
*
* This code is granted free of charge for non-commercial research and
* education purposes. However you must obtain a license from the author
* to use it for commercial purposes.

* The software and derivatives of the software must not be distributed
* without prior permission of the author.
*
*  If you use this code you should cite the following paper:
*  K. Mikolajczyk, C. Schmid,  A performance evaluation of local descriptors. In PAMI 27(10):1615-1630
*/

// Definition de la classe ImageContent
#ifndef _kmaimageContent_h_
#define _kmaimageContent_h_

#include <cstdlib>
#include <cstring>

namespace kma {
  class ImageContent {
   
  private :
    unsigned int x_size,y_size;
    unsigned int tsize;
    void writePGM(const char *nom,unsigned char *buff, const char* comments);
    void write(const char *nom, const char* comments);
    void initFloat(unsigned int ,unsigned int);	   
    void initUChar(unsigned int ,unsigned int);	   
    void init3Float(unsigned int ,unsigned int);	   
    void init3UChar(unsigned int ,unsigned int);	   
    int buftype;
  
  public :
    float **fel;
    float **felr;
    float **felg;
    float **felb;
    char *filename;
    unsigned char **bel;
    unsigned char **belr;
    unsigned char **belg;
    unsigned char **belb;
    ImageContent(void){};   
    ImageContent(const char *);	   
    ImageContent(ImageContent *im);	   
    ImageContent(unsigned int y_size_in,unsigned int x_size_in){initFloat( y_size_in, x_size_in);};	   
    ImageContent(int y_size_in ,int x_size_in){initFloat((unsigned int)y_size_in,(unsigned int)x_size_in);};	   
    ImageContent(unsigned int ,unsigned int, const char *);	   
    ImageContent(unsigned int ,unsigned int, const char *, float);	   
    ImageContent(unsigned int y_size_in, unsigned int x_size_in, float val){initFloat( y_size_in, x_size_in);set(val);};	   

    ~ImageContent();

    inline unsigned int x() const { return x_size;}
    inline unsigned int y() const { return y_size;}
    inline unsigned int size() const { return tsize;}
    int getType() const { return buftype;}
    const char* name(){return filename;}
    void write(const char *nom);
    void writePNG(const char* name);
    void writeR(const char *nom);
    void writeG(const char *nom);
    void writeB(const char *nom);
    void RGB2xyY();
    void RGB2lbrg();
    void RGB2rgb();
    void float2char();
    void char2float();
    void flipH();
    void flipV();
    void toGRAY();
    void set(float);
    void set(ImageContent*);
    void set(const char *name){strcpy(filename,name);}
    void normalize(float min_in, float max_in);
    void normalize();
    void scale(ImageContent *im_in, float scalex, float scaley);
    float getValue(float x, float y);
    void interpolate(ImageContent *sface, float m_x, float m_y, 
		     float scalex, float scaley, float angle);
    void interpolate(ImageContent *im_in, float m_x, float m_y, float vec0x, float vec0y,
		     float vec1x, float vec1y);
    void crop(ImageContent *img, int x, int y);
  };
  /*! \brief returns the format of the given png file
   * 0 - not a png of the file cannot be open for reading
   * 1 - P5
   * 2 - Pg
   * 3 - P6
   */
  int verif_format(const char* name); 

  ImageContent *load_convert_gray_image(const char *image_filename);

} /// end namespace kma

kma::ImageContent *add_image_border(kma::ImageContent *kmaimg, int b);
kma::ImageContent *add_image_border2(kma::ImageContent *kmaimg, int b, int &corrected_b);
       
#endif
