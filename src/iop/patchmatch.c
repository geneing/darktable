/*
    This file is part of darktable,
    copyright (c) 2020 eugene ingerman.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * Inpaint using the PatchMatch Algorithm
 *
 * | PatchMatch : A Randomized Correspondence Algorithm for Structural Image Editing
 * | by Connelly Barnes and Eli Shechtman and Adam Finkelstein and Dan B Goldman
 * | ACM Transactions on Graphics (Proc. SIGGRAPH), vol.28, aug-2009
 *
 * Original author Xavier Philippeau
 * Code adopted from: David Chatting https://github.com/davidchatting/PatchMatch
 */

/* The C version is adapted from https://github.com/younesse-cv/PatchMatch with author's permission
   with modifications from https://github.com/KDE/krita/blob/master/plugins/tools/tool_smart_patch/kis_inpaint.cpp
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "develop/imageop.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

//#define DUMP

#define INCREASE_PYRAMID_SIZE_RATE 2

typedef unsigned char uchar;
typedef unsigned char mask_t;

typedef struct{
    float* imageData;
    int nChannels;
    int height;
    int width;
} IplImage;

typedef struct{
    // image data
    mask_t* mask;
    IplImage* image;

    int mask_x[2], mask_y[2];

    int isNew;
} MaskedImage_T;

typedef MaskedImage_T* MaskedImage_P;

typedef struct{
    // image
    MaskedImage_P input;
    MaskedImage_P output;

    //  patch size
    int S;

    // Nearest-Neighbor Field 1 pixel = { x_target, y_target, distance_scaled }
    struct Field {
        int xt, yt;
        float distance;
    } **field;
    int fieldH;
    int fieldW;
} NNF_T;
typedef NNF_T* NNF_P;
typedef struct Field Field;

typedef struct{
    //initial image
    MaskedImage_P initial;

    // Nearest-Neighbor Fields
    NNF_P nnf_TargetToSource;

    // patch radius
    int radius;

    // Pyramid of downsampled initial images
    MaskedImage_P* pyramid;
    int nbEltPyramid;
    int nbEltMaxPyramid;
} Inpaint_T;

typedef Inpaint_T* Inpaint_P;


typedef struct{
    float r,g,b;
    float w;
} Vote_T;
typedef Vote_T* Vote_P;

typedef struct{
    Vote_P vote;
    int rows, cols;
    int size;
} Votes_T;
typedef Votes_T* Votes_P;

static Votes_P alloc_votes( int rows, int cols )
{
    Votes_P votes = (Votes_P) malloc(sizeof(Votes_T));
    votes->rows = rows;
    votes->cols = cols;
    votes->size = rows*cols;
    votes->vote = (Vote_P) malloc(sizeof(Vote_T)*votes->size);
    memset(votes->vote, 0, sizeof(Vote_T)*votes->size);
    return votes;
}

static void free_votes( Votes_P votes)
{
    free(votes->vote);
    free(votes);
}

static inline Vote_P get_vote(Votes_P votes, int x, int y)
{
    assert( x>=0 && x<votes->rows );
    assert( y>=0 && y<votes->cols );
    return votes->vote+x*votes->cols+y;
}


static inline float max1(float a, float b)
{
    return (a > b)? a : b;
}

static inline float min1(float a, float b)
{
    return (a < b)? a : b;
}

static inline float square(float x)
{
    return x*x;
}

void updateMaskBounds( MaskedImage_P im );
int isMasked(MaskedImage_P mIm, int x, int y);

//compute mask bounding box to speedup containsMasked computation
void updateMaskBounds( MaskedImage_P im )
{
    int numMasked = 0;

    im->mask_x[0] = im->image->height;
    im->mask_x[1] = 0;
    im->mask_y[0] = im->image->width;
    im->mask_y[1] = 0;

    for(int x=0; x<im->image->height; ++x){
        for(int y=0; y<im->image->width; ++y){
            if( isMasked(im, x, y) ){
                im->mask_x[0] = MIN(im->mask_x[0], x);
                im->mask_x[1] = MAX(im->mask_x[1], x);
                im->mask_y[0] = MIN(im->mask_y[0], y);
                im->mask_y[1] = MAX(im->mask_y[1], y);
                numMasked++;
            }
        }
    }
    if( numMasked==0 ){
        im->mask_x[0] = -1;
        im->mask_x[1] = 0;
        im->mask_y[0] = -1;
        im->mask_y[1] = 0;
    }
}


MaskedImage_P initMaskedImage(IplImage* image, mask_t* mask)
{
    MaskedImage_P mIm = (MaskedImage_P)malloc(sizeof(MaskedImage_T));
    // image data
    mIm->mask = mask;
    mIm->image = image;
    mIm->isNew=0;

    return mIm;
}


MaskedImage_P initNewMaskedImage(int width, int height, int ch)
{
    MaskedImage_P mIm = (MaskedImage_P)malloc(sizeof(MaskedImage_T));

    // image data
    mIm->mask = (mask_t*)calloc(width*height, sizeof(mask_t));

    mIm->image = (IplImage*) malloc(sizeof(IplImage));
    mIm->image->width = width;
    mIm->image->height = height;
    mIm->image->nChannels = ch;
    mIm->image->imageData = (float*) malloc(width*height*ch*sizeof(float));

    mIm->mask_x[0] = -1;
    mIm->mask_x[1] = 0;
    mIm->mask_y[0] = -1;
    mIm->mask_y[1] = 0;

    mIm->isNew=1;
    return mIm;
}

void freeMaskedImage(MaskedImage_P mIm)
{
    if (mIm!=NULL) {
        if (mIm->isNew) {
            if (mIm->mask!=NULL) {
                free(mIm->mask);
                mIm->mask=NULL;
            }
            if (mIm->image!=NULL) {
                free(mIm->image->imageData);
                free(mIm->image);
                mIm->image=NULL;
            }
        }
        free(mIm);
        mIm = NULL;
    }
}

//TODO: Optimize this!
static inline float* getSampleMaskedImage(const MaskedImage_P mIm, int x, int y)
{
    int channels=mIm->image->nChannels;
    int step = mIm->image->width * channels;
    return (mIm->image->imageData+x*step+y*channels);
}

//TODO: Optimize this!
void setSampleMaskedImage(MaskedImage_P mIm, int x, int y, float* value)
{
    int channels=mIm->image->nChannels;
    int step = mIm->image->width * channels;
    for(int band=0; band<3; ++band )
        mIm->image->imageData[x*step+y*channels+band] = value[band];
}

int isMasked(MaskedImage_P mIm, int x, int y)
{
    return (mIm->mask[x * mIm->image->width + y]==1);
}

void setMask(MaskedImage_P mIm, int x, int y, float value) {
    mIm->mask[x * mIm->image->width + y]= (value>0.f);
}

// return true if the patch contains one (or more) masked pixel
int containsMasked(MaskedImage_P mIm, int x, int y, int S)
{
    int min_ys = MAX(0, y-S);
    int max_ys = MIN( mIm->image->width, y+S);
    int min_xs = MAX(0, x-S);
    int max_xs = MIN( mIm->image->height-1, x+S);

    //shortcircuit - point is outside mask bounding box
    if( (min_xs > mIm->mask_x[1]) || (max_xs < mIm->mask_x[0]) ||
        (min_ys > mIm->mask_y[1]) || (max_ys < mIm->mask_y[0]) )
        return 0;

    for(int xs=min_xs; xs<=max_xs; xs++) {
        for(int ys=min_ys; ys<max_ys; ys++) {
            if( isMasked(mIm, xs, ys) ){
                return 1;
            }
        }
    }
    return 0;
}

static inline float gammaConversion( float rgb)
{
      return rgb <= 0.0031308 ? 12.92 * rgb : (1.0 + 0.055) * powf(rgb, 1.0 / 2.4) - 0.055;
}

static inline float RGB_distance( const float* c1, const float* c2)
{
    float mean_red = (c1[0] + c2[0])/2.f;
    float dRed = (c1[0]-c2[0]);
    float dGreen = (c1[1]-c2[1]);
    float dBlue = (c1[2]-c2[2]);

    //this is supposed to be distance in colorspace that reflects human perception better
    // https://en.wikipedia.org/wiki/Color_difference
    float distance = sqrtf( (2.f+mean_red)*dRed*dRed + 4.f*dGreen*dGreen + (3.f-mean_red)*dBlue*dBlue );
    return min1(distance/3.f, 1.f);
}

// distance between two patches in two images
float distanceMaskedImage(MaskedImage_P source, int xs, int ys, MaskedImage_P target, int xt, int yt, int S)
{
    float distance=0.f;
    int wsum=0;
    const float distmax = 1.f;
    int dy, dx;
    int xks, yks;
    int xkt, ykt;
    float rgb_dist = 0;
    float res;

    // for each pixel in the source patch
    for ( dy=-S ; dy<=S ; ++dy ) {
        for ( dx=-S ; dx<=S ; ++dx ) {
            wsum++;

            xks = xs+dx;
            yks = ys+dy;
            xkt = xt+dx;
            ykt = yt+dy;

            // cannot use masked pixels as a valid source of information
            if ( xks<0 || xks>=source->image->height || yks<0 || yks>=source->image->width ||
                 xkt<0 || xkt>=target->image->height || ykt<0 || ykt>=target->image->width ||
                 isMasked(source, xks, yks) || isMasked(target, xkt, ykt) ){
                distance+=distmax;
                continue;
            }

            // pixel values
            const float *s_value = getSampleMaskedImage(source, xks, yks);
            const float *t_value = getSampleMaskedImage(source, xkt, ykt);
            rgb_dist = RGB_distance(s_value, t_value); // distance between values in [0,1]

            // add pixel distance to global patch distance
            distance += rgb_dist/distmax;
        }
    }

    res = distance/wsum;
    return res;
}

// return a copy of the image
MaskedImage_P copyMaskedImage(MaskedImage_P mIm)
{
    int W = mIm->image->width;
    int H = mIm->image->height;

    MaskedImage_P copy;
    mask_t* newmask = (mask_t*)calloc(W*H, sizeof(mask_t));
    memcpy(newmask, mIm->mask, W*H*sizeof(mask_t));


    IplImage* newimage = (IplImage*) malloc(sizeof(IplImage));
    newimage->width = mIm->image->width;
    newimage->height = mIm->image->height;
    newimage->nChannels = mIm->image->nChannels;
    newimage->imageData = (float*) malloc(newimage->width*newimage->height*newimage->nChannels*sizeof(float));

    memcpy(newimage->imageData, mIm->image->imageData, W*H*mIm->image->nChannels*sizeof(float));

    copy = initMaskedImage(newimage, newmask);
    copy->isNew=1;
    copy->mask_x[0] = mIm->mask_x[0];
    copy->mask_x[1] = mIm->mask_x[1];
    copy->mask_y[0] = mIm->mask_y[0];
    copy->mask_y[1] = mIm->mask_y[1];

    return copy;
}

// return a downsampled image (factor 1/2)
MaskedImage_P downsample(MaskedImage_P source)
{
    const float kernel[2] = {1.,1.};
    int H, W;
    int x, y;
    int dx, dy;
    int xk, yk;
    float k, ky;
    float r=0, g=0, b=0, m=0, ksum=0;
    H=source->image->height;
    W=source->image->width;
    int newW=W/2, newH=H/2;

    MaskedImage_P newimage = initNewMaskedImage(newW, newH, source->image->nChannels);
    for (x=0;x<H-1;x+=2) {
        for (y=0;y<W-1;y+=2) {
            r=0; g=0; b=0; m=0; ksum=0;

            for (dy=0;dy<=1;++dy) {
                yk=y+dy;
                ky = kernel[dy];

                for (dx=0;dx<=1;++dx) {
                    xk = x+dx;

                    if (isMasked(source, xk, yk))
                        continue;

                    k = kernel[dx]*ky;
                    float* rgb = getSampleMaskedImage(source, xk, yk);
                    r+= k*rgb[0];
                    g+= k*rgb[1];
                    b+= k*rgb[2];
                    ksum+=k;
                    m++;
                }
            }

            if (ksum>0) {
                float rgb[]={ r/ksum, g/ksum, b/ksum };
                setSampleMaskedImage(newimage, x/2, y/2, rgb);
                setMask(newimage, x/2, y/2, 0.f);
            } else {
                float rgb[]={ 0, 0, 0 };
                setSampleMaskedImage(newimage, x/2, y/2, rgb);
                setMask(newimage, x/2, y/2, 1.f);
            }
        }
    }
    updateMaskBounds(newimage);
    return newimage;
}

#ifdef DUMP
void dumpMaskedImage( MaskedImage_P img, int level, int emloop, int tag )
{
    char buf[256];

    sprintf(buf,"img_%d_%d_%d.csv", level, emloop, tag);
    FILE *fimg = g_fopen(buf, "w");

    int H = img->image->height;
    int W = img->image->width;
    int c = img->image->nChannels;

    for(int x=0; x<H; ++x){
        for(int y=0; y<W; ++y){
            float sum = 0.f;
            for(int k=0; k<c; k++)
                sum += gammaConversion(img->image->imageData[(x*W+y)*c+k]);
            fprintf(fimg, "%g, ", sum );
//            fprintf(fimg, "%g, ", gammaConversion(img->image->imageData[(x*W+y)*c+0]));
        }
        fprintf(fimg, "\n");
    }


    sprintf(buf,"mask_%d_%d_%d.csv", level, emloop, tag);
    FILE *fmask = g_fopen(buf, "w");
    for(int x=0; x<H; ++x){
        for(int y=0; y<W; ++y){
            fprintf(fmask, "%d, ", img->mask[x*W+y]);
        }
        fprintf(fmask, "\n");
    }
    fclose(fmask);
    fclose(fimg);
}


void dumpField( NNF_P nnf, int level, int emloop )
{
    char buf[256];

    sprintf(buf,"x_%d_%d.csv", level, emloop);
    FILE *f0 = g_fopen(buf, "w");
    sprintf(buf,"y_%d_%d.csv", level, emloop);
    FILE *f1 = g_fopen(buf, "w");
    sprintf(buf,"d_%d_%d.csv", level, emloop);
    FILE *f2 = g_fopen(buf, "w");

    for(int x=0; x<nnf->fieldH; ++x){
        for(int y=0; y<nnf->fieldW; ++y){
            fprintf(f0, "%d, ", nnf->field[x][y].xt);
            fprintf(f1, "%d, ", nnf->field[x][y].yt);
            fprintf(f2, "%g, ", nnf->field[x][y].distance);
        }
        fprintf(f0, "\n");
        fprintf(f1, "\n");
        fprintf(f2, "\n");
    }
    fclose(f2);
    fclose(f1);
    fclose(f0);
}

void dumpVote(Votes_P votes, MaskedImage_P img, int level, int emloop)
{
    char buf[256];
    sprintf(buf,"r_%d_%d.csv", level, emloop);
    FILE *f0 = g_fopen(buf,"w");
    sprintf(buf,"g_%d_%d.csv", level, emloop);
    FILE *f1 = g_fopen(buf,"w");
    sprintf(buf,"w_%d_%d.csv", level, emloop);
    FILE *f2 = g_fopen(buf,"w");

    for( int x=0 ; x<img->image->height ; ++x){
        for( int y=0 ; y<img->image->width ; ++y){
            Vote_P vote = get_vote(votes, x, y);
            fprintf(f0, "%g, ", vote->r);
            fprintf(f1, "%g, ", vote->g);
            fprintf(f2, "%g, ", vote->w);
        }
        fprintf(f0, "\n");
        fprintf(f1, "\n");
        fprintf(f2, "\n");
    }
    fclose(f2);
    fclose(f1);
    fclose(f0);
}

#else
void dumpMaskedImage( MaskedImage_P img, int level, int emloop, int tag ){}
void dumpField( NNF_P nnf, int level, int emloop ){}
void dumpVote(Votes_P votes, MaskedImage_P img, int level, int emloop){}
#endif


void clearMask( MaskedImage_P img )
{
    for(int i=0; i< img->image->width*img->image->height; ++i)
        img->mask[i]=0;
}

// return an upscaled image
MaskedImage_P upscale(MaskedImage_P source, int newW,int newH)
{
    int x, y;
    int xs, ys;
    int H, W;

    H=source->image->height;
    W=source->image->width;

    MaskedImage_P newimage = initNewMaskedImage(newW, newH, source->image->nChannels);

    for (x=0;x<newH;x++) {
        for (y=0;y<newW;y++) {

            // original pixel
            ys = (y*W)/newW;
            xs = (x*H)/newH;

            // copy to new image
            if ( isMasked(source, xs, ys) ) {
                float rgb[]={ 0, 0, 0 };
                setSampleMaskedImage(newimage, x, y, rgb);
                setMask(newimage, x, y, 1);
            } else {
                float * rgb = getSampleMaskedImage(source, xs, ys);
                setSampleMaskedImage(newimage, x, y, rgb);
                setMask(newimage, x, y, 0);
            }
        }
    }
    updateMaskBounds(newimage);
    return newimage;
}

/**
* Nearest-Neighbor Field (see PatchMatch algorithm)
*  This algorithme uses a version proposed by Xavier Philippeau
*
*/
NNF_P initNNF(MaskedImage_P input, MaskedImage_P output, int patchsize)
{
    NNF_P nnf = (NNF_P)malloc(sizeof(NNF_T));
    nnf->input = input;
    nnf->output= output;
    nnf->S = patchsize;
    nnf->fieldH = nnf->input->image->height;
    nnf->fieldW = nnf->input->image->width;
    nnf->field=NULL;

    return nnf;
}


// compute distance between two patch
float distanceNNF(NNF_P nnf, int x,int y, int xp,int yp)
{
    return distanceMaskedImage(nnf->input,x,y, nnf->output,xp,yp, nnf->S);
}

void allocNNFField(NNF_P nnf)
{
    int i;
    if (nnf!=NULL){
        nnf->fieldH=nnf->input->image->height;
        nnf->fieldW=nnf->input->image->width;
        nnf->field = (Field **) malloc(sizeof(Field*)*nnf->fieldH);

        for ( i=0 ; i < nnf->fieldH ; i++ ) {
            nnf->field[i] = (Field *) malloc(sizeof(Field)*nnf->fieldW);
        }
    }
}

void freeNNFField(NNF_P nnf)
{
    int i;
    if ( nnf->field != NULL ){
        for ( i=0 ; i < nnf->fieldH ; ++i ){
            free(nnf->field[i]);
        }
        free(nnf->field);
        nnf->field=NULL;
    }
}

void freeNNF(NNF_P nnf)
{
    if (nnf!=NULL) {
        freeNNFField(nnf);
        free(nnf);
        nnf=NULL;
    }
}


// compute initial value of the distance term
void initializeNNF(NNF_P nnf)
{
    int y, x;
    int iter=0;
    const int maxretry=20;

    for (x=0;x<nnf->fieldH;++x) {
        for (y=0;y<nnf->fieldW;++y) {
            nnf->field[x][y].distance = distanceNNF(nnf, x, y,  nnf->field[x][y].xt, nnf->field[x][y].yt);
            // if the distance is high (all pixels masked ?), try to find a better link
            iter=0;
            while ( nnf->field[x][y].distance >= 1 && iter<maxretry) {
                nnf->field[x][y].xt = rand() % (nnf->output->image->height);
                nnf->field[x][y].yt = rand() % (nnf->output->image->width);
                nnf->field[x][y].distance = distanceNNF(nnf, x, y, nnf->field[x][y].xt, nnf->field[x][y].yt);
                iter++;
            }
        }
    }
}


// initialize field with random values
void randomize(NNF_P nnf)
{
    int i, j;
    // field
    allocNNFField(nnf);
    for (i=0; i<nnf->input->image->height; ++i){
        for (j=0; j<nnf->input->image->width; ++j){
            nnf->field[i][j].xt = rand() % (nnf->output->image->height);
            nnf->field[i][j].yt = rand() % (nnf->output->image->width);
            nnf->field[i][j].distance = 1;
        }
    }
    initializeNNF(nnf);
}

// initialize field from an existing (possibily smaller) NNF
void initializeNNFFromOtherNNF(NNF_P nnf, NNF_P otherNnf)
{
    int fx, fy, x, y, xlow, ylow;
    // field
    allocNNFField(nnf);
    fy = nnf->fieldW/otherNnf->fieldW;
    fx = nnf->fieldH/otherNnf->fieldH;
    nnf->S = otherNnf->S;

    for (x=0;x<nnf->fieldH;++x) {
        for (y=0;y<nnf->fieldW;++y) {
            xlow = (int)(min1(x/fx, otherNnf->input->image->height-1));
            ylow = (int)(min1(y/fy, otherNnf->input->image->width-1));
            nnf->field[x][y].xt = otherNnf->field[xlow][ylow].xt*fx;
            nnf->field[x][y].yt = otherNnf->field[xlow][ylow].yt*fy;
            nnf->field[x][y].distance = 1;
        }
    }
    initializeNNF(nnf);
}


// minimize a single link (see "PatchMatch" - page 4)
void minimizeLinkNNF(NNF_P nnf, int x, int y, int dir)
{
    int xp,yp,wi, xpi, ypi;
    float dp;
    //Propagation Up/Down
    if (x-dir>=0 && x-dir<nnf->input->image->height) {
        xp = nnf->field[x-dir][y].xt+dir;
        yp = nnf->field[x-dir][y].yt;
        dp = distanceNNF(nnf, x, y, xp, yp);
        if (dp < nnf->field[x][y].distance && xp < nnf->input->image->height && yp < nnf->input->image->width ) {
            nnf->field[x][y].xt = xp;
            nnf->field[x][y].yt = yp;
            nnf->field[x][y].distance = dp;
        }
    }

    //Propagation Left/Right
    if (y-dir>=0 && y-dir<nnf->input->image->width) {
        xp = nnf->field[x][y-dir].xt;
        yp = nnf->field[x][y-dir].yt+dir;
        dp = distanceNNF(nnf, x, y, xp, yp);
        if (dp < nnf->field[x][y].distance && xp < nnf->output->image->height && yp < nnf->output->image->width ) {
            nnf->field[x][y].xt = xp;
            nnf->field[x][y].yt = yp;
            nnf->field[x][y].distance = dp;
        }
    }

    //Random search
    wi=MAX(nnf->output->image->width, nnf->output->image->height);
    xpi=nnf->field[x][y].xt;
    ypi=nnf->field[x][y].yt;
    while (wi>0) {
        int r=(rand() % (2*wi)) - wi;
        xp = xpi + r;
        r=(rand() % (2*wi)) - wi;
        yp = ypi + r;
        xp = MAX(0, MIN(nnf->output->image->height-1, xp ));
        yp = MAX(0, MIN(nnf->output->image->width-1, yp ));

        dp = distanceNNF(nnf,x,y, xp,yp);
        if (dp<nnf->field[x][y].distance) {
            nnf->field[x][y].xt = xp;
            nnf->field[x][y].yt = yp;
            nnf->field[x][y].distance = dp;
        }
        wi/=2;
    }
}


// multi-pass NN-field minimization (see "PatchMatch" - page 4)
void minimizeNNF(NNF_P nnf, int pass)
{
    int min_x=0, min_y=0;
    int max_y=nnf->input->image->width-1;
    int max_x=nnf->input->image->height-1;

    // multi-pass minimization
    for (int i=0; i<pass; i++) {
        // scanline order
        for (int y=min_y; y<=max_y; ++y)
            for (int x=min_x; x<max_x; ++x)
                if (nnf->field[x][y].distance>0)
                    minimizeLinkNNF(nnf, x, y, +1);

        // reverse scanline order
        for (int y=max_y; y>=min_y; y--)
            for (int x=max_x; x>=min_x; x--)
                if (nnf->field[x][y].distance>0)
                    minimizeLinkNNF(nnf, x, y, -1);
    }
}

Inpaint_P initInpaint()
{
    Inpaint_P inp = (Inpaint_P)malloc(sizeof(Inpaint_T));
    //initial image
    inp->initial = NULL;

    // Nearest-Neighbor Fields
    inp->nnf_TargetToSource = NULL;

    // Pyramid of downsampled initial images
    inp->pyramid = NULL;
    inp->nbEltPyramid = 0;
    inp->nbEltMaxPyramid = 0;

    return inp;
}

void addEltInpaintingPyramid(Inpaint_P imp, MaskedImage_P elt)
{
    int inc = INCREASE_PYRAMID_SIZE_RATE;
    if (inc<2)
        inc = 2;

    if (imp->pyramid == NULL || imp->nbEltMaxPyramid == 0) {
        imp->nbEltMaxPyramid = inc;
        imp->pyramid = (MaskedImage_P*)malloc(sizeof(MaskedImage_P)*imp->nbEltMaxPyramid);
    } else if (imp->nbEltPyramid == imp->nbEltMaxPyramid) {
        imp->nbEltMaxPyramid = imp->nbEltMaxPyramid*inc;
        imp->pyramid = (MaskedImage_P*)realloc(imp->pyramid, sizeof(MaskedImage_P)*imp->nbEltMaxPyramid);
    }

    imp->pyramid[imp->nbEltPyramid] = copyMaskedImage(elt);
    imp->nbEltPyramid++;
}

// Maximization Step : Maximum likelihood of target pixel
void MaximizationStep(MaskedImage_P target, Votes_P votes)
{
    int y, x, H, W;
    H = target->image->height;
    W = target->image->width;
    for( x=0 ; x<H ; ++x){
        for( y=0 ; y<W ; ++y){
            Vote_P vote = get_vote(votes, x, y); //TODO this can be optimized by using iterator
            if (vote->w > 0) {
                float r = (vote->r / vote->w);
                float g = (vote->g / vote->w);
                float b = (vote->b / vote->w);
                float rgb[]={r, g, b};
                setSampleMaskedImage(target, x, y, rgb);
                setMask(target, x, y, 0);
            }
        }
    }
    updateMaskBounds(target);
}


void weightedCopy(MaskedImage_P src, int xs, int ys, Votes_P votes, int xd,int yd, float w)
{
    if (isMasked(src, xs, ys))
        return;

    float *rgb = getSampleMaskedImage(src, xs, ys);
    Vote_P vote = get_vote(votes, xd, yd);
    vote->r += w*rgb[0];
    vote->g += w*rgb[1];
    vote->b += w*rgb[2];
    vote->w += w;
}

static inline float similarity( float distance )
{
    const float s_zero = 0.999f;
    const float t_halfmax = 0.1f;
    const float x  = (s_zero - 0.5f) * 2.f;
    const float invtanh = 0.5f * logf((1.f + x) / (1.f - x));
    const float coef = invtanh / t_halfmax;

    if( distance<0 || distance >= 1. )
        return 0;

    float similarity = 0.5f - 0.5f * tanh(coef * (distance - t_halfmax));

    return similarity;
}

// Expectation Step : vote for best estimations of each pixel
void ExpectationStep(NNF_P nnf, Votes_P votes, MaskedImage_P source, MaskedImage_P target, int upscale)
{
    int y, x, /*H, W,*/ dy, dx;
    Field** field = nnf->field;
    int R = nnf->S; /////////////int R = nnf->PatchSize;
    if(upscale)
        R *= 2;

    float w;

    int H = nnf->fieldH;
    int W = nnf->fieldW;
    int H_target = target->image->height;
    int W_target = target->image->width;
    int H_source = source->image->height;
    int W_source = source->image->width;

    for ( x=0 ; x<H_target ; ++x) {
        for ( y=0 ; y<W_target; ++y) {

            // vote for each pixel inside the input patch

            if( !containsMasked(source, x, y, R+4) ){ //why R+4?
                float *rgb = getSampleMaskedImage(source, x, y);
                Vote_P vote = get_vote(votes, x, y);
                vote->r = rgb[0];
                vote->g = rgb[1];
                vote->b = rgb[2];
                vote->w = 1.f;
            }
            else {
                // vote for each pixel inside the input patch
                for ( dy=-R ; dy<=R ; ++dy) {
                    for ( dx=-R ; dx<=R; ++dx) {

                        // xpt,ypt = center pixel of the target patch
                        int xpt = x + dx;
                        int ypt = y + dy;

                        int xst, yst;

                        // add vote for the value
                        if (upscale) {
                            if (xpt<0 || (xpt/2)>=H) continue;
                            if (ypt<0 || (ypt/2)>=W) continue;
                            xst = 2 * field[xpt / 2][ypt / 2].xt + (xpt % 2);
                            yst = 2 * field[xpt / 2][ypt / 2].yt + (ypt % 2);

                            int dp = field[xpt / 2][ypt / 2].distance;
                            // similarity measure between the two patches
                            w = similarity(dp);
                        } else {
                            if (xpt<0 || xpt>=H) continue;
                            if (ypt<0 || ypt>=W) continue;
                            xst = field[xpt][ypt].xt;
                            yst = field[xpt][ypt].yt;
                            int dp = field[xpt][ypt].distance;
                            // similarity measure between the two patches
                            w = similarity(dp);
                        }
                        int xs = xst - dx;
                        int ys = yst - dy;

                        if (xs < 0 || xs >= H_source || ys < 0 || ys >= W_source)
                            continue;
                        weightedCopy(source, xs, ys, votes, x, y, w);
                    }
                }
            }
        }
    }
}

// EM-Like algorithm (see "PatchMatch" - page 6)
// Returns a double sized target image
MaskedImage_P ExpectationMaximization(Inpaint_P imp, int level)
{
    int emloop, H, W;

    int iterEM = MIN(2*level, 4);
    int iterNNF = MIN(5, 1+level);

    int upscaled;
    MaskedImage_P newsource = imp->pyramid[level-1];
    MaskedImage_P source = imp->nnf_TargetToSource->output;
    MaskedImage_P target = imp->nnf_TargetToSource->input;
    MaskedImage_P newtarget = NULL;

    // EM Loop
    for (emloop=1; emloop<=iterEM; emloop++) {
        // set the new target as current target
        if (newtarget!=NULL) {
            imp->nnf_TargetToSource->input = newtarget;
            target = newtarget;
            newtarget = NULL;
        }

        H = target->image->height;
        W = target->image->width;
        dumpMaskedImage(target, level, emloop, 0);
        dumpMaskedImage(source, level, emloop, 1);

        for (int x=0 ; x<H ; ++x){
            for (int y=0 ; y<W ; ++y){
                if (!containsMasked(source, x, y, imp->radius)) {
                    imp->nnf_TargetToSource->field[x][y].xt = x;
                    imp->nnf_TargetToSource->field[x][y].yt = y;
                    imp->nnf_TargetToSource->field[x][y].distance = 0;
                }
            }
        }

        // -- minimize the NNF
        minimizeNNF(imp->nnf_TargetToSource, iterNNF);
        dumpField(imp->nnf_TargetToSource, level, emloop);

        // -- Now we rebuild the target using best patches from source

        // Instead of upsizing the final target, we build the last target from the next level source image
        // So the final target is less blurry (see "Space-Time Video Completion" - page 5)
        if (level>=1 && (emloop==iterEM)) {
            newsource = imp->pyramid[level-1];
            newtarget = upscale(target, newsource->image->width, newsource->image->height);
            upscaled = 1;
        } else {
            newsource = imp->pyramid[level];
            newtarget = copyMaskedImage(target);
            upscaled = 0;
        }

        dumpMaskedImage(newsource, level, emloop, 2);
        dumpMaskedImage(newtarget, level, emloop, 3);

        // --- EXPECTATION STEP ---
        // votes for best patch from NNF Source->Target (completeness) and Target->Source (coherence)
        Votes_P votes =  alloc_votes(newtarget->image->height, newtarget->image->width);

        ExpectationStep(imp->nnf_TargetToSource, votes, newsource, newtarget, upscaled);

        // --- MAXIMIZATION STEP ---
        // compile votes and update pixel values
        dumpVote(votes, newtarget, level, emloop);
        MaximizationStep(newtarget, votes);
        dumpMaskedImage(newtarget, level, emloop, 4);
        free_votes( votes );
    }

    return newtarget;
}

int countMasked(MaskedImage_P source)
{
    int count = 0;
    for(int i=0; i<source->image->width*source->image->height; ++i)
        count += (source->mask[i]>0 );
    return count;
}

MaskedImage_P inpaint_impl(IplImage* input, mask_t* mask, int radius)
{
    Inpaint_P imp = initInpaint();
    NNF_P new_nnf_rev;

    // initial image
    imp->initial = initMaskedImage(input, mask);
    updateMaskBounds(imp->initial);

    // patch radius
    imp->radius = radius;

    // working copies
    MaskedImage_P source = imp->initial;

    // build pyramid of downscaled images
    addEltInpaintingPyramid(imp, imp->initial);

    while ( countMasked(source) > 0 && source->image->width>radius && source->image->height>radius ) {
        source = downsample(source);
        addEltInpaintingPyramid(imp, source);
    }
    int maxlevel=imp->nbEltPyramid;

    MaskedImage_P target = copyMaskedImage(source);
    clearMask(target);
    updateMaskBounds(target);

    // for each level of the pyramid
    for (int level=maxlevel-1 ; level>0 ; level--) {

        // create Nearest-Neighbor Fields (direct and reverse)
        source = imp->pyramid[level];

        if (level==maxlevel-1) {
            // at first, we use the same image for target and source
            // and use random data as initial guess

            imp->nnf_TargetToSource = initNNF(target, source, radius);
            randomize(imp->nnf_TargetToSource);
        } else {
            // then, we use the rebuilt (upscaled) target
            // and re-use the previous NNF as initial guess
            new_nnf_rev = initNNF(target, source, radius);
            initializeNNFFromOtherNNF(new_nnf_rev, imp->nnf_TargetToSource);
            imp->nnf_TargetToSource = new_nnf_rev;
        }
        //Build an upscaled target by EM-like algorithm (see "PatchMatch" paper referenced above - page 6)
        target = ExpectationMaximization(imp, level);
    }

    free(imp);
    return target;
}

void inpaint( const float *const in,
              float *const out, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out, float *const mask, int nChannels )
{
    clock_t start, end;
    double cpu_time_used;


    const int radius = 2.;
    int nPix = roi_in->width*roi_in->height*nChannels;
    mask_t* newmask = (mask_t*) malloc(nPix*sizeof(mask_t));

    float* data = (float*) malloc(nPix*sizeof(float));
    memcpy(data, in, nPix*sizeof(float));

    IplImage image;
    image.imageData = data;
    image.width = roi_in->width;
    image.height = roi_in->height;
    image.nChannels = nChannels;

    for(int i=0; i<roi_in->width*roi_in->height; ++i){
        newmask[i] = (mask[i]>0.f);
    }

    start = clock();
    MaskedImage_P output = inpaint_impl(&image, newmask, radius);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    fprintf(stderr, "[patchmatch] %dx%d : %g sec\n", roi_in->width, roi_in->height, cpu_time_used);
    //TODO: this is actually wrong. Needs correct implementation of coordinate transforms.
    for(int i=0; i<nPix; ++i){
        out[i] = output->image->imageData[i];
    }
    freeMaskedImage( output );

    //TODO: copy data to out, taking masking into account
    free(data);
    free(newmask);
}


void freeInpaintingPyramid(Inpaint_P inp)
{
    int i;
    if (inp->pyramid != NULL) {
        for ( i=0 ; i<inp->nbEltPyramid ; ++i)
            freeMaskedImage(inp->pyramid[i]);

        free(inp->pyramid);
        inp->pyramid = NULL;
    }
}
