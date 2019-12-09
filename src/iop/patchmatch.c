/*
    This file is part of darktable,
    copyright (c) 2019 eugene ingerman.

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

// the maximum value returned by MaskedImage.distance()
#define DSCALE 65535
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

    // array for converting distance to similarity
    float * similarity;

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
    int *** field;
    int fieldH;
    int fieldW;
} NNF_T;
typedef NNF_T* NNF_P;

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


float max1(float a, float b)
{
    return (a + b + fabs(a-b) ) / 2;
}

float min1(float a, float b)
{
    return (a + b - fabs(a-b) ) / 2;
}

float square(float x)
{
    return x*x;
}

// Variables globales
float* G_globalSimilarity;
int G_initSim;


void initSimilarity0()
{
    int i, j, k, length;
    float base[11] = {1.0, 0.99, 0.96, 0.83, 0.38, 0.11, 0.02, 0.005, 0.0006, 0.0001, 0 };
    float t, vj, vk;
    length = (DSCALE+1);
    if (!G_initSim) {
        G_globalSimilarity = (float *) calloc(length, sizeof(float));
        for ( i=0 ; i<length ; ++i) {
            t = (float)i/length;
            j = (int)(100*t);
            k=j+1;
            vj = (j<11)?base[j]:0;
            vk = (k<11)?base[k]:0;
            G_globalSimilarity[i] = vj + (100*t-j)*(vk-vj);
        }
    }
    G_initSim = 1;
}

void initSimilarity()
{
    int i, length;
    float s_zero = 0.999f;
    float t_halfmax = 0.10f;
    float t;
    float x  = (s_zero - 0.5f) * 2.f;
    float invtanh = 0.5f * logf((1.f + x) / (1.f - x));
    float coef = invtanh / t_halfmax;

    length = (DSCALE+1);
    if (!G_initSim){
        G_globalSimilarity = (float *) calloc(length, sizeof(float));
        for (i=0;i<length;i++) {
            t = (float)i/length;
            G_globalSimilarity[i] = 0.5f - 0.5f * tanh(coef * (t - t_halfmax));
        }
    }
    G_initSim = 1;
}

MaskedImage_P initMaskedImage(IplImage* image, mask_t* mask)
{
    MaskedImage_P mIm = (MaskedImage_P)malloc(sizeof(MaskedImage_T));
    // image data
    mIm->mask = mask;
    mIm->image = image;

    initSimilarity();
    mIm->similarity = G_globalSimilarity;

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

    //TODO: remove the lines below - redundant
    initSimilarity();
    mIm->similarity = G_globalSimilarity;

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

float getSampleMaskedImage(const MaskedImage_P mIm, int x, int y, int band)
{
    int channels=mIm->image->nChannels;
    int step = mIm->image->width * channels;
    return mIm->image->imageData[x*step+y*channels+band];
}

void setSampleMaskedImage(MaskedImage_P mIm, int x, int y, int band, float value)
{
    int channels=mIm->image->nChannels;
    int step = mIm->image->width * channels;
    mIm->image->imageData[x*step+y*channels+band] = value;
}

int isMasked(MaskedImage_P mIm, int x, int y)
{
//    if (mIm==NULL || mIm->mask==NULL)
//        return 0;
    return mIm->mask[x * mIm->image->width + y];
}

void setMask(MaskedImage_P mIm, int x, int y, float value) {
//    if (mIm==NULL || mIm->mask==NULL)
//        return;
    mIm->mask[x * mIm->image->width + y]= (value>0.f);
}

// return true if the patch contains one (or more) masked pixel
int containsMasked(MaskedImage_P mIm, int x, int y, int S)
{
    int dy, dx;
    int xs, ys;
    for (dy=-S;dy<=S;dy++) {
        for (dx=-S;dx<=S;dx++) {
            xs=x+dx;
            ys=y+dy;
            if (xs<0 || xs>=mIm->image->height)
                continue;
            if (ys<0 || ys>=mIm->image->width)
                continue;
            if ( isMasked(mIm, xs, ys) )
                return 1;
        }
    }
    return 0;
}

// distance between two patches in two images
int distanceMaskedImage(MaskedImage_P source, int xs, int ys, MaskedImage_P target, int xt, int yt, int S)
{
    float distance=0.f;
    int wsum=0;
//    const float ssdmax = 9;
    const float ssdmax = 3;
    int dy, dx, band;
    int xks, yks;
    int xkt, ykt;
    float ssd = 0;
    long res;
    float s_value, t_value;
    //float s_gx, t_gx, s_gy, t_gy;

    // for each pixel in the source patch
    for ( dy=-S ; dy<=S ; ++dy ) {
        for ( dx=-S ; dx<=S ; ++dx ) {

            xks = xs+dx;
            yks = ys+dy;
            xkt = xt+dx;
            ykt = yt+dy;
            wsum++;

            if ( xks<1 || xks>=source->image->height-1 ) {distance++; continue;}
            if ( yks<1 || yks>=source->image->width-1 ) {distance++; continue;}

            // cannot use masked pixels as a valid source of information
            if (isMasked(source, xks, yks)) {distance++; continue;}

            // corresponding pixel in the target patch
            if (xkt<1 || xkt>=target->image->height-1) {distance++; continue;}
            if (ykt<1 || ykt>=target->image->width-1) {distance++; continue;}

            // cannot use masked pixels as a valid source of information
            if (isMasked(target, xkt, ykt)) {distance++; continue;}

            ssd=0;
            for (band=0; band<3; ++band) {
                // pixel values
                s_value = getSampleMaskedImage(source, xks, yks, band);
                t_value = getSampleMaskedImage(source, xkt, ykt, band);

//                // pixel horizontal gradients (Gx)
//                s_gx = (getSampleMaskedImage(source, xks+1, yks, band) - getSampleMaskedImage(source, xks-1, yks, band))/2;
//                t_gx = (getSampleMaskedImage(target, xkt+1, ykt, band) - getSampleMaskedImage(target, xkt-1, ykt, band))/2;

//                // pixel vertical gradients (Gy)
//                s_gy = (getSampleMaskedImage(source, xks, yks+1, band) - getSampleMaskedImage(source, xks, yks-1, band))/2;
//                t_gy = (getSampleMaskedImage(target, xkt, ykt+1, band) - getSampleMaskedImage(target, xkt, ykt-1, band))/2;

                ssd += square((float)s_value-t_value); // distance between values in [0,1]
//                ssd += square((float)s_gx-t_gx); // distance between Gx in [0,1]
//                ssd += square((float)s_gy-t_gy); // distance between Gy in [0,1]
            }

            // add pixel distance to global patch distance
            distance += ssd/ssdmax;
        }
    }

    res = (wsum>0) ? (int)(DSCALE*distance/wsum) : DSCALE;
    if (res < 0 || res > DSCALE)
        return DSCALE;
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

    return copy;
}

// return a downsampled image (factor 1/2)
MaskedImage_P downsample(MaskedImage_P source) {
    const float kernel[6] = {1,2,4,4,2,1};
    int H, W;
    int x, y;
    int xs, ys;
    int dx, dy;
    int xk, yk;
    float k, ky;
    float r=0, g=0, b=0, m=0, ksum=0;

    H=source->image->height;
    W=source->image->width;
    int newW=W/2, newH=H/2;

    MaskedImage_P newimage = initNewMaskedImage(newW, newH, source->image->nChannels);
    xs=0;
    for(x=0;x<newH;++x) {
        ys=0;
        for(y=0;y<newW;++y) {
            r=0; g=0; b=0; m=0; ksum=0;

            for(dy=-2;dy<=3;dy++) {
                yk=ys+dy;
                if (yk<0 || yk>=W) continue;
                ky = kernel[2+dy];
                for(dx=-2;dx<=3;dx++) {
                    xk = xs+dx;
                    if (xk<0 || xk>=H) continue;

                    if (source->mask[xk*W+yk]) continue;

                    k = kernel[2+dx]*ky;
                    r+= k*getSampleMaskedImage(source, xk, yk, 0);
                    g+= k*getSampleMaskedImage(source, xk, yk, 1);
                    b+= k*getSampleMaskedImage(source, xk, yk, 2);
                    ksum+=k;
                    m++;
                }
            }
            if (ksum>0) {r/=ksum; g/=ksum; b/=ksum;}

            if (m!=0) {
                setSampleMaskedImage(newimage, x, y, 0, r);
                setSampleMaskedImage(newimage, x, y, 1, g);
                setSampleMaskedImage(newimage, x, y, 2, b);
                setMask(newimage, x, y, 0);
            } else {
                setMask(newimage, x, y, 1);
                setSampleMaskedImage(newimage, x, y, 0, 0);
                setSampleMaskedImage(newimage, x, y, 1, 0);
                setSampleMaskedImage(newimage, x, y, 2, 0);
            }
            ys+=2;
        }
        xs+=2;
    }
    return newimage;
}

// return a downsampled image (factor 1/2)
MaskedImage_P downsample2(MaskedImage_P source)
{
    const float kernel[6] = {1,5,10,10,5,1};
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

            for (dy=-2;dy<=3;++dy) {
                yk=y+dy;
                if (yk<0 || yk>=W)
                    continue;
                ky = kernel[2+dy];
                for (dx=-2;dx<=3;++dx) {
                    xk = x+dx;
                    if (xk<0 || xk>=H)
                        continue;

                    if (source->mask[xk*W+yk])
                        continue;

                    k = kernel[2+dx]*ky;
                    r+= k*getSampleMaskedImage(source, xk, yk, 0);
                    g+= k*getSampleMaskedImage(source, xk, yk, 1);
                    b+= k*getSampleMaskedImage(source, xk, yk, 2);
                    ksum+=k;
                    m++;
                }
            }
            if (ksum>0) {
                r/=ksum;
                g/=ksum;
                b/=ksum;
            }

            if (m!=0) {
                setSampleMaskedImage(newimage, x/2, y/2, 0, r);
                setSampleMaskedImage(newimage, x/2, y/2, 1, g);
                setSampleMaskedImage(newimage, x/2, y/2, 2, b);
                setMask(newimage, x/2, y/2, 0);
            } else {
                setMask(newimage, x/2, y/2, 1);
            }
        }
    }
    return newimage;
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
            if (source->mask[xs*W+ys]) {
                setSampleMaskedImage(newimage, x, y, 0, 0);
                setSampleMaskedImage(newimage, x, y, 1, 0);
                setSampleMaskedImage(newimage, x, y, 2, 0);
                setMask(newimage, x, y, 1);
            } else {
                setSampleMaskedImage(newimage, x, y, 0, getSampleMaskedImage(source, xs, ys, 0));
                setSampleMaskedImage(newimage, x, y, 1, getSampleMaskedImage(source, xs, ys, 1));
                setSampleMaskedImage(newimage, x, y, 2, getSampleMaskedImage(source, xs, ys, 2));
                setMask(newimage, x, y, 0);
            }
        }
    }

    return newimage;
}


void dumpField( NNF_P nnf )
{
    FILE *f0 = g_fopen("x.csv","w");
    FILE *f1 = g_fopen("y.csv","w");
    FILE *f2 = g_fopen("d.csv","w");

    for(int x=0; x<nnf->fieldH; ++x){
        for(int y=0; y<nnf->fieldW; ++y){
            fprintf(f0, "%d, ", nnf->field[x][y][0]);
            fprintf(f1, "%d, ", nnf->field[x][y][1]);
            fprintf(f2, "%d, ", nnf->field[x][y][2]);
        }
        fprintf(f0, "\n");
        fprintf(f1, "\n");
        fprintf(f2, "\n");
    }
    fclose(f2);
    fclose(f1);
    fclose(f0);
}

void dumpVote(float*** vote, MaskedImage_P img)
{
    FILE *f0 = g_fopen("r.csv","w");
    FILE *f1 = g_fopen("g.csv","w");
    FILE *f2 = g_fopen("w.csv","w");

    for( int x=0 ; x<img->image->height ; ++x){
        for( int y=0 ; y<img->image->width ; ++y){
            fprintf(f0, "%g, ", vote[x][y][0]);
            fprintf(f1, "%g, ", vote[x][y][1]);
            fprintf(f2, "%g, ", vote[x][y][3]);
        }
        fprintf(f0, "\n");
        fprintf(f1, "\n");
        fprintf(f2, "\n");
    }
    fclose(f2);
    fclose(f1);
    fclose(f0);
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
    nnf->fieldH = 0;
    nnf->fieldW = 0;
    nnf->field=NULL;

    return nnf;
}


// compute distance between two patch
int distanceNNF(NNF_P nnf, int x,int y, int xp,int yp)
{
    return distanceMaskedImage(nnf->input,x,y, nnf->output,xp,yp, nnf->S);
}

void allocNNFField(NNF_P nnf)
{
    int i, j;
    if (nnf!=NULL){
        nnf->fieldH=nnf->input->image->height;
        nnf->fieldW=nnf->input->image->width;
        nnf->field = (int ***) malloc(sizeof(int**)*nnf->fieldH);

        for ( i=0 ; i < nnf->fieldH ; i++ ) {
            nnf->field[i] = (int **) malloc(sizeof(int*)*nnf->fieldW);
            for (j=0 ; j<nnf->fieldW ; j++ ) {
                nnf->field[i][j] = (int *) calloc(3,sizeof(int));
            }
        }
    }
}

void freeNNFField(NNF_P nnf)
{
    int i, j;
    if ( nnf->field != NULL ){
        for ( i=0 ; i < nnf->fieldH ; ++i ){
            for ( j=0 ; j < nnf->fieldW ; ++j ){
                free( nnf->field[i][j] );
            }
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
            nnf->field[x][y][2] = distanceNNF(nnf, x,y,  nnf->field[x][y][0],nnf->field[x][y][1]);
            // if the distance is INFINITY (all pixels masked ?), try to find a better link
            iter=0;
            while ( nnf->field[x][y][2] == DSCALE && iter<maxretry) {
                nnf->field[x][y][0] = rand() % (nnf->output->image->height + 1);
                nnf->field[x][y][1] = rand() % (nnf->output->image->width + 1);
                nnf->field[x][y][2] = distanceNNF(nnf, x, y, nnf->field[x][y][0], nnf->field[x][y][1]);
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
            nnf->field[i][j][0] = rand() % (nnf->output->image->height + 1);
            nnf->field[i][j][1] = rand() % (nnf->output->image->width + 1);
            nnf->field[i][j][2] = DSCALE;
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
            nnf->field[x][y][0] = otherNnf->field[xlow][ylow][0]*fx;
            nnf->field[x][y][1] = otherNnf->field[xlow][ylow][1]*fy;
            nnf->field[x][y][2] = DSCALE;
        }
    }
    initializeNNF(nnf);
}


// minimize a single link (see "PatchMatch" - page 4)
void minimizeLinkNNF(NNF_P nnf, int x, int y, int dir)
{
    int xp,yp,dp,wi, xpi, ypi;
    //Propagation Up/Down
    if (x-dir>0 && x-dir<nnf->input->image->height) {
        xp = nnf->field[x-dir][y][0]+dir;
        yp = nnf->field[x-dir][y][1];
        dp = distanceNNF(nnf,x,y, xp,yp);
        if (dp < nnf->field[x][y][2]) {
            nnf->field[x][y][0] = xp;
            nnf->field[x][y][1] = yp;
            nnf->field[x][y][2] = dp;
        }
    }

    //Propagation Left/Right
    if (y-dir>0 && y-dir<nnf->input->image->width) {
        xp = nnf->field[x][y-dir][0];
        yp = nnf->field[x][y-dir][1]+dir;
        dp = distanceNNF(nnf,x,y, xp,yp);
        if (dp<nnf->field[x][y][2]) {
            nnf->field[x][y][0] = xp;
            nnf->field[x][y][1] = yp;
            nnf->field[x][y][2] = dp;
        }
    }

    //Random search
    wi=MAX(nnf->output->image->width, nnf->output->image->height);
    xpi=nnf->field[x][y][0];
    ypi=nnf->field[x][y][1];
    while (wi>0) {
        int r=(rand() % (2*wi)) - wi;
        xp = xpi + r;
        r=(rand() % (2*wi)) - wi;
        yp = ypi + r;
        xp = MAX(0, MIN(nnf->output->image->height-1, xp ));
        yp = MAX(0, MIN(nnf->output->image->width-1, yp ));

        dp = distanceNNF(nnf,x,y, xp,yp);
        if (dp<nnf->field[x][y][2]) {
            nnf->field[x][y][0] = xp;
            nnf->field[x][y][1] = yp;
            nnf->field[x][y][2] = dp;
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
        for (int x=min_x; x<max_x; ++x)
            for (int y=min_y; y<=max_y; ++y)
                if (nnf->field[x][y][2]>0)
                    minimizeLinkNNF(nnf, x, y, +1);

        // reverse scanline order
        for (int x=max_x; x>=min_x; x--)
            for (int y=max_y; y>=min_y; y--)
                if (nnf->field[x][y][2]>0)
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

    imp->pyramid[imp->nbEltPyramid] = elt;
    imp->nbEltPyramid++;
}

// Maximization Step : Maximum likelihood of target pixel
void MaximizationStep(MaskedImage_P target, float*** vote)
{
    int y, x, H, W;
    H = target->image->height;
    W = target->image->width;
    for( x=0 ; x<H ; ++x){
        for( y=0 ; y<W ; ++y){
            if (vote[x][y][3]>0) {
                float r = (vote[x][y][0]/vote[x][y][3]);
                float g = (vote[x][y][1]/vote[x][y][3]);
                float b = (vote[x][y][2]/vote[x][y][3]);

                setSampleMaskedImage(target, x, y, 0, r );
                setSampleMaskedImage(target, x, y, 1, g );
                setSampleMaskedImage(target, x, y, 2, b );
                setMask(target, x, y, 0);
            }
        }
    }
}


void weightedCopy(MaskedImage_P src, int xs, int ys, float*** vote, int xd,int yd, float w)
{
//    if (isMasked(src, xs, ys))
//        return;

    vote[xd][yd][0] += w*getSampleMaskedImage(src, xs, ys, 0);
    vote[xd][yd][1] += w*getSampleMaskedImage(src, xs, ys, 1);
    vote[xd][yd][2] += w*getSampleMaskedImage(src, xs, ys, 2);
    vote[xd][yd][3] += w;
}


// Expectation Step : vote for best estimations of each pixel
void ExpectationStep(NNF_P nnf, float*** vote, MaskedImage_P source, MaskedImage_P target, int upscale)
{
    int y, x, H, W, dp, dy, dx;
    int xs, ys;
    int*** field = nnf->field;
    int R = nnf->S; /////////////int R = nnf->PatchSize;
    float w;

    H = nnf->fieldH;
    W = nnf->fieldW;
    int H_target = target->image->height;
    int W_target = target->image->width;
    int H_source = source->image->height;
    int W_source = source->image->width;

    for ( x=0 ; x<H_target ; ++x) {
        for ( y=0 ; y<W_target; ++y) {
            // x,y = center pixel of patch in input
            // xp,yp = center pixel of best corresponding patch in output
//            int xp=field[x][y][0];
//            int yp=field[x][y][1];
//            dp=field[x][y][2];

            // similarity measure between the two patches
//            w = G_globalSimilarity[dp];

            if( !containsMasked(source, x, y, R+4) ){ //why R+4?
                vote[x][y][0] = getSampleMaskedImage(source, x, y, 0);
                vote[x][y][1] = getSampleMaskedImage(source, x, y, 1);
                vote[x][y][2] = getSampleMaskedImage(source, x, y, 2);
                vote[x][y][3] = 1.f;
            }
            else{
                // vote for each pixel inside the input patch
                for ( dy=-R ; dy<=R ; ++dy) {
                    for ( dx=-R ; dx<=R; ++dx) {
                        // xpt,ypt = center pixel of the target patch
                        int xpt = x + dx;
                        int ypt = y + dy;
                        int xst, yst;

                        if (!upscale) {
                            if (xpt < 0 || xpt >= H || ypt < 0 || ypt >= W)
                                continue;

                            xst=field[xpt][ypt][0];
                            yst=field[xpt][ypt][1];
                            dp=field[xpt][ypt][2];
                            w = G_globalSimilarity[dp];
                        }
                        else{
                            if (xpt < 0 || (xpt / 2) >= H || ypt < 0 || (ypt / 2) >= W)
                                continue;
                            xst = 2 * field[xpt/2][ypt/2][0] + (xpt % 2);
                            yst = 2 * field[xpt/2][ypt/2][1] + (ypt % 2);
                            dp = field[xpt/2][ypt/2][2];

                            // similarity measure between the two patches
                            w = G_globalSimilarity[dp];
                        }

                        xs = xst - dx;
                        ys = yst - dy;

                        if (xs < 0 || xs >= H_source || ys < 0 || ys >= W_source)
                            continue;

                        weightedCopy(source, xs, ys, vote, xpt, ypt, w);
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
    float*** vote;

    int iterEM = MIN(2*level, 4);
    int iterNNF = MIN(5, 1+level);

    int upscaled;
    MaskedImage_P newsource;
    MaskedImage_P source = imp->nnf_TargetToSource->input;
    MaskedImage_P target = imp->nnf_TargetToSource->output;
    MaskedImage_P newtarget = NULL;

    //printf("EM loop (em=%d,nnf=%d) : ", iterEM, iterNNF);

    // EM Loop
    for (emloop=1; emloop<=iterEM; emloop++) {
        //printf(" %d", 1+iterEM-emloop);
        // set the new target as current target
        if (newtarget!=NULL) {
            imp->nnf_TargetToSource->input = newtarget;
            target = newtarget;
            newtarget = NULL;
        }

        H = target->image->height;
        W = target->image->width;
        for (int x=0 ; x<H ; ++x){
            for (int y=0 ; y<W ; ++y){
                if (!containsMasked(source, x, y, imp->radius)) {
                    imp->nnf_TargetToSource->field[x][y][0] = x;
                    imp->nnf_TargetToSource->field[x][y][1] = y;
                    imp->nnf_TargetToSource->field[x][y][2] = 0;
                }
                else{
                    imp->nnf_TargetToSource->field[x][y][2] = 10*DSCALE;
                }
            }
        }

        // -- minimize the NNF
        minimizeNNF(imp->nnf_TargetToSource, iterNNF);

        if( level==1 && emloop==2)
            dumpField(imp->nnf_TargetToSource);


        // -- Now we rebuild the target using best patches from source
        upscaled = 0;

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

        // --- EXPECTATION STEP ---

        // votes for best patch from NNF Source->Target (completeness) and Target->Source (coherence)

        vote = (float ***)malloc(newtarget->image->height*sizeof(float **));
        for (int i=0 ; i<newtarget->image->height ; ++i ){
            vote[i] = (float **)malloc(newtarget->image->width*sizeof(float *));
            for  (int j=0 ; j<newtarget->image->width ; ++j) {
                vote[i][j] = (float *)calloc(4, sizeof(float));
            }
        }

        ExpectationStep(imp->nnf_TargetToSource, vote, newsource, newtarget, upscaled);
        if( level==1 && emloop==2)
            dumpVote(vote, newtarget);

        // --- MAXIMIZATION STEP ---

        // compile votes and update pixel values
        MaximizationStep(newtarget, vote);

        for (int i=0;i<newtarget->image->height;i++) {
            for (int j=0;j<newtarget->image->width;j++)
                free(vote[i][j]);

            free(vote[i]);
        }
        free(vote);
    }

    //printf("\n");

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

    // patch radius
    imp->radius = radius;

    // working copies
    MaskedImage_P source = imp->initial;
    MaskedImage_P target = NULL;

    // build pyramid of downscaled images
    addEltInpaintingPyramid(imp, source);
    while (/*countMasked(source) > 0 && */source->image->width>radius && source->image->height>radius ) {
        source = downsample(source);
        addEltInpaintingPyramid(imp, source);
    }
    int maxlevel=imp->nbEltPyramid;

    // for each level of the pyramid
    for (int level=maxlevel-1 ; level>0 ; level--) {

        // create Nearest-Neighbor Fields (direct and reverse)
        source = imp->pyramid[level];

        if (level==maxlevel-1) {
            // at first, we use the same image for target and source
            // and use random data as initial guess
            target = copyMaskedImage(source);
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
    const int radius = 4.;
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

    MaskedImage_P output = inpaint_impl(&image, newmask, radius);
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
