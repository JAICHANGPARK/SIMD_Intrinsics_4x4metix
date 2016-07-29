//
//  main.c
//  C_intrinsic__Mul_4x4_Matrix
//
//  Created by PARK JAICHANG on 7/22/16.
//  Copyright © 2016 JAICHANGPARK. All rights reserved.
//
//  4x4 metrix in c and intrinsic 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <emmintrin.h>
#include <x86intrin.h>
#include <immintrin.h>

//Global variable......
struct timeval start,stop;
double Result_time = 0;
int i = 0;

void Matrix4x4C(float *MatrixA, float *MatrixB, float *MatrixDest);
void Matrix4x4Intrinsic(float *MatrixA, float *MatrixB, float *MatrixDest);
double timedifference_msec(struct timeval t0, struct timeval t1);

int main(int argc, const char * argv[]) {
   
    float MatrixA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float MatrixB[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float MatrixDest[16] = {0};
    
    gettimeofday(&start, NULL);
    for (i = 0; i < 10000; i++)
    Matrix4x4C(MatrixA, MatrixB, MatrixDest);
    gettimeofday(&stop, NULL);
    Result_time = timedifference_msec(stop, start);
    printf("4x4 C : executed time : %f in second \n", Result_time);
    printf("%f,%f,%f,%f\n",MatrixDest[0],MatrixDest[1],MatrixDest[2],MatrixDest[3]);
    printf("%f,%f,%f,%f\n",MatrixDest[4],MatrixDest[5],MatrixDest[6],MatrixDest[7]);
    printf("%f,%f,%f,%f\n",MatrixDest[8],MatrixDest[9],MatrixDest[10],MatrixDest[11]);
    printf("%f,%f,%f,%f\n",MatrixDest[12],MatrixDest[13],MatrixDest[14],MatrixDest[15]);
    
   // for (i = 0; i < 10000; i++)
    Matrix4x4Intrinsic(MatrixA, MatrixB, MatrixDest);
    printf("4x4 intrinsic : executed time : in second \n");
    printf("%f,%f,%f,%f\n",MatrixDest[0],MatrixDest[1],MatrixDest[2],MatrixDest[3]);
    printf("%f,%f,%f,%f\n",MatrixDest[4],MatrixDest[5],MatrixDest[6],MatrixDest[7]);
    printf("%f,%f,%f,%f\n",MatrixDest[8],MatrixDest[9],MatrixDest[10],MatrixDest[11]);
    printf("%f,%f,%f,%f\n",MatrixDest[12],MatrixDest[13],MatrixDest[14],MatrixDest[15]);
    
    return 0;
}

void Matrix4x4C(float *MatrixA, float *MatrixB, float *MatrixDest){

    MatrixDest[0] = MatrixA[0] * MatrixB[0] + MatrixA[1] * MatrixB[4] + MatrixA[2] * MatrixB[8] + MatrixA[3] * MatrixB[12];
    MatrixDest[1] = MatrixA[0] * MatrixB[1] + MatrixA[1] * MatrixB[5] + MatrixA[2] * MatrixB[9] + MatrixA[3] * MatrixB[13];
    MatrixDest[2] = MatrixA[0] * MatrixB[2] + MatrixA[1] * MatrixB[6] + MatrixA[2] * MatrixB[10] + MatrixA[3] * MatrixB[14];
    MatrixDest[3] = MatrixA[0] * MatrixB[3] + MatrixA[1] * MatrixB[7] + MatrixA[2] * MatrixB[11] + MatrixA[3] * MatrixB[15];
    
    MatrixDest[4] = MatrixA[4] * MatrixB[0] + MatrixA[5] * MatrixB[4] + MatrixA[6] * MatrixB[8] + MatrixA[7] * MatrixB[12];
    MatrixDest[5] = MatrixA[4] * MatrixB[1] + MatrixA[5] * MatrixB[5] + MatrixA[6] * MatrixB[9] + MatrixA[7] * MatrixB[13];
    MatrixDest[6] = MatrixA[4] * MatrixB[2] + MatrixA[5] * MatrixB[6] + MatrixA[6] * MatrixB[10] + MatrixA[7] * MatrixB[14];
    MatrixDest[7] = MatrixA[4] * MatrixB[3] + MatrixA[5] * MatrixB[7] + MatrixA[6] * MatrixB[11] + MatrixA[7] * MatrixB[15];
    
    MatrixDest[8] = MatrixA[8] * MatrixB[0] + MatrixA[9] * MatrixB[4] + MatrixA[10] * MatrixB[8] + MatrixA[11] * MatrixB[12];
    MatrixDest[9] = MatrixA[8] * MatrixB[1] + MatrixA[9] * MatrixB[5] + MatrixA[10] * MatrixB[9] + MatrixA[11] * MatrixB[13];
    MatrixDest[10] = MatrixA[8] * MatrixB[2] + MatrixA[9] * MatrixB[6] + MatrixA[10] * MatrixB[10] + MatrixA[11] * MatrixB[14];
    MatrixDest[11] = MatrixA[8] * MatrixB[3] + MatrixA[9] * MatrixB[7] + MatrixA[10] * MatrixB[11] + MatrixA[11] * MatrixB[15];
    
    MatrixDest[12] = MatrixA[12] * MatrixB[0] + MatrixA[13] * MatrixB[4] + MatrixA[14] * MatrixB[8] + MatrixA[15] * MatrixB[12];
    MatrixDest[13] = MatrixA[12] * MatrixB[1] + MatrixA[13] * MatrixB[5] + MatrixA[14] * MatrixB[9] + MatrixA[15] * MatrixB[13];
    MatrixDest[14] = MatrixA[12] * MatrixB[2] + MatrixA[13] * MatrixB[6] + MatrixA[14] * MatrixB[10] + MatrixA[15] * MatrixB[14];
    MatrixDest[15] = MatrixA[12] * MatrixB[3] + MatrixA[13] * MatrixB[7] + MatrixA[14] * MatrixB[11] + MatrixA[15] * MatrixB[15];

}

void Matrix4x4Intrinsic(float* MatrixA, float* MatrixB, float* MatrixDest){
    
   // int indexA = 0;
    //int indexB = 0;
    
    __m128 xmmB[4];
    for (i=0; i < 4; i++)
        xmmB[i] = _mm_load_ps((MatrixB + (i*4)));
    
    __m128 xmmR;
    for (i = 0; i < 4; i++) {
      
       
        xmmR = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(MatrixA[i]), xmmB[i]),_mm_add_ps(_mm_mul_ps(_mm_set1_ps(MatrixA[i]), xmmB[i]),
        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(MatrixA[i]),xmmB[i]),_mm_mul_ps(_mm_set1_ps(MatrixA[i]), xmmB[i]))));
        
        _mm_store_ps((MatrixDest+(i*4)), xmmR);
    }
    
}

double timedifference_msec(struct timeval t0, struct timeval t1){
    
    return (double)(t0.tv_usec - t1.tv_usec) / 1000000 + (double)(t0.tv_sec - t1.tv_sec);
}
