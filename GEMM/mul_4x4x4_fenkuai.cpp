#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define A(h,l,w) a[h * w + l]
#define B(h,l,w) b[h * w + l]
#define C(h,l,w) c[h * w + l]

#define MM  100   
#define NN 130
#define KK   40

#define LOOP 10000


void randArray(float *array, int H, int L) {
	if(array == NULL || H < 2 || L < 2) return;

	for(int h=0; h < H; h++) {
		for(int l=0; l < L; l++){
			array[h*L + l] = rand() * 0.33;
		}
	}
}


void process4x4(float *lda, float *ldb, double *c, int M, int N, int K, int indexM, int indexN) {


	float *tempA0 = (lda);
	float *tempA1 = (lda+K);
	float *tempA2 = (lda+K*2);
	float *tempA3 = (lda+K*3);

	float *tempB0 = ldb;
	float *tempB1 = ldb+1;
	float *tempB2 = ldb+2;
	float *tempB3 = ldb+3;


	double *tempC00 = &C(indexM,indexN,N);
	double *tempC01 = tempC00 + 1;
	double *tempC02 = tempC00 + 2;
	double *tempC03 = tempC00 + 3;


	double *tempC10 = &C((indexM+1),indexN,N);
	double *tempC11 = tempC10 + 1;
	double *tempC12 = tempC10 + 2;
	double *tempC13 = tempC10 + 3;


	double *tempC20 = &C((indexM +2),indexN,N);
	double *tempC21 = tempC20 + 1;
	double *tempC22 = tempC20 + 2;
	double *tempC23 = tempC20 + 3;

	double *tempC30 = &C((indexM + 3),indexN,N);
	double *tempC31 = tempC30 + 1;
	double *tempC32 = tempC30 + 2;
	double *tempC33 = tempC30 + 3;

	
	register double c00 = 0;
	register double c01 = 0;
	register double c02 = 0;
	register double c03 = 0;
	register double c10 = 0;
	register double c11 = 0;
	register double c12 = 0;
	register double c13 = 0;
	register double c20 = 0;
	register double c21 = 0;
	register double c22 = 0;
	register double c23 = 0;
	register double c30 = 0;
	register double c31 = 0;
	register double c32 = 0;
	register double c33 = 0;

	for(int indexK = 0; indexK < K; indexK ++) {
			
		float smallTempB0 = *((indexK * N) + tempB0);
		float smallTempB1 = *((indexK * N) + tempB1);
		float smallTempB2 = *((indexK * N) + tempB2);
		float smallTempB3 = *((indexK * N) + tempB3);
		
	 	float smallTempA0 = (*(tempA0 + indexK));
		float smallTempA1 = (*(tempA1 + indexK));
		float smallTempA2 = (*(tempA2 + indexK));
		float smallTempA3 = (*(tempA3 + indexK));

		c00 += smallTempA0 * (smallTempB0);
		c01 += smallTempA0 * (smallTempB1);
		c02 += smallTempA0 * (smallTempB2);
		c03 += smallTempA0 * (smallTempB3);
	
		c10 += smallTempA1 * (smallTempB0);
		c11 += smallTempA1 * (smallTempB1);
		c12 += smallTempA1 * (smallTempB2);
		c13 += smallTempA1 * (smallTempB3);

		c20 += smallTempA2 * (smallTempB0);
		c21 += smallTempA2 * (smallTempB1);
		c22 += smallTempA2 * (smallTempB2);
		c23 += smallTempA2 * (smallTempB3);


		c30 += smallTempA3 * (smallTempB0);
		c31 += smallTempA3 * (smallTempB1);
		c32 += smallTempA3 * (smallTempB2);
		c33 += smallTempA3 * (smallTempB3);
	}
   
    *tempC00 += c00; 
    *tempC01 += c01;
    *tempC02 += c02;
    *tempC03 += c03;
    
    (*tempC10) += c10;
    (*tempC11) += c11;
    (*tempC12) += c12;
    (*tempC13) += c13;
    
    (*tempC20) += c20;
    (*tempC21) += c21;
    (*tempC22) += c22;
    (*tempC23) += c23;
    
    (*tempC30) += c30;
    (*tempC31) += c31;
    (*tempC32) += c32;
    (*tempC33) += c33;

}	
void process(float *a, float *b, double *c, int M, int N, int K) {
	for(int m = 0; m < M; m += 8) {
		for(int n=0; n < N; n += 8) {
			for (int mi = 0; mi < 2;mi ++) {
     			 for (int ni = 0; ni < 2; ni++) {
					float *tempA = &A(m+mi*4,0,K);
					float *tempB = &B(0,n+ni*4,N);
					process4x4(tempA, tempB, c, M, N, K, m, n);
				}
			}
		}
	}
}

int main() {

	for(int loop=100; loop < 1200; loop += 100) {
		int mm=loop;
		int nn=loop;
		int kk=loop;

		float *a = (float *) malloc(sizeof(float) * mm * mm);
		float *b = (float *) malloc(sizeof(float) * kk * nn);
		double *c = (double *) malloc(sizeof(double) * mm * nn);
	
		randArray(a, mm, kk);
		randArray(b, kk, nn);
		struct timeval start;
		gettimeofday(&start, NULL);

		process(a, b, c, mm, nn, kk);
	
		struct timeval end;
		gettimeofday(&end, NULL);

		float duration = (float)(end.tv_sec - start.tv_sec)  + (float)(end.tv_usec - start.tv_usec)/1000000.0;
		long double allops = 2 * mm * nn * kk;
		double flops = (double)(allops/1000000000.0)/(double)duration;
		printf("[%d] gfloaps is %lf\n",loop,flops);
	}
}
