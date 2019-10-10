#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <xmmintrin.h>

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


void process4x4(float *lda, float *ldb, float *c, int M, int N, int K, int indexM, int indexN) {


	float *tempA0 = (lda);
	float *tempA1 = (lda+K);
	float *tempA2 = (lda+K*2);
	float *tempA3 = (lda+K*3);

	float *tempB0 = ldb;
	float *tempB1 = ldb+1;
	float *tempB2 = ldb+2;
	float *tempB3 = ldb+3;


	float *tempC00 = &C(indexM,indexN,N);
	float *tempC01 = tempC00 + 1;
	float *tempC02 = tempC00 + 2;
	float *tempC03 = tempC00 + 3;


	float *tempC10 = &C((indexM+1),indexN,N);
	float *tempC11 = tempC10 + 1;
	float *tempC12 = tempC10 + 2;
	float *tempC13 = tempC10 + 3;


	float *tempC20 = &C((indexM +2),indexN,N);
	float *tempC21 = tempC20 + 1;
	float *tempC22 = tempC20 + 2;
	float *tempC23 = tempC20 + 3;

	float *tempC30 = &C((indexM + 3),indexN,N);
	float *tempC31 = tempC30 + 1;
	float *tempC32 = tempC30 + 2;
	float *tempC33 = tempC30 + 3;

	
	register float c00 = 0;
	register float c01 = 0;
	register float c02 = 0;
	register float c03 = 0;
	register float c10 = 0;
	register float c11 = 0;
	register float c12 = 0;
	register float c13 = 0;
	register float c20 = 0;
	register float c21 = 0;
	register float c22 = 0;
	register float c23 = 0;
	register float c30 = 0;
	register float c31 = 0;
	register float c32 = 0;
	register float c33 = 0;

    __m128 b;
    __m128 a;
    __m128 temp,temp2;
    __m128 c0,c1,c2,c3;
	for(int indexK = 0; indexK < K; indexK ++) {
        
        b = _mm_load_ps(((indexK * N) + tempB0));
        

        a = _mm_load1_ps(tempA0 + indexK);
        temp = _mm_mul_ps(a,b);
        c0 = _mm_add_ps(temp,c0);
        
        a = _mm_load1_ps(tempA1 + indexK);
        temp = _mm_mul_ps(a,b);
        c1 = _mm_add_ps(temp,c1);
        
        a = _mm_load1_ps(tempA2 + indexK);
        temp = _mm_mul_ps(a,b);
        c2 = _mm_add_ps(temp,c2);
        
        a = _mm_load1_ps(tempA3 + indexK);
        temp = _mm_mul_ps(a,b);
        c3 = _mm_add_ps(temp,c3);
        
	}
    _mm_store_ps(tempC00,c0);
    _mm_store_ps(tempC10,c1);
    _mm_store_ps(tempC20,c2);
    _mm_store_ps(tempC30,c3);


}	
void process(float *a, float *b, float *c, int M, int N, int K) {
	for(int m = 0; m < M; m += 8) {
		for(int n=0; n < N; n += 8) {
             for (int mi = 0; mi < 2;mi ++) {
                  float *tempA = &A(m+mi*4,0,K);
                  for (int ni = 0; ni < 2; ni++) {
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
		float *c = (float *) malloc(sizeof(float) * mm * nn);
	
		randArray(a, mm, kk);
		randArray(b, kk, nn);
		struct timeval start;
		gettimeofday(&start, NULL);

		process(a, b, c, mm, nn, kk);
	
		struct timeval end;
		gettimeofday(&end, NULL);

		float duration = (float)(end.tv_sec - start.tv_sec)  + (float)(end.tv_usec - start.tv_usec)/1000000.0;
		long double allops = 2 * mm * nn * kk;
		float flops = (float)(allops/1000000000.0)/(float)duration;
		printf("[%d] gfloaps is %lf\n",loop,flops);
	}
}
