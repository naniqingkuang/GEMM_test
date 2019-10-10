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

    __m128 c0,c1,c2,c3;
    __m128 b0,b1,b2,b3;
	
    __m128 a;
    __m128 tempAdd;
    __m128 tempMul;
	for(int indexK = 0; indexK < K; indexK += 4) {

			
		float* smallTempB00 = (((indexK+0) * N) + tempB0);
		float* smallTempB10 = (((indexK+1) * N) + tempB0);
		float* smallTempB20 = (((indexK+2) * N) + tempB0);
		float* smallTempB30 = (((indexK+3) * N) + tempB0);

		float* smallTempA00 = ((tempA0 + indexK));
		float* smallTempA10 = ((tempA1 + indexK));
		float* smallTempA20 = ((tempA2 + indexK));
		float* smallTempA30 = ((tempA3 + indexK));

	 	
		float* smallTempA01 = ((tempA0 + indexK+1));
		float* smallTempA11 = ((tempA1 + indexK+1));
		float* smallTempA21 = ((tempA2 + indexK+1));
		float* smallTempA31 = ((tempA3 + indexK+1));
	 	
		float* smallTempA02 = ((tempA0 + indexK+2));
		float* smallTempA12 = ((tempA1 + indexK+2));
		float* smallTempA22 = ((tempA2 + indexK+2));
		float* smallTempA32 = ((tempA3 + indexK+2));
	 	
		float* smallTempA03 = ((tempA0 + indexK+3));
		float* smallTempA13 = ((tempA1 + indexK+3));
		float* smallTempA23 = ((tempA2 + indexK+3));
		float* smallTempA33 = ((tempA3 + indexK+3));

        b0 = _mm_load_ps(smallTempB00);
        b1 = _mm_load_ps(smallTempB10);
        b2 = _mm_load_ps(smallTempB20);
        b3 = _mm_load_ps(smallTempB30);

        a = _mm_load1_ps(smallTempA00);
        tempAdd = _mm_mul_ps(a,b0);
        c0 = _mm_add_ps(c0,tempAdd);


        a = _mm_load1_ps(smallTempA01);
        tempAdd = _mm_mul_ps(a,b1);
        c0 = _mm_add_ps(c0,tempAdd);

        a = _mm_load1_ps(smallTempA02);
        tempAdd = _mm_mul_ps(a,b2);
        c0 = _mm_add_ps(c0,tempAdd);

        a = _mm_load1_ps(smallTempA03);
        tempAdd = _mm_mul_ps(a,b3);
        c0 = _mm_add_ps(c0,tempAdd);

        //1
        a = _mm_load1_ps(smallTempA10);
        tempAdd = _mm_mul_ps(a,b0);
        c1 = _mm_add_ps(c1,tempAdd);


        a = _mm_load1_ps(smallTempA11);
        tempAdd = _mm_mul_ps(a,b1);
        c1 = _mm_add_ps(c1,tempAdd);

        a = _mm_load1_ps(smallTempA12);
        tempAdd = _mm_mul_ps(a,b2);
        c1 = _mm_add_ps(c1,tempAdd);

        a = _mm_load1_ps(smallTempA13);
        tempAdd = _mm_mul_ps(a,b3);
        c1 = _mm_add_ps(c1,tempAdd);


        //2
        a = _mm_load1_ps(smallTempA20);
        tempAdd = _mm_mul_ps(a,b0);
        c2 = _mm_add_ps(c2,tempAdd);


        a = _mm_load1_ps(smallTempA21);
        tempAdd = _mm_mul_ps(a,b1);
        c2 = _mm_add_ps(c2,tempAdd);

        a = _mm_load1_ps(smallTempA22);
        tempAdd = _mm_mul_ps(a,b2);
        c2 = _mm_add_ps(c2,tempAdd);

        a = _mm_load1_ps(smallTempA23);
        tempAdd = _mm_mul_ps(a,b3);
        c2 = _mm_add_ps(c2,tempAdd);


        //3
        a = _mm_load1_ps(smallTempA30);
        tempAdd = _mm_mul_ps(a,b0);
        c3 = _mm_add_ps(c3,tempAdd);


        a = _mm_load1_ps(smallTempA31);
        tempAdd = _mm_mul_ps(a,b1);
        c3 = _mm_add_ps(c3,tempAdd);

        a = _mm_load1_ps(smallTempA32);
        tempAdd = _mm_mul_ps(a,b2);
        c3 = _mm_add_ps(c3,tempAdd);

        a = _mm_load1_ps(smallTempA33);
        tempAdd = _mm_mul_ps(a,b3);
        c3 = _mm_add_ps(c3,tempAdd);
                                
                                
                               
		
	}
   
                                _mm_store_ps(tempC00,c0);
                                _mm_store_ps(tempC10,c1);
                                _mm_store_ps(tempC20,c2);
                                _mm_store_ps(tempC30,c3);


}

void process2(float *a, float *b, float *c, int M, int N, int K) {
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

void process1(float *a, float *b, float *c, int M, int N, int K) {
    for(int m = 0; m < M; m += 4) {
        float *tempA = &A(m,0,K);
        for(int n=0; n < N; n += 4) {
            float *tempB = &B(0,n,N);
            process4x4(tempA, tempB, c, M, N, K, m, n);
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

		process1(a, b, c, mm, nn, kk);
	
		struct timeval end;
		gettimeofday(&end, NULL);

		float duration = (float)(end.tv_sec - start.tv_sec)  + (float)(end.tv_usec - start.tv_usec)/1000000.0;
		long double allops = 2 * mm * nn * kk;
		float flops = (float)(allops/1000000000.0)/(float)duration;
		printf("[%d] gfloaps is %lf\n",loop,flops);
	}
}
