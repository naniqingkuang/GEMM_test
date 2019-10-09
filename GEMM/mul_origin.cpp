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

void process(float *a, float *b, double *c, int M, int N, int K) {
	for(int m = 0; m < M; m += 1) {
		for(int n=0; n < N; n += 1) {
			for(int k=0; k < K; k++){
				C(m,n,N) += A(m,k,K) * B(k,n,N);
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
