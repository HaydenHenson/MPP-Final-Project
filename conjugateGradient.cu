#include <cmath>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
using namespace std;

#define BLOCK_SIZE 256


		//****************************************************************************
              	//*************************** Timer from MPPLABS *****************************
              	//****************************************************************************

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;


void startTime(Timer* timer){
	gettimeofday(&(timer->startTime), NULL);
}
void stopTime(Timer* timer){
	gettimeofday(&(timer->endTime), NULL);
}
float elapsedTime(Timer timer){
	return (float)((timer.endTime.tv_sec - timer.startTime.tv_sec) * 1000.0f + (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1000.0f);
}




__global__ void vecAddKernel(float *x, float *y, float *result, int n)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<n){
	  result[idx] = x[idx] + y[idx];
	}
}//kernel


__global__ void vecDiffKernel(float *x, float *y, float *result, int n)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<n){
	  result[idx] = x[idx] - y[idx];
	}
}//kernel


//Matrix-Vector Multiplication Kernel
__global__ void matrixVecMultKernel(float *A, float *x, float *Ax, int rows, int cols)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	if(row < rows){
	  float sum = 0;

	  for(int col = 0; col<cols; col++){
	     sum += A[row*cols + col] * x[col];
	  }
	  Ax[row] = sum;
	}
}//kernel


__global__ void scalarProdKernel(float *x, float a, float *result, int n)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx<n){
	  result[idx] = a*x[idx];
	}
}//kernel



__global__ void dotProdKernel(float *x, float *y, float *result, int n)
{
	__shared__ float temp[BLOCK_SIZE]; 		//temp vector to hold products
	int idx = blockIdx.x*blockDim.x + threadIdx.x;	//global index of element in vector(s)
	int tid = threadIdx.x; 				//thread index within block
	
	if(idx < n){
	  temp[tid] = x[idx]*y[idx];
	}
	else {
		temp[tid] = 0.0f;
	}

	__syncthreads();

	//Parallel reduction
	for(int stride = blockDim.x/2; stride >0; stride/=2){
	   if(tid<stride){
	     temp[tid] += temp[tid+stride];
	   }
	   __syncthreads();
	}

	if(tid == 0) {
	  atomicAdd(result,temp[0]);
	}
}//kernel


void conjugateGradientCuda(float *A, float *b, float *x, float *r0, float *r1, float *p, float *Ap, float *dotProd_result, float *scalarProd_result, int n)
{
	int grid = (n+BLOCK_SIZE-1) / BLOCK_SIZE;
	dim3 dimGrid(grid,1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);
	

	float alpha, beta, rnorm;
	int k = 0;
	float rtol = 1e-6;
	float dot_r; 
	//float rnorm;

	
	//Compute r0 = b - A*x
	matrixVecMultKernel<<<dimGrid, dimBlock>>>(A, x, Ap, n, n);
	vecDiffKernel<<<dimGrid, dimBlock>>>(b, Ap, r0, n);

	//Compute dotProd(r0,r0)
	cudaMemset(dotProd_result, 0, sizeof(float));
	dotProdKernel<<<dimGrid, dimBlock>>>(r0, r0, dotProd_result, n);
	cudaMemcpy(&dot_r, dotProd_result, sizeof(float), cudaMemcpyDeviceToHost); //copy dotProd(r0,r0);

	//compute ||r0||
	rnorm = sqrt(dot_r);
	cout << "||r0|| = " << rnorm << endl;

	cudaMemcpy(p, r0, n*sizeof(float), cudaMemcpyDeviceToDevice);

	cout << "Iteration " << k << ", Residual Norm: " << rnorm << endl;

	while( (rnorm>rtol) && (k<n) ){

		//calculate Ap = A*p
		matrixVecMultKernel<<<dimGrid, dimBlock>>>(A, p, Ap, n, n);


		//calculate alpha = dotProd(r0, r0) / dot(p, Ap)
		cudaMemset(dotProd_result, 0, sizeof(float));
		dotProdKernel<<<dimGrid, dimBlock>>>(p, Ap, dotProd_result, n);
		cudaMemcpy(&alpha, dotProd_result, sizeof(float), cudaMemcpyDeviceToHost);
		alpha = dot_r / alpha;


		//update x = x + alpha*p
		scalarProdKernel<<<dimGrid, dimBlock>>>(p, alpha, scalarProd_result, n);	// scalarProd_result = alpha*p;
		vecAddKernel<<<dimGrid, dimBlock>>>(x, scalarProd_result, x, n);


		//update r1 = r0 - alpha*Ap
		scalarProdKernel<<<dimGrid, dimBlock>>>(Ap, alpha, scalarProd_result, n);	//scalarProd_result = alpha*Ap;
		vecDiffKernel<<<dimGrid, dimBlock>>>(r0, scalarProd_result, r1, n);
		

		//calculate beta = dot(r1,r1)  dot(r0,r0)
		cudaMemset(dotProd_result, 0, sizeof(float));		
		dotProdKernel<<<dimGrid, dimBlock>>>(r1, r1, dotProd_result, n);
		cudaMemcpy(&beta, dotProd_result, sizeof(float), cudaMemcpyDeviceToHost);
		beta = beta / dot_r;


		//update p = r1 + beta*p
		scalarProdKernel<<<dimGrid, dimBlock>>>(p, beta, scalarProd_result, n);		//scalarProd_result = beta*p;
		vecAddKernel<<<dimGrid, dimBlock>>>(r1, scalarProd_result, p, n);


		//update r0
		cudaMemcpy(r0, r1, n*sizeof(float), cudaMemcpyDeviceToDevice);


		//calculate rnorm
		cudaMemset(dotProd_result, 0, sizeof(float));			
		dotProdKernel<<<dimGrid, dimBlock>>>(r0, r0, dotProd_result, n); 		//compute dotProd(r0,r0)
		cudaMemcpy(&dot_r, dotProd_result, sizeof(float), cudaMemcpyDeviceToHost); 	//copy dotProd(r0,r0);

		//compute ||r0||
		rnorm = sqrt(dot_r);

		//increase k
		k++;

		cout << "Iteration " << k << ": ||r|| = " << rnorm << endl;
	
	}
} //CG CUDA



int main()
{
	Timer timer;
	cudaError_t cuda_ret;

	//Initiate host variables
	int n = 5000;	 	//Size of matrix & vectors
	float A[n*n];		//Define A
	float b[n];		//Define b
	float x[n] = {0.0f};	//Define x

	cout << "A = " <<  n << "x" << n << " matrix \n";


	//***************************************************************************
	//********************************* Upload A ********************************
	//***************************************************************************

	string fileA = "/home/hensonh/finalProj/Code/A5000.txt";
	ifstream infileA(fileA);
	if(!infileA.is_open()) {
	  cerr << "Error opening A file" << endl;
	  return 1;
	}
	
	for(int row=0; row<n; row++) {
	   for(int col=0; col<n; col++) {
	      infileA >> A[row*n + col];
	   }
	}

	infileA.close();

	//Print A
	/*cout << "A = \n";
	for(int r=0; r<n; r++){
	   for(int c=0; c<n; c++){
	//compute ||r0||
	rnorm = sqrt(dot_r);
	cout << "||r0|| = " << rnorm << endl;*/
		     
	
	//****************************************************************************
	//******************************** Upload b **********************************
	//****************************************************************************
	
	string fileb = "/home/hensonh/finalProj/Code/b5000.txt";
	ifstream infileb(fileb);

	if(!infileb.is_open()) {
	  cout << "Error opening fileb" << endl;
	  return 1;
	}

	for(int i=0; i<n; i++){
	   infileb >> b[i];
	}
	
	infileb.close();


	//********************************************************************************
	//************************* Allocate memory on device ****************************
	//********************************************************************************

	printf("Allocating device variables..."); fflush(stdout);
	startTime(&timer);
	
	float *A_d, *b_d, *x_d, *r0_d, *r1_d, *p_d, *Ap_d, *dotProd_result_d, *scalarProd_result_d;
	cuda_ret = cudaMalloc(&A_d, n*n*sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocate A_d device memory" << endl;
	cuda_ret = cudaMalloc(&b_d, n*sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocate b_d device memory" << endl;
	cuda_ret = cudaMalloc(&x_d, n*sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocate x_d device memory" << endl;
	//----------------PROBABLY DO NOT NEED THESE LOWER ONES---------------------------
	cuda_ret = cudaMalloc(&r0_d, n*sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocate r0_d device memory" << endl;
	cuda_ret = cudaMalloc(&r1_d, n*sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocate r1_d device memory" << endl;
	cuda_ret = cudaMalloc(&p_d, n*sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocate p_d device memory" << endl;
	cuda_ret = cudaMalloc(&Ap_d, n*sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocate Ap_d device memory" << endl;
	cuda_ret = cudaMalloc(&dotProd_result_d, sizeof(float));
	if(cuda_ret != cudaSuccess) cerr << "Unable to allocated dotProd device memory" << endl;
	cuda_ret = cudaMalloc(&scalarProd_result_d, n*sizeof(float));

	cudaDeviceSynchronize();
	stopTime(&timer); printf("%f s\n", elapsedTime(timer)/1000);


	//*********************************************************************************
	//************************* Copy data form host to device *************************
	//*********************************************************************************

	printf("Copying data from host to device..."); fflush(stdout);
	startTime(&timer);

	cuda_ret = cudaMemcpy(A_d, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) cerr << "Unable to copy A to device" << endl;
	cudaMemcpy(b_d, b, n*sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) cerr << "Unable to copy b to device" << endl;
	cudaMemcpy(x_d, x, n*sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) cerr << "Unable to copy x to device" << endl;

	cudaDeviceSynchronize();
	stopTime(&timer); printf("%f s\n", elapsedTime(timer)/1000);

	//*********************************************************************************
	//********************** Call the cojugate gradient function **********************
	//*********************************************************************************


	startTime(&timer);
	conjugateGradientCuda(A_d, b_d, x_d, r0_d, r1_d, p_d, Ap_d, dotProd_result_d, scalarProd_result_d, n);

	cudaDeviceSynchronize();
	printf("Launching Kernels..."); fflush(stdout);
	stopTime(&timer); printf("%f s\n", elapsedTime(timer)/1000);

	//*********************************************************************************
	//************************* Copy data from device to host *************************
	//*********************************************************************************
	
	printf("Copying data from device to host..."); fflush(stdout);
	startTime(&timer);

	cuda_ret = cudaMemcpy(x, x_d, n*sizeof(float), cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) cerr << "Unable to copy x back to host" << endl;

	cudaDeviceSynchronize();
	stopTime(&timer); printf("%f s\n", elapsedTime(timer)/1000);
	
	//*********************************************************************************
	//********************************** Free memory **********************************
	//*********************************************************************************
	cudaFree(A_d);
	cudaFree(b_d);
	cudaFree(x_d);
	cudaFree(r0_d);
	cudaFree(r1_d);
	cudaFree(p_d);
	cudaFree(Ap_d);
	cudaFree(dotProd_result_d);
	cudaFree(scalarProd_result_d);

	//Output solution vector x
	cout << "x = {";
	if(n>=10) {
	  for(int i = 0; i < 8; i++){
	     cout << x[i] << ", ";
	  }
	  cout << x[9] << "}" << endl;
	} else {
		for(int i=0; i<n-1; i++) {
		   cout << x[i] << ", ";
		}
		cout << x[n-1] << "}" << endl;
	}
	return 0;

}

