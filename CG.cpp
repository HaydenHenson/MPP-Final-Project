#include <cmath>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
using namespace std;


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






//Function that reads matrix file
std::vector<std::vector<float>> readMatrix(const std::string &filename, size_t rows, size_t cols)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<float>> A(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!(infile >> A[i][j])) {
                throw std::runtime_error("Error reading matrix entry at row "
                                         + std::to_string(i) + ", col " + std::to_string(j));
            }
        }
    }
    return A;
}


//Calculate the difference between two vectors
std::vector<float> vecDiff(const std::vector<float> &x, const std::vector<float> &y)
{
	int n = x.size();
	std::vector<float> result(n);
	for(int i=0; i<n; i++){
	    result[i] =  x[i] - y[i];
	}
	return result;
}


//Calculate the sum between two vectors
std::vector<float> vecAdd(const std::vector<float> &x, const std::vector<float> &y)
{
	int n = x.size();
	std::vector<float> result(n);
	for(int i=0; i<n; i++){
	    result[i] =  x[i] + y[i];
	}
	return result;
}


//Multiply a vector 'x' by a scalar 'a'
std::vector<float> scalarProd(std::vector<float> &x, const float a)
{
	int n = x.size();
	std::vector<float> result(n);
	for(int i=0; i<n; i++){
	   result[i] = a*x[i];
	}
	return result;

}


//Matrix-vector multiplication
std::vector<float> matrixVecMult(const std::vector<std::vector<float>> &A,const std::vector<float> &x)
{
	int n = A.size();
	int m = A[0].size();
	std:: vector <float> Ax(n);

	for(int i=0; i<n; i++){
	   float sum = 0.0;
	   for(int j=0; j<m; j++){
	      sum += A[i][j]*x[j];
	   } 
	   Ax[i] = sum;
	}
	return Ax;
}


//Vector dot product
float dotProd(const std::vector<float> &x,const std::vector<float> &y)
{
	int n = x.size();
	float sum = 0;

	for(int i=0; i<n; i++){
	   sum += x[i]*y[i];
	}
	return sum;
}


//Calculate the residual vector
std::vector<float> resVec(const std::vector<std::vector<float>> &A, const std::vector<float> &b, const std::vector<float> &x)
{
	int n = A.size();
	std::vector<float> r(n);
	r = vecDiff(b, matrixVecMult(A,x));
	return r;
}


//Print vector
void printVec(const std::vector<float> &x)
{
  int m;
  cout << "{";
  if(x.size() > 10){
    m = 9;
  } else {
       m = x.size();
  }
  
	for(int i = 0; i<m-1; i++){
	   cout << x[i] << ",";
	}
	cout << x[m-1] << "}" << endl;
}


//The Conjugate Gradient Method
std::vector<float> conjugateGradientSerial(const std::vector<std::vector<float>> &A, const std::vector<float> &b, std::vector<float> &x)
{
	int n = A.size();
	float rtol = 1e-6; // residual tolerance for convergence

	//Initialize variables
	std::vector<float> r0;
	std::vector<float> r1;

	r0 = resVec(A,b,x);
	std:: vector<float> p = r0;
  float dot_r = dotProd(r0,r0);
  float rnorm = sqrt(dot_r);


	int k = 0;
	float alpha;
	float beta;
	std::vector<float> Ap;

  cout << "||r0||=" << rnorm << ", k=" << k << endl;
	while(rnorm>rtol && k < n){
		Ap = matrixVecMult(A,p);                   //Calculate Ap
    alpha = dot_r / dotProd(p,Ap);             //Calculate alpha = dot(r0,r0) / dot(p,Ap)
		x = vecAdd(x,scalarProd(p,alpha));         //Update x = x + alpha*p
		r1 = vecDiff(r0,scalarProd(Ap,alpha));     //Update r1 = r0 - alpha*Ap
    beta = dotProd(r1,r1) / dot_r;             //Calcluate beta = dot(r1,r1) / dot(r0,r0)
		p = vecAdd(r1,scalarProd(p,beta));         //Update p = r1 + beta*p
		r0 = r1;                                   //Update r0
    dot_r = dotProd(r0,r0);                    
    rnorm = sqrt(dot_r);                       //Calculate ||r0||
		k++;
    cout << "||r||=" << rnorm << ", k=" << k << endl;
	}
  
	return  x;

}





int main()
{

  Timer timer;
  
	std:: vector<std:: vector<float>> A;
	std:: vector <float> b;
 
	


	//***************************************************************************
	//********************************* Upload A ********************************
	//***************************************************************************
    try {
        size_t rows = 5000;  // set known number of rows
        size_t cols = 5000;  // set known number of columns

        std::string infileA = "/home/hensonh/finalProj/Code/A5000.txt";
        A = readMatrix(infileA, rows, cols);

        std::cout << "Matrix read successfully: " << A.size() << " x " << A[0].size() << "\n";

    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
 
	//****************************************************************************
	//******************************** Upload b **********************************
	//****************************************************************************
  std::string fileb = "/home/hensonh/finalProj/Code/b5000.txt";
	std::ifstream infileb(fileb);

	if(!infileb.is_open()) {
	  cout << "Error opening fileb" << endl;
	  return 1;
	}
 
  float val;

	while(infileb >> val) {
   b.push_back(val);
  }
	
	infileb.close();

  //****************************************************************************
	//******************************** Define x **********************************
	//****************************************************************************
  std:: vector <float> x(A.size(),0.0);



  //****************************************************************************
	//********************************* Run CG ***********************************
	//****************************************************************************
  printf("Running Serial Conjugate Gradient Method...\n"); fflush(stdout);
  startTime(&timer);    //start taking time
  
  //Solve for x using the conjugate gradient method
	x = conjugateGradientSerial(A,b,x);
 
  printf("Serial Conjugate Gradient Method: "); fflush(stdout);
  stopTime(&timer); printf("%f s\n", elapsedTime(timer)/1000);  //stop taking time
 
  cout << "x = ";
	printVec(x);

}
