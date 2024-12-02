#include "cufft_example.h"

//Based on example found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example

__device__ Complex complexScaleMult(Complex a, Complex b, float scalar)
{
    // Create a variable of type Complex named c
    Complex c;

    // Calculate the x value for c by scalar * (a.x * b.x)
    c.x = scalar * (a.x * b.x - a.y * b.y);  // Adjusting for the real part of complex multiplication

    // Calculate the y value for c by scalar * (a.y * b.y)
    c.y = scalar * (a.x * b.y + a.y * b.x);  // Adjusting for the imaginary part of complex multiplication

    return c;
}
__global__ void complexProcess(Complex *a, Complex *b, Complex *c, int size, float scalar)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < size) {
        c[threadId] = complexScaleMult(a[threadId], b[threadId], scalar);

    }
}

__host__ std::tuple<int, int> parseCommandLineArguments(int argc, char** argv) 
{
    // Default value for N is 16
    int N = 16;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == 'n' && argv[i+1]) {
            // Correctly convert the next argument to an integer
            N = (int)strtol(argv[++i], NULL, 10);
        }
    }

    // Calculate SIZE as N squared
    int SIZE = N * N;

    return {N, SIZE};
}


__host__ Complex *generateComplexPointer(int SIZE)
{
    // Allocate memory for SIZE Complex elements
    Complex *complex = new Complex[SIZE];

    // Populate properties x and y of each Complex object at index i
    for (int i = 0; i < SIZE; i++) {
        complex[i].x = 2.0f;  // Setting the x component to 2
        complex[i].y = 3.0f;  // Setting the y component to 3
    }

    return complex;
}


__host__ void printComplexPointer(Complex *complex, int N)
{
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << complex[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;
}
__host__ cufftComplex *generateCuFFTComplexPointerFromHostComplex(int mem_size, Complex *hostComplex)
{
    // Allocate memory for cufftComplex array on the host
    cufftComplex *outputComplex = new cufftComplex[mem_size];

    // Populate the cufftComplex array from the provided hostComplex array
    for (int i = 0; i < mem_size; i++) {
        outputComplex[i].x = hostComplex[i].x;
        outputComplex[i].y = hostComplex[i].y;
    }

    return outputComplex;
}
__host__ cufftHandle transformFromTimeToSignalDomain(int N, cufftComplex *d_a, cufftComplex *d_b, cufftComplex *d_c)
{
    cufftHandle plan;
    int n[2] = {N, N}; // Define the dimensions of the data

    // Create a 2D FFT plan for complex to complex transformation
    if (cufftPlan2d(&plan, N, N, CUFFT_C2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return 0; // Return an invalid handle on failure
    }

    // Execute Complex to Complex Forward Transformation for d_a
    if (cufftExecC2C(plan, d_a, d_a, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C forward failed for d_a");
        cufftDestroy(plan);
        return 0; // Clean up and return on failure
    }

    // Execute Complex to Complex Forward Transformation for d_b
    if (cufftExecC2C(plan, d_b, d_b, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C forward failed for d_b");
        cufftDestroy(plan);
        return 0; // Clean up and return on failure
    }

    // Execute Complex to Complex Forward Transformation for d_c
    if (cufftExecC2C(plan, d_c, d_c, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C forward failed for d_c");
        cufftDestroy(plan);
        return 0; // Clean up and return on failure
    }

    printf("Performing Forward Transformation of a, b, and c\n");

    // Return the cufftHandle for later use
    return plan;
}
__host__ Complex *transformFromSignalToTimeDomain(cufftHandle plan, int SIZE, cufftComplex *d_c)
{
    // Allocate memory on the host to store the results
    Complex *results = new Complex[SIZE];

    // Perform the Complex to Complex INVERSE transformation using the passed-in plan and d_c
    if (cufftExecC2C(plan, d_c, d_c, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C inverse failed\n");
        delete[] results;  // Clean up the allocated memory before returning
        return nullptr;
    }

    printf("Transforming signal back with cufftExecC2C\n");

    // Perform memory copy from d_c (device) into results (host)
    cudaError_t cudaStatus = cudaMemcpy(results, d_c, SIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMemcpy failed with status %s\n", cudaGetErrorString(cudaStatus));
        delete[] results;  // Clean up the allocated memory before returning
        return nullptr;
    }

    // Scale down each element in the result by SIZE to normalize after the inverse FFT
    for (int i = 0; i < SIZE; i++) {
        results[i].x /= SIZE;
        results[i].y /= SIZE;
    }

    return results;
}


int main(int argc, char** argv)
{
    auto[N, SIZE] = parseCommandLineArguments(argc, argv);

    // Host memory allocation and initialization
    Complex *a = generateComplexPointer(SIZE);
    Complex *b = generateComplexPointer(SIZE);
    Complex *c = generateComplexPointer(SIZE);

    cout << "Input random data a:" << endl;
    printComplexPointer(a, N);
    cout << "Input random data b:" << endl;
    printComplexPointer(b, N);

    // Device memory allocation
    cufftComplex *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, SIZE * sizeof(cufftComplex));
    cudaMalloc((void**)&d_b, SIZE * sizeof(cufftComplex));
    cudaMalloc((void**)&d_c, SIZE * sizeof(cufftComplex));

    // Copy data from host to device
    cudaMemcpy(d_a, a, SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // FFT plan and execution
    cufftHandle plan = transformFromTimeToSignalDomain(N, d_a, d_b, d_c);

    // Kernel launch with adjusted block and grid configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    int scalar = (rand() % 5) + 1;
    cout << "Scalar value: " << scalar << endl;
    complexProcess <<< blocksPerGrid, threadsPerBlock >>>(d_a, d_b, d_c, SIZE, scalar);
    cudaDeviceSynchronize();

    // Transform back and copy results to host
    Complex *results = transformFromSignalToTimeDomain(plan, SIZE, d_c);
    for (int i = 0; i < SIZE; i++) {
        results[i].x /= SIZE;  // Scale the result to adjust for the inverse FFT
        results[i].y /= SIZE;
    }

    cout << "Output data c: " << endl;
    printComplexPointer(results, N);

    // Clean up
    delete[] results;
    delete[] a;
    delete[] b;
    delete[] c;
    cufftDestroy(plan);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
