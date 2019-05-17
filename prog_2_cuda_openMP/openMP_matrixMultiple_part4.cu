#include <stdio.h>

#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>



typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

// Multiple Matrix C = A * B by using GPU
void
openMP_matrixMulGPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    #pragma omp target 
    {
        
        #pragma omp parallel for
        for (unsigned int i = 0; i < hA; ++i)
            #pragma omp parallel for
            for (unsigned int j = 0; j < wB; ++j)
            {
                double sum = 0;
                #pragma omp parallel for
                for (unsigned int k = 0; k < wA; ++k)
                {
                    double a = A[i * wA + k];
                    double b = B[k * wB + j];
                    sum += a * b;
                }

                C[i * wB + j] = (float)sum;
            }
    }
    
}

void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{

    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}


// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}


int main(int argc, char const *argv[])
{
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    sMatrixSize matrix_size;
    cudaError_t error;
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        error = cudaSetDevice(devID);

        if (error != cudaSuccess)
        {
            printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }


    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // the range of iSizemultiple should be less than 10 and more than 1
    int iSizeMultiple = 5;
    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);



    // use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    //! Run a simple test matrix multiply using CUBLAS
    ////////////////////////////////////////////////////////////////////////////////

    // set seed for rand()
    srand(2006);

    // Initialzie Matrix A, B, C
    float *d_A, *d_B, *d_C;
    //Matrix A
    printf("Create Matrix A.....\n");
    unsigned int size_A = matrix_size.uiHA * matrix_size.uiWA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    randomInit(h_A, size_A);
    printf("done.\n");

    //Matrix B
    printf("Create Matrix B.....\n");
    unsigned int size_B = matrix_size.uiHB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    randomInit(h_B,size_B);
    printf("done.\n");

    //Matrix C
    printf("Create Matrix C.....\n");
    unsigned int size_C = matrix_size.uiHC * matrix_size.uiWC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C = (float *)malloc(mem_size_C);
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
    printf("done.\n");



    //Calculate the number of blocks and threads
    dim3 threads(block_size,block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);
    // dim3 grid(matrix_size.uiWB/block_size, matrix_size.uiHA/block_size);

    {
        // Initialize CUDA for timing
        cudaEvent_t cuda_start, cuda_stop;
        cublasHandle_t handle;

        checkCudaErrors(cublasCreate(&handle));

        //Perform warmup operation with cublas
        //checkCudaErrors(matrixMulKernal<<<grid,threads>>>(d_C, d_A, d_C, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB));

        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&cuda_start));
        checkCudaErrors(cudaEventCreate(&cuda_stop));

        // Record the start event
        printf("Time recording begin....\n");
        checkCudaErrors(cudaEventRecord(cuda_start, NULL));

        //Calculate Matrix C on GPU
        int nIter = 30;
        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            openMP_matrixMulGPU(h_C,h_A,h_B,matrix_size.uiHA,matrix_size.uiWA,matrix_size.uiWB);


        }

        printf("Time recording done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(cuda_stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(cuda_stop));

        float msecTotal = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, cuda_start, cuda_stop));

        // Compute and print the performance
        
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "\n\nPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // Get the result from Device
        printf("Receiving the result from GPU....\n");
        printf("done.\n");
        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }

    // compute reference solution
    printf("Computing result using host CPU...");
    float *reference = (float *)malloc(mem_size_C);
    matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    printf("done.\n");

    // check result (CUBLAS)
    bool resCUBLAS = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);

    if (resCUBLAS != true)
    {
        printDiff(reference, h_C, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
    }

    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");
    //clean up memory
    printf("Clean up Host Memory.....\n");
    free(h_A);
    free(h_B);
    free(h_C);
    printf("done.\n");

    printf("Clean up Deivce Memory.....\n");
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    printf("done.\n");
    cudaDeviceReset();
    printf("Program Completed!!!\n");
    return 0;
}
