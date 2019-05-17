
#include <stdio.h>
#include <pthread.h>

#include "CycleTimer.h"

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
    
} WorkerArgs;

//int roundNumber = 0; // this parameter is to count the number of times the thread process image.
// pthread_mutex_t lock;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
    // workerThreadStart --
    //
    // Thread entrypoint.
void *workerThreadStart(void *threadArgs)
{

    WorkerArgs *args = static_cast<WorkerArgs *>(threadArgs);

    // TODO: Implement worker thread here.

 
    int numRows = 1;
    
     for (int i = 0; i < (args->height / args->numThreads); i++)
    {  
        int startRow = args->threadId + i*args->numThreads;
        mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                         args->width, args->height, startRow, numRows, args->maxIterations,
                         args->output);
    }
    

    printf("Hello world from thread %d\n", args->threadId);
 

    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    printf("Numthread:%d\n", numThreads);

    for (int i = 0; i < numThreads; i++)
    {
        // TODO: Set thread arguments here.
        args[i].numThreads = numThreads;
        args[i].threadId = i;
        // printf("ThreadId:%d\n", args->threadId);
        args[i].height = height;
        // printf("height: %d\n", args->height);
        args[i].width = width;
        // printf("width: %d\n", args->width);
        args[i].maxIterations = maxIterations;
        args[i].x0 = x0;
        // printf("xo: %f\n", args->x0);
        args[i].x1 = x1;
        // printf("x1: %f\n", args->x1);
        args[i].y0 = y0;
        // printf("y0: %f\n", args->y0);
        args[i].y1 = y1;
        // printf("y1: %f\n", args->y1);
        args[i].output = output;
    }

    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.
    for (int i = 1; i < numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);


    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i = 1; i < numThreads; i++)
        pthread_join(workers[i], NULL);

}
