/******************************************************************************************
*
*	Filename:	summa.c
*	Purpose:	A paritally implemented program for MSCS6060 HW. Students will complete 
*			the program by adding SUMMA implementation for matrix multiplication C = A * B.  
*	Assumptions:    A, B, and C are square matrices n by n; 
*			the total number of processors (np) is a square number (q^2).
*	To compile, use 
*	    mpicc -o summa summa.c
*       To run, use
*	    mpiexec -n $(NPROCS) ./summa
*********************************************************************************************/

#include <stdio.h>
#include <time.h>	
#include <stdlib.h>	
#include <math.h>	
#include "mpi.h"
#include <string.h>

#define min(a, b) ((a < b) ? a : b)
//#define SZ 4000		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.



/**
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {
	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));
        array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
        for (i=1; i<n_rows; i++){
                array[i] = array[0] + i * n_cols;
        }
        return array;
}

/**
*	Initialize arrays A and B with random numbers, and array C with zeros. 
*	Each array is setup as a square block of blck_sz.
*	ii,jj is the index of each block
*	And i,j is the index of the whole matrix
**/

// void initialize_for_test(double **lA, double **lB, double **lC, int blck_sz, int coordinates[2]){
// 	int i, j, ii, jj;

// 	for (ii=0; ii<blck_sz; ii++){
// 		for (jj=0; jj<blck_sz; jj++){		
// 			i = ii + blck_sz * coordinates[0];
// 			j = jj + blck_sz * coordinates[1];
			
// 			if (i == j) {
// 				lA[ii][jj] = 1.0;
// 				lB[ii][jj] = 1.0;
// 			}else if( (j-1) == i ){
// 				lB[ii][jj] = 1.0;
// 				lA[ii][jj] = 0.0;
// 			}else if( (i-1)== j ){
// 				lA[ii][jj] = 1.0;
// 				lB[ii][jj] = 0.0;
// 			}else{
// 				lA[ii][jj] =0.0;
// 				lB[ii][jj] =0.0;
// 			}
			
// 			lC[ii][jj] = 0.0;
// 		}
// 	}

// }

void initialize(double **lA, double **lB, double **lC, int blck_sz){
//	Set random values...technically it is already random and this is redundant
	int i,j;
	for (i=0; i<blck_sz; i++){
		for (j=0; j<blck_sz; j++){
			lA[i][j] = (double)rand() / (double)RAND_MAX;
			lB[i][j] = (double)rand() / (double)RAND_MAX;
			lC[i][j] = 0.0;
		}
	}
}

//Basic SUMMA Algorithm
void matmulAdd(double **my_C, double **my_A, double **my_B, int block_sz){

	int i, j, k;
	
	for(k = 0; k < block_sz; k++)
	{	
		for(i = 0; i < block_sz; i++)
		{
			for( j = 0; j < block_sz; j++)
			{
				my_C[i][j] += my_A[i][k] * my_B[k][j];
			}
		}
	}
}

/**
*	Perform the SUMMA matrix multiplication. 
*       Follow the pseudo code in lecture slides.
*/
void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
						double **my_B, double **my_C, int coordinates[2], MPI_Comm grid_comm){

	//Add your implementation of SUMMA algorithm
	// int wraparound[2];
	//int coordinates[2];
	// int dimsizes[2];
	int free_coords[2];
	// int reorder = 1;
	int k;

	// dimsizes[0] = dimsizes[1] = proc_grid_sz;
	// wraparound[0] = wraparound[1] = 1;

	double **buff_A;
	buff_A =  alloc_2d_double(block_sz, block_sz);
	double **buff_B; 
	buff_B=  alloc_2d_double(block_sz, block_sz);

	MPI_Comm row_comm;
	MPI_Comm col_comm;
	//MPI_Comm grid_comm;

	// MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
	// MPI_Comm_rank(grid_comm, &my_rank);
	// MPI_Cart_coords(grid_comm, my_rank, 2, coordinates);

	free_coords[0] = 0;
	free_coords[1] = 1;
	MPI_Cart_sub(grid_comm,free_coords, &row_comm);

	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);
	//printf("MPI_Cart_sub Created....\n");
	for(k = 0; k < proc_grid_sz; k++)
	{	
		if (coordinates[1] == k) {

			memcpy(buff_A[0], my_A[0], block_sz*block_sz*sizeof(double));
			//printf("Buff_A copy to My_A completed!....\n");

		}
		MPI_Bcast(*buff_A, block_sz*block_sz, MPI_DOUBLE, k, row_comm);
		//printf("Row_comm Bcast Completed....\n");
		
		if (coordinates[0] == k) {
			
			memcpy(buff_B[0], my_B[0], block_sz*block_sz*sizeof(double));
			//printf("Buff_B copy to My_B completed!....\n");

		}
		MPI_Bcast(*buff_B, block_sz*block_sz, MPI_DOUBLE, k, col_comm);
		//printf("Col_comm Bcast Completed....\n");
		
		if (coordinates[0] == k && coordinates[1] == k) {
			
			matmulAdd(my_C, my_A, my_B, block_sz);

		}else if(coordinates[0] == k){

			matmulAdd(my_C, buff_A, my_B, block_sz);

		}else if(coordinates[1] == k){

			matmulAdd(my_C, my_A, buff_B, block_sz);
			
		}else{
			matmulAdd(my_C, buff_A, buff_B, block_sz);
		}
		
		
	}
	
}




int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides

	srand(time(NULL));							// Seed random numbers

	int SZ = 0;
	
	if (argc == 2) {
		SZ = atoi(argv[1]);
	}else{
		SZ = 4000;
	}
	
	int wraparound[2];
	int coordinates[2];
	int dimsizes[2];
	int reorder = 1;




/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/

	
	MPI_Init (&argc, &argv);
	MPI_Comm grid_comm;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);


/* assign values to 1) proc_grid_sz and 2) block_sz*/
	
	proc_grid_sz = (int)sqrt((double)num_proc);
	block_sz = SZ/proc_grid_sz;
	dimsizes[0] = dimsizes[1] = proc_grid_sz;
	wraparound[0] = wraparound[1] = 1;

	
	
	int my_rank;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Cart_coords(grid_comm, my_rank, 2, coordinates);
	if (SZ % proc_grid_sz != 0){
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process

	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);

	//Call this function to create Bi/tridigonal for testing
	//initialize_for_test(A, B, C, block_sz, coordinates);

	initialize(A, B, C, block_sz);
	//printf("Initialize Finished....\n");

	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();


	// Use SUMMA algorithm to calculate product C
	//printf("Summa Begin....\n");	
	matmul(rank, proc_grid_sz, block_sz, A, B, C, coordinates, grid_comm);
	//printf("Summa Finished....\n");	

	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();


	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time; 
	//printf("Computation Time: %f ms. \n", total_time);

//----------------------------------------------------------------------------------
	// Insert statements for testing
	//printf("Matrix Comparation Testing Begin....\n");
	// int ii,jj,i,j;
	
	// for(ii = 0; ii < block_sz; ii++)
	// {
		
	// 	for(jj = 0; jj < block_sz; jj++)
	// 	{
	// 		i = ii + block_sz * coordinates[0];
	// 		j = jj + block_sz * coordinates[1];

			
	// 		if (i == 0 && j==0 ) {
				
	// 			if (C[ii][jj]!=1) {
	// 				printf("C[%d][%d] is incorrect\n", ii,jj);
					
	// 			}
				
	// 		}else if(i == j){
	// 			if (C[ii][jj]!=2) {
	// 				printf("C[%d][%d] is incorrect\n", ii,jj);
					
	// 			}
	// 		}else if( (i-1) == j){
	// 			if (C[ii][jj]!=1) {
	// 				printf("C[%d][%d] is incorrect\n", ii,jj);
					
	// 			}
	// 		}else if(i == (j-1) ){
	// 			if (C[ii][jj]!=1) {
	// 				printf("C[%d][%d] is incorrect\n", ii,jj);
					
	// 			}
	// 		}
			
	// 	}
		
	// }
	// printf("Matrix Comparation Testing Finished....\n");
//----------------------------------------------------------------------------------
	if (rank == 0){
		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength:%d,numMPICopies:%d,walltime:%lf ms\n",
			SZ, num_proc, total_time);
	}

	// Destroy MPI processes

	MPI_Finalize();
	//printf("Program Compeleted....\n");
	return 0;
}
