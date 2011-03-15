#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

#define ROOT_RANK 0
#define BASE 2
#define DEFAULT_N 64
#define MAX 9
#define TAG_ALLOC 0
#define TAG_DATA 1

int my_rank, p;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Status status;
	
	// For sake of simplicity
	if(!(p == 2 || p == 4 || p == 8))
	{
		if(my_rank == ROOT_RANK)
			printf("Number of nodes must be 2, 4, or 8. Given: %i\n", p);
		return 1;
	}

	int n, procN;
	int *data;

	if(my_rank == ROOT_RANK)
	{
		n = 0;
		if(argc == 1)
			n = DEFAULT_N;
		else
			n = atoi(argv[1]);
		
		// Determine array length

		int paddedN = n;
		procN = n / p;
		if(n % p > 0)
		{
			paddedN = ((n / p) + 1) * p;
			procN = paddedN / p;
		}

		data = malloc(sizeof(int) * paddedN);

		// Generate random data
		
		srand(time(NULL));
		int i;
		for(i = 0; i < n; ++i)
			data[i] = rand() % MAX;
		//for(i = n; i < paddedN; ++i)
		//	data[i] = 0;

		// Broadcast data

		for(i = 0; i < p; ++p)
		{
			if(i != ROOT_RANK)
			{
				MPI_Send(&paddedN, 1, MPI_INT, i, TAG_ALLOC, MPI_COMM_WORLD);
				MPI_Send(&data, paddedN, MPI_INT, i, TAG_DATA, MPI_COMM_WORLD);
			}
		}
	} else {
		// Receive data
		
		MPI_Recv(&n, 1, MPI_INT, ROOT_RANK, TAG_ALLOC, MPI_COMM_WORLD, &status);

		printf("NODE %i: Recv: n = %i\n", my_rank, n);
		
		data = malloc(sizeof(int) * n);

		procN = n / p;

		MPI_Recv(&data, n, MPI_INT, ROOT_RANK, TAG_DATA, MPI_COMM_WORLD, &status);
	}

	// Calculate sum

	int i;
	int sum = 0;
	for(i = procN * my_rank; i < procN * (my_rank + 1); ++i)
		sum += data[i];

	printf("NODE %i: sum = %i\n", my_rank, sum);

	// Return sum to parent node, converging at node 0
	
	int count = 0;
	int iDiff = 1;
	while(my_rank % (iDiff * BASE) == 0 && iDiff < p)
	{
		int nextSum;
		// NOTE must add loop for BASE > 2
		MPI_Recv(&nextSum, 1, MPI_INT, my_rank + iDiff, TAG_DATA, MPI_COMM_WORLD, &status);
		sum += nextSum;
		iDiff *= BASE; 
	}
	
	if(my_rank != 0)
		MPI_Send(&sum, 1, MPI_INT, my_rank - iDiff, TAG_DATA, MPI_COMM_WORLD);

	printf("NODE %i: Local Sum = %i\n", my_rank, sum);

	free(data);

	MPI_Finalize();    

	return 0;
}
