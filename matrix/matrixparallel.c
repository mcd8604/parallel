#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

size_t SIZE_MatrixElement = sizeof(float);
typedef struct { int w, h; float **data; } matrix;
typedef struct { int w1, h1, w2; } Info_Alloc;

#define COUNT_ALLOC	3
#define TAG_ALLOC 	0
#define TAG_DATA 	1
#define ROOT_RANK 	0
#define MY_RAND_MAX 	9

matrix generateMatrix(int w, int h) {
	matrix m;
	m.w = w;
	m.h = h;
	m.data = malloc(w * h * sizeof(float));
    
	// generate random data
	int x, y;
	srand(time(NULL));
	for(y = 0; y < m.h; y++)
    		for(x = 0; x < m.w; x++)
    			m.data[x][y] = rand() % MY_RAND_MAX;
    
	return m;
}

void printMatrix(matrix m)
{
	int x, y;
	for(y = 0; y < m.h; y++)
	{
		for(x = 0; x < m.w; x++)
			printf("%f ", m.data[x][y]);
		printf("\n");
	}
	printf("\n");
}

int my_rank, p;

int main(int argc, char **argv)
{ 


	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(my_rank == ROOT_RANK)
	{
		matrix m1 = generateMatrix(3, 5);
		matrix m2 = generateMatrix(6, 3);
		matrix m3;
		m3.w = m2.w;
		m3.h = m1.h;
		m3.data = malloc(m3.w * m3.h * SIZE_MatrixElement);

		// 1. Send (Broadcast) allocation info and data
		Info_Alloc a;
		a.w1 = m1.w;
		a.h1 = m1.h;
		a.w2 = m2.w;

		// The number of vectors to send, to each process
		int n = m2.w / p;
		int r = m2.w % p;
		
		// The matrix data to send: matrix 1 followed by
		// relevant vectors from matrix 2
		int COUNT_MatrixData = m1.w * m1.h + (n + 1) * m2.h;
		void *data = malloc(COUNT_MatrixData * SIZE_MatrixElement);
		void *data2 = data + m1.w * m1.h + 1;
		memcpy(data, m1.data, m1.w * m1.h * SIZE_MatrixElement);
		
		int i;
		int nRoot, sRoot;
		for(i = 0; i < p; ++p)
		{

			// Number of vectors to send, including remainder
			int iN = i < r ? n + 1 : n;

			if(i == ROOT_RANK)
			{
				// Store data locally for root, instead of sending via MPI
				nRoot = iN;
				sRoot = n * i + (i == 0 ? 0 : (r < i ? r : i - 1));
			}
			else
			{
				if(iN > 0)
				{
					MPI_Send(&a, COUNT_ALLOC, MPI_INT, i, TAG_ALLOC, MPI_COMM_WORLD);

					// Start index of vectors from matrix 2
					int s = n * i + (i == 0 ? 0 : (r < i ? r : i - 1));
					void *src = m2.data + m2.h * s;
					
					// Copy the relevant vectors
					memcpy(data2, src, iN * m2.h * SIZE_MatrixElement);

					MPI_Send(&data, COUNT_MatrixData, MPI_FLOAT, i, TAG_DATA, MPI_COMM_WORLD);
				}
			}
		}

		free(data);

		// 2. Calculate own products

		int x, y, v;
		int xEnd = sRoot + nRoot;
		for(x = sRoot; x < xEnd; ++x)
			for(x = 0; x < m2.w; ++x)
				for(v = 0; v < m1.w; ++v) 
					m3.data[x][y] += m1.data[v][y] * m2.data[x][v];

		// 3. Receive (Gather) other products
		
		int wait = p - 1;
		int count = m3.w * (n + 1);
		float *buffer = malloc(count * SIZE_MatrixElement);
		MPI_Status status;
		while(wait > 0)
		{
			MPI_Recv(buffer, count, MPI_FLOAT, MPI_ANY_SOURCE, TAG_DATA, MPI_COMM_WORLD, &status); 

			int i = status.MPI_SOURCE;
			int iN = i < r ? n + 1 : n;
			int s = n * i + (i == 0 ? 0 : (r < i ? r : i - 1));
			int x, y;
			
			// Update product matrix
			for(x = 0; x < iN; ++x)
				for(y = 0; y < m3.h; ++y)
					m3.data[s + x][y] = buffer[x * m3.h + y];
			--wait;
		}

		// 4. Print result
		printMatrix(m3);

		free(m1.data);
		free(m2.data);
		free(m3.data);
	} else {
		// 1. Receive data
		
		Info_Alloc a;
		MPI_Status status;
		MPI_Recv(&a, COUNT_ALLOC, MPI_INT, ROOT_RANK, TAG_ALLOC, MPI_COMM_WORLD, &status);
		
		int n = a.w2 / p;
		int r = a.w2 % p;
		int myN = my_rank < r ? n + 1 : n;

		int count = a.w1 * a.h1 + (n + 1) * a.w1;
		float *buffer = malloc(count * SIZE_MatrixElement);
		MPI_Recv(buffer, count, MPI_FLOAT, ROOT_RANK, TAG_DATA, MPI_COMM_WORLD, &status);

		// 2. Calculate products
		
		count = a.h1 * (n + 1);
		float *vectors = buffer + a.w1 * a.h1;
		float **product = malloc(count * SIZE_MatrixElement);
		int x, y, v;
		for(y = 0; y < myN; ++y)
			for(x = y; x < a.h1; ++x)
				for(v = 0; v < a.w1; ++v)
					product[x][y] += buffer[v * a.h1 + y] * vectors[x * a.w1 + v];

		// 3. Return products
		
		MPI_Send(product, count, MPI_FLOAT, ROOT_RANK, TAG_DATA, MPI_COMM_WORLD);

		free(buffer);
		free(product);
	}

	MPI_Finalize();    

	return 0;
}
