#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <vector>

#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR
#include <mpi.h>

#define MY_RAND_MAX 9

using namespace std;

struct matrix2D
{
	int w;
	int h;
	vector<vector<int> > data;
};

struct matrix2D *generateMatrix2D(int w, int h)
{
	struct matrix2D *m;
	m->w = w;
	m->h = h;
	
	int data[w][h];
	
    // generate random data
    int x, y;
    srand(time(NULL));
    for(y = 0; y < m->h; y++)
    {
    	vector<int> row;
    	for(x = 0; x < m->w; x++)
    		row.push_back(rand() % MY_RAND_MAX);
    	m->data.push_back(row);
    }
    
	return m;
}

int matrixMultiply(struct matrix2D *m1, struct matrix2D *m2)
{
	if(m1->w != m2->h)
		return 0;
	
	vector<vector<int> > m3;
	int x, y;
	for(y = 0; y < m1->h; y++)
	{	
		vector<int> row;
		for(x = 0; x < m2->w; x++)
		{	
			int i;
			int dot = 0;
			for(i = 0; i < m1->w; i++)
				dot += m1->data[i][y] * m2->data[x][i];
			row.push_back(dot);
		}
		m3.push_back(row);
	}
	m1->w = m2->w;
	m1->data = m3;
	return 1;
}

void printMatrix(struct matrix2D *m)
{
	int y;
	for(y = 0; y < m->h; y++)
	{
		int x;
		for(x = 0; x < m->w; x++)
			cout << m->data[x][y] << " ";
		cout << "\n";
	}
	cout << "\n";
}

void testSerial(struct matrix2D *m1, struct matrix2D *m2)
{    
    // print initial data
    printMatrix(m1);
    printMatrix(m2);
    // run the test
    if(matrixMultiply(m1, m2))
        printMatrix(m1);
    else
    	cout << "Invalid operands\n";
}

void testParallel()
{
	int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
}

int main(int argc, char **argv)
{	
    struct matrix2D *m1 = generateMatrix2D(3, 5);
    struct matrix2D *m2 = generateMatrix2D(6, 3);
    
    /* Start up MPI */
    MPI_Init(&argc, &argv);

	//testSerial();
    testParallel();

    delete m1;
    delete m2;
    
	/* Shut down MPI */
    MPI_Finalize();    

	return 0;
}
