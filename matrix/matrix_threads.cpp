#include <stdlib.h>
#include <iostream>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthreads.h>
#endif

#define MY_RAND_MAX 9

using namespace std;

typedef struct Matrix {
	int w, h;
	float *data;
};

typedef struct ThreadData {
	int i, p;
	Matrix *m1;
	Matrix *m2;
	Matrix *m3;
};

void generateMatrix(Matrix *m, int w, int h) {
	if(m->data)
		free(m->data);
	m->w = w;
	m->h = h;
	m->data = (float *)malloc(sizeof(float) * w * h);	
    int x, y;
    srand(time(NULL));
    for(y = 0; y < m->h; y++)
    	for(x = 0; x < m->w; x++)
    		m->data[y * w + x] = rand() % MY_RAND_MAX;
}

void printMatrix(Matrix *m)
{
	int x, y;
	for(y = 0; y < m->h; y++)
	{
		for(x = 0; x < m->w; x++)
			cout << m->data[y * m->w + x] << " ";
		cout << "\n";
	}
	cout << "\n";
}

#ifdef _WIN32
DWORD WINAPI matrixMult(LPVOID threadData)
#else
void *matrixMult(void *threadData)
#endif
{
	ThreadData d = *(ThreadData *)threadData;
	Matrix *m1 = d.m1;
	Matrix *m2 = d.m2;
	Matrix *m3 = d.m3;
		
	int x, y, i;
	for(y = d.i; y < m3->h; y+=d.p)
		for(x = 0; x < m3->w; x++)
			for(i = 0; i < m1->w; i++)
				m3->data[y * m3->h + x] += m1->data[m1->h * y + i] * m2->data[x * m2-> w + i];
	return 0;
}

int main(int argc, char **argv)
{	
	Matrix m1;
	generateMatrix(&m1, 3, 5);
	
	Matrix m2;
	generateMatrix(&m2, 6, 3);

	Matrix m3;
	m3.w = m1.h;
	m3.h = m2.w;
	m3.data = (float *)malloc(sizeof(float) * m3.w * m3.h);

	printMatrix(&m1);
	printMatrix(&m2);
	
	// number of threads
	const int p = 8;

#ifdef _WIN32
	HANDLE hThreadArray[p];
#else
	pthread threads[p];
#endif

	int i;
	for(i = 0; i < p; ++i)
	{
		ThreadData threadData;
		threadData.m1 = &m1;
		threadData.m2 = &m2;
		threadData.m3 = &m3;
		threadData.i = i;
		threadData.p = p;
#ifdef _WIN32
        hThreadArray[i] = CreateThread( 
            NULL,                   // default security attributes
            0,                      // use default stack size  
            matrixMult,				// thread function name
            &threadData,		    // argument to thread function 
            0,                      // use default creation flags 
            NULL);					// returns the thread identifier 
#else
		pthread_create(&threads[i], NULL, matrixMult, &threadData);
#endif
	}
	
#ifdef _WIN32
    WaitForMultipleObjects(p, hThreadArray, TRUE, INFINITE);
#else
	for(i = 0; i < p; ++i)
		pthread_join(&threads[i]);
#endif

	printMatrix(&m3);

	free(m1.data);
	free(m2.data);
	free(m3.data);

	return 0;
}
