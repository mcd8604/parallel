#include "RayTracer.h"
#include <iostream>
#include <GL/glut.h>
#include <pthread.h>
#include <stdio.h>

// Screen size
#define RES_WIDTH 640.0
#define RES_HEIGHT 480.0

using namespace std;
using namespace RayTracer;

float *pixelData;
//Vector4 *vectorData;
Scene *s;

#define T 1 
pthread_t threads[T];

//pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;
double t = 0;
int count = 0;

// Draws the graphics
void Draw() {
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	//pthread_mutex_lock(&mutex);
	glDrawPixels(RES_WIDTH, RES_HEIGHT, GL_RGBA, GL_FLOAT, pixelData);
	//GLenum err = glGetError();
	//if(err != 0) cout << err << "\n";
	//pthread_mutex_unlock(&mutex);
	glutSwapBuffers();
}

void GetSceneData()
{
	// TODO: read initialization data from file, data source, or user input

	s->SetRecursionDepth(6);
	s->SetBackground(Vector4(.4, .6, .93, 1));
	s->SetAmbient(Vector4(.3, .3, .3, 1));
	s->AddLight(Vector3(5, 8, 15), Vector4(1, 1, 1, 1));
	s->AddLight(Vector3(-5, 8, 15), Vector4(1, 1, 1, 1));
	s->SetViewProjection(Vector3(3, 4, 15), Vector3(3, 0, -70), Vector3(0, 1, 0),
			45.0, RES_WIDTH, RES_HEIGHT, 0.1, 100);

	RTTriangle *floor1 = new RTTriangle(Vector3(8, 0, 16), Vector3(-8, 0, -16), Vector3(8, 0, -16), Vector3(0, 1, 0));
	RTTriangle *floor2 = new RTTriangle(Vector3(8, 0, 16), Vector3(-8, 0, -16), Vector3(-8, 0, 16), Vector3(0, 1, 0));
    Material *floorMat = new Material();
    floorMat->ambientStrength = 0.25;
    floorMat->diffuseStrength = 0.75;
    //floorMat->setAmbientColor(Vector4(0.2, 1, 0.2, 1));
    //floorMat->setDiffuseColor(Vector4(0.2, 1, 0.2, 1));
    //floorMat->setSpecularColor(Vector4(0.2, 1, 0.2, 1));
    floor1->SetMaterial(floorMat, 10, 15);
    floor2->SetMaterial(floorMat, 10, 15);
    //floor.MaxU = 10;
    //floor.MaxV = 15;
    s->AddObject(floor1);
    s->AddObject(floor2);

    RTObject *sphere1 = new RTSphere(Vector3(3, 4, 11), 1);
    Material *glass = new Material();
//    glass->ambientStrength = 0.075;
//    glass->diffuseStrength = 0.075;
//    glass->specularStrength = 0.2;
//    glass->exponent = 20;
//    glass->setAmbientColor(Vector4(1, 1, 1, 1));
//    glass->setDiffuseColor(Vector4(1, 1, 1, 1));
//    glass->setSpecularColor(Vector4(1, 1, 1, 1));
//    glass->kR = .01;
//    glass->kT = .99;
//    glass->n = .99;
	glass->ambientStrength = 0.5;
	glass->diffuseStrength = 0.5;
	glass->specularStrength = 0.5;
	glass->exponent = 20;
	glass->setAmbientColor(Vector4(.7, .7, .7, .7));
	glass->setDiffuseColor(Vector4(1, 1, 1, 1));
	glass->setSpecularColor(Vector4(1, 1, 1, 1));
	glass->kR = .75;

    sphere1->SetMaterial(glass);
    s->AddObject(sphere1);

    RTSphere *sphere2 = new RTSphere(Vector3(1.5, 3, 9), 1);
    Material *mirror = new Material();
    mirror->ambientStrength = 0.5;
    mirror->diffuseStrength = 0.5;
    mirror->specularStrength = 0.5;
    mirror->exponent = 20;
    mirror->setAmbientColor(Vector4(.7, .7, .7, .7));
    mirror->setDiffuseColor(Vector4(1, 1, 1, 1));
    mirror->setSpecularColor(Vector4(1, 1, 1, 1));
    mirror->kR = .75;
    sphere2->SetMaterial(mirror);
    s->AddObject(sphere2);
}

//r - radius
//d - diameter
//sp - spacing
void GetScene2Data(int rows, int columns, float r, float sp) {
	float d = 2 * r;
	s->SetRecursionDepth(3);
	s->SetBackground(Vector4(.4, .6, .93, 1));
	s->SetAmbient(Vector4(.3, .3, .3, 1));

	// v4	v3	|
	//			z
	// v1	v2	+
	// ---x+++
	
	double x = rows * (sp + d);
	double z = columns * (sp + d);
	Vector3 v1, v2, v3, v4;
	v1 = Vector3(-sp, 0, sp);
	v2 = Vector3(x, 0, sp);
	v3 = Vector3(x, 0, -z);
	v4 = Vector3(-sp, 0, -z);
	
	Vector3 h = Vector3(0, z, 0);
	Vector4 l = Vector4(1, 1, 1, 1);
	s->AddLight(v1 - v3 + h, l);
	s->AddLight(v2 + v4 + h, l);
	s->AddLight(v3 + v3 + h, l);
	s->AddLight(v4 - v2 + h, l);
	
	s->SetViewProjection(
			Vector3((x - sp) / 2, r, 0), 
			Vector3(x / 2, 0, -z/2), 
			Vector3(0, 1, 0),
			45.0, RES_WIDTH, RES_HEIGHT, 0.1, 100);

	Vector3 n = Vector3(0, 1, 0);
	RTTriangle *floor1 = new RTTriangle(v1, v2, v3, n);
	RTTriangle *floor2 = new RTTriangle(v1, v3, v4, n);
	
	Material *floorMat = new Material();
    floorMat->ambientStrength = 0.25;
    floorMat->diffuseStrength = 0.5;
    //floorMat->setAmbientColor(Vector4(0.2, 1, 0.2, 1));
    //floorMat->setDiffuseColor(Vector4(0.2, 1, 0.2, 1));
    //floorMat->setSpecularColor(Vector4(0.2, 1, 0.2, 1));
    floor1->SetMaterial(floorMat, rows * 2, columns * 2);
    floor2->SetMaterial(floorMat, rows * 2, columns * 2);
    //floor.MaxU = 10;
    //floor.MaxV = 15;
    s->AddObject(floor1);
    s->AddObject(floor2);

    Material *mirror = new Material();
    mirror->ambientStrength = 0.5;
    mirror->diffuseStrength = 0.5;
    mirror->specularStrength = 0.5;
    mirror->exponent = 20;
    mirror->setAmbientColor(Vector4(.7, .7, .7, .7));
    mirror->setDiffuseColor(Vector4(1, 1, 1, 1));
    mirror->setSpecularColor(Vector4(1, 1, 1, 1));
    mirror->kR = .75;
    
    int xI, zI;
    for(xI = 0; xI < rows; ++xI) {
    	for(zI = 0; zI < columns; ++zI) {
    	    RTObject *sphere = new RTSphere(
    	    		Vector3(
    	    				xI * (d + sp) + r, 
    	    				r, 
    	    				-zI * (d + sp) - r), 
    	    		r);
    	    sphere->SetMaterial(mirror);
    	    s->AddObject(sphere);
    	}
    }
}

void *trace(void *threadID) {
	clock_t start,end;
	int id = (int)threadID;
	cout << "THREADID = " << id << "\n";
	while(true) {
		// perform the ray tracing
		start = clock();
		s->trace(pixelData, id, T);
		end = clock();
		double dif = double(end - start) / CLOCKS_PER_SEC;
		//cout << dif << "\n";
		
		pthread_mutex_lock(&count_mutex);
		
		++count;
		t += dif;
		if(count == T) {
			printf("ID(%i): %.5f\n", id, t);
			count = 0;
			t = 0;
			pthread_cond_signal(&count_threshold_cv);
		} else {
			pthread_cond_wait(&count_threshold_cv, &count_mutex);
		}
		
		pthread_mutex_unlock(&count_mutex);
		
		//pthread_mutex_lock(&mutex);
//		int i;
//		int p = id;
//		for(i = id; i < RES_WIDTH * RES_HEIGHT; i += T)
//		{
//			Vector4 v = vectorData[i];
//			pixelData[p] = (float)v.x;
//			pixelData[p + 1] = (float)v.y;
//			pixelData[p + 2] = (float)v.z;
//			//pixelData[p + 3] = (float)v.w;
//			p += 3 * T;
//		}
		//pthread_mutex_unlock(&mutex);
		//glutPostRedisplay();
		//cout << "Update\n";
	}

	pthread_exit(NULL);

	return 0;
}

void idle() {
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	Vector3 camPos = s->GetCameraPosition();
	int d = s->GetRecursionDepth();
    switch(key) {
        case 27:
            exit(0);
            break;
        case 'w':
        	camPos.z -= 0.1;
            s->SetCameraPosition(camPos);
            break;
        case 's':
        	camPos.z += 0.1;
            s->SetCameraPosition(camPos);
            break;
        case 'a':
        	camPos.x -= 0.1;
            s->SetCameraPosition(camPos);
            break;
        case 'd':
        	camPos.x += 0.1;
            s->SetCameraPosition(camPos);
            break;
        case '[':
        	s->SetRecursionDepth(--d);
            cout << "DEPTH = " << d << "\n";
            break;
        case ']':
        	s->SetRecursionDepth(++d);
            cout << "DEPTH = " << d << "\n";
            break;

        default:
            break;
    }
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(RES_WIDTH,RES_HEIGHT);
	glutCreateWindow("Serial CPU Ray Tracer");
    glViewport(0, 0, RES_WIDTH,RES_HEIGHT);

	glutKeyboardFunc(keyboard);
	glutDisplayFunc(Draw);
	glutIdleFunc(idle);
	//glutReshapeFunc(reshape);
	//glutKeyboardFunc(keyboard);

	s = new Scene();
	GetSceneData();
    //GetScene2Data(4, 20, 1.0, 3);
	pixelData = new float[(int)RES_WIDTH * (int)RES_HEIGHT * 4];
	//vectorData = new Vector4[(int)RES_WIDTH * (int)RES_HEIGHT];
	//pthread_mutex_init(&mutex, NULL);
	pthread_mutex_init(&count_mutex, NULL);
	pthread_cond_init (&count_threshold_cv, NULL);

	// run the raytracer on multiple seperate threads

	int i;
	for(i = 0; i < T; ++i)
		pthread_create(&threads[i], NULL, trace, (void *)i);

	pthread_join(threads[0], NULL);
	//glutMainLoop();

	delete pixelData;
	//delete vectorData;
	
	return 0;
}
