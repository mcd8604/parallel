#include "RayTracer.h"
#include <iostream>
#include <GL/glut.h>
#include <pthread.h>

// Screen size
#define RES_WIDTH 800.0
#define RES_HEIGHT 600.0

using namespace std;
using namespace RayTracer;

float *pixelData;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Draws the graphics
void Draw() {
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	pthread_mutex_lock(&mutex);
	glDrawPixels(RES_WIDTH, RES_HEIGHT, GL_RGB, GL_FLOAT, pixelData);
	//GLenum err = glGetError();
	//if(err != 0) cout << err << "\n";
	pthread_mutex_unlock(&mutex);
	glutSwapBuffers();
}

void GetSceneData(Scene *s)
{
	// TODO: read initialization data from file, data source, or user input

	s->SetViewProjection(Vector3(3, 4, 15), Vector3(3, 0, 70), Vector3(0, 1, 0),
			45.0, RES_WIDTH, RES_HEIGHT, 0.1, 100);
	s->SetRecursionDepth(5);
	s->SetBackground(Vector4(.5, .7, .9, 1));
	s->SetAmbient(Vector4(.6, .6, .6, 1));
	s->AddLight(Vector3(5, 8, 15), Vector4(1, 1, 1, 1));
	s->AddLight(Vector3(-5, 8, 15), Vector4(1, 1, 1, 1));

	RTTriangle *floor1 = new RTTriangle(Vector3(8, 0, 16), Vector3(-8, 0, -16), Vector3(8, 0, -16), Vector3(0, 1, 0));
	RTTriangle *floor2 = new RTTriangle(Vector3(8, 0, 16), Vector3(-8, 0, -16), Vector3(-8, 0, 16), Vector3(0, 1, 0));
    Material *floorMat = new Material();
    floorMat->ambientStrength = 1;
    floorMat->diffuseStrength = 1;
    floorMat->setAmbientColor(Vector4(0.2, 1, 0.2, 1));
    floorMat->setDiffuseColor(Vector4(0.2, 1, 0.2, 1));
    floorMat->setSpecularColor(Vector4(0.2, 1, 0.2, 1));
    floor1->SetMaterial(floorMat);
    floor2->SetMaterial(floorMat);
    //floor.MaxU = 10;
    //floor.MaxV = 15;
    s->AddObject(floor1);
    s->AddObject(floor2);

    RTObject *sphere1 = new RTSphere(Vector3(3, 4, 11), 1);
    Material *glass = new Material();
    glass->ambientStrength = 0.075;
    glass->diffuseStrength = 0.075;
    glass->specularStrength = 0.2;
    glass->exponent = 20;
    glass->setAmbientColor(Vector4(1, 1, 1, 1));
    glass->setDiffuseColor(Vector4(1, 1, 1, 1));
    glass->setSpecularColor(Vector4(1, 1, 1, 1));
    glass->kR = .01;
    glass->kT = .99;
    glass->n = .99;
    sphere1->SetMaterial(glass);
    s->AddObject(sphere1);

    RTSphere *sphere2 = new RTSphere(Vector3(1.5, 3, 9), 1);
    Material *mirror = new Material();
    mirror->ambientStrength = 0.15;
    mirror->diffuseStrength = 0.25;
    mirror->specularStrength = 1;
    mirror->exponent = 20;
    mirror->setAmbientColor(Vector4(.7, .7, .7, .7));
    mirror->setDiffuseColor(Vector4(.7, .7, .7, .7));
    mirror->setSpecularColor(Vector4(1, 1, 1, 1));
    mirror->kR = .75;
    sphere2->SetMaterial(mirror);
    s->AddObject(sphere2);
}

void *trace(void *threadID) {
	Scene *s = new Scene();
	GetSceneData(s);
	Vector4 *vectorData = new Vector4[(int)RES_WIDTH * (int)RES_HEIGHT];

	while(true) {
		// perform the ray tracing
		s->trace(vectorData);
		pthread_mutex_lock(&mutex);
		int i = 0, p = 0;
		for(;i < RES_WIDTH * RES_HEIGHT; ++i)
		{
			Vector4 v = vectorData[i];
			pixelData[p] = (float)v.x;
			pixelData[p + 1] = (float)v.y;
			pixelData[p + 2] = (float)v.z;
			//pixelData[p + 3] = (float)v.w;
			p+=3;
		}
		pthread_mutex_unlock(&mutex);
		glutPostRedisplay();
		//cout << "Update\n";
	}

	delete s;
	delete vectorData;

	pthread_exit(NULL);

	return 0;
}

void idle() {
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(RES_WIDTH,RES_HEIGHT);
	glutCreateWindow("Ray Tracer Test GL");

	glutDisplayFunc(Draw);
	glutIdleFunc(idle);
	//glutReshapeFunc(reshape);
	//glutKeyboardFunc(keyboard);

	pixelData = new float[(int)RES_WIDTH * (int)RES_HEIGHT * 3];
	pthread_mutex_init(&mutex, NULL);

	// run the raytracer on a seperate thread
	pthread_t rtThread;
	pthread_create(&rtThread, NULL, trace, NULL);

	glutMainLoop();

	delete pixelData;

	return 0;
}
