#include <GL/glut.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.h>
//#include <device_types.h>
#include "RayTracer.h"
#include <stdlib.h>
#include <time.h>
#include <float.h>

// Screen size
#define RES_WIDTH 800.0
#define RES_HEIGHT 600.0

unsigned int width;
unsigned int height;
float3 camPos;
float3 camTar;
float3 camUp;
float fovy;
float near;
float far;

int numTriangles;
Triangle *triangles;
int numSpheres;
Sphere *spheres;
int numLights;
Light *lights;
float4 backgroundColor;
float4 ambientLight;

float4 **pixelData;

void SetSceneData(float width, float height, float4 backgroundColor, float4 ambientLight,
		unsigned int numLights, Light *lights, unsigned int numTriangles, Triangle *triangles, unsigned int numSpheres, Sphere *spheres);
void FreeSceneData();
void SetViewMatrix(float invViewMatrix[12]);
void GetPixelData(float4 **pixelData);

void GetSceneData()
{
	// TODO: read initialization data from file, data source, or user input

	width = RES_WIDTH;
	height = RES_HEIGHT;

	camPos = make_float3(3, 4, 15);
	camTar = make_float3(3, 0, -70);
	camUp = make_float3(0, 1, 0);
	fovy = 45.0;
	near = 0.1;
	far = 100;
	
	backgroundColor = make_float4(.5, .7, .9, 1);
	ambientLight = make_float4(.6, .6, .6, 1);
	
	numLights = 2;
	lights = (Light *)malloc(numLights * sizeof(Light));
	lights[0].Position = make_float3(5, 8, 15);
	lights[0].Color = make_float4(1, 1, 1, 1);
	lights[1].Position = make_float3(-5, 8, 15);
	lights[1].Color = make_float4(1, 1, 1, 1);

	Triangle floor1;
	floor1.v1 = make_float3(8, 0, 16);
	floor1.v2 = make_float3(-8, 0, -16);
	floor1.v3 = make_float3(8, 0, -16); 
	floor1.n = make_float3(0, 1, 0);
	
	Triangle floor2;
	floor2.v1 = make_float3(8, 0, 16);
	floor2.v2 = make_float3(-8, 0, -16);
	floor2.v3 = make_float3(-8, 0, 16); 
	floor2.n = make_float3(0, 1, 0);

	Material floorM;
	floorM.ambientStrength = 1;
	floorM.diffuseStrength = 1;
    floorM.ambientColor = make_float4(0.2, 1, 0.2, 1);
    floorM.diffuseColor = make_float4(0.2, 1, 0.2, 1);
    floorM.specularColor = make_float4(0.2, 1, 0.2, 1);
	floor1.m = floorM;
	floor2.m = floorM;

	numTriangles = 2;
	triangles = (Triangle *)malloc(numTriangles * sizeof(Triangle));
	triangles[0] = floor1;
	triangles[1] = floor2;

    Sphere sphere1;
	sphere1.p = make_float3(3, 4, 11);
	sphere1.r = 1;
	// glass material
    Material glass;
    glass.ambientStrength = 0.075;
    glass.diffuseStrength = 0.075;
    glass.specularStrength = 0.2;
    glass.exponent = 20;
    glass.ambientColor = make_float4(1, 1, 1, 1);
    glass.diffuseColor = make_float4(1, 1, 1, 1);
    glass.specularColor = make_float4(1, 1, 1, 1);
    glass.kR = .01;
    glass.kT = .99;
    glass.n = .99;
    sphere1.m = glass;

    Sphere sphere2;
	sphere2.p = make_float3(1.5, 3, 9);
	sphere2.r = 1;
	// mirror material
    Material mirror;
    mirror.ambientStrength = 0.15;
    mirror.diffuseStrength = 0.25;
    mirror.specularStrength = 1;
    mirror.exponent = 20;
    mirror.ambientColor = make_float4(.7, .7, .7, .7);
    mirror.diffuseColor = make_float4(.7, .7, .7, .7);
    mirror.specularColor = make_float4(1, 1, 1, 1);
    mirror.kR = .75;
    sphere2.m = mirror;
	
	numSpheres = 2;
	spheres = (Sphere *)malloc(numSpheres * sizeof(Sphere));
	spheres[0] = sphere1;
	spheres[1] = sphere2;
}

// Creates an inverse view matrix from the camera variables then
// copies it to device memory
// This matrix is used to create world-space rays from screen-space pixels
void UpdateViewMatrix()
{
    // use OpenGL to build view matrix
	// NOTE: modified code from NVIDIA CUDA SDK sample code, volumeRender.cpp
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
        glLoadIdentity();
		gluLookAt(camPos.x, camPos.y, camPos.z, 
			camTar.x, camTar.y, camTar.z, 
			camUp.x, camUp.y, camUp.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();
		
	float invViewMatrix[12];
    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

	SetViewMatrix(invViewMatrix);
}

// Draws the graphics
void Draw() {
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	GetPixelData(pixelData);

	glutSwapBuffers();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(RES_WIDTH, RES_HEIGHT);
	glutCreateWindow("CUDA Ray Tracer");

	glutDisplayFunc(Draw);
	
	GetSceneData();
	SetSceneData(width, height, ambientLight, backgroundColor,
			numLights, lights, numTriangles, triangles, numSpheres, spheres);
	UpdateViewMatrix();

	pixelData = (float4 **)malloc(sizeof(float4) * width * height);

	glutMainLoop();

	FreeSceneData();

	return 0;
}
