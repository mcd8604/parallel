#include <GL/glut.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.h>
//#include <device_types.h>
#include "RayTracer.h"
#include <stdlib.h>
#include <time.h>

// Screen size
#define RES_WIDTH 800.0
#define RES_HEIGHT 600.0

// Host data
unsigned int width;
unsigned int height;
float3 camPos;
float3 camTar;
float3 camUp;
float fovy;
float near;
float far;

int maxTriangles;
Triangle *triangles;
int maxSpheres;
Sphere *spheres;
int maxLights;
Light *lights;
float4 backgroundColor;
float4 ambientLight;

// Device data
__constant__ unsigned int d_width;
__constant__ unsigned int d_height;
__constant__ float3x4 d_invViewMatrix;
__constant__ float4 d_AmbientLight;
__constant__ float4 d_BackgroundColor;
__device__ Light *d_Lights;
__device__ Triangle *d_Triangles;
__device__ Sphere *d_Spheres;

// Kernel functions
__global__ void trace(float4 **d_PixelData, size_t pitch);

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
	
	maxLights = 2;
	lights = (Light *)malloc(maxLights * sizeof(Light));
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

	maxTriangles = 2;
	triangles = (Triangle *)malloc(maxTriangles * sizeof(Triangle));
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
	
	maxSpheres = 2;
	spheres = (Sphere *)malloc(maxSpheres * sizeof(Sphere));
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

    cudaMemcpy(&d_invViewMatrix, invViewMatrix, 12 * sizeof(float), cudaMemcpyHostToDevice);
}

// Copy scene data to device
void SetSceneData()
{
    cudaMemcpy(&d_width, &width, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_height, &height, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_AmbientLight, &ambientLight, sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_BackgroundColor, &backgroundColor, sizeof(float4), cudaMemcpyHostToDevice);

	size_t sizeLights = maxLights * sizeof(Light);
	cudaMalloc((void **)&d_Lights, sizeLights);
	cudaMemcpy(&d_Lights, lights, sizeLights, cudaMemcpyHostToDevice);
	
	size_t sizeTriangles = maxTriangles * sizeof(Triangle);
	cudaMalloc((void **)&d_Triangles, sizeTriangles);
	cudaMemcpy(&d_Triangles, triangles, sizeTriangles, cudaMemcpyHostToDevice);
	
	size_t sizeSpheres = maxSpheres * sizeof(Sphere);
	cudaMalloc((void **)&d_Spheres, sizeSpheres);
	cudaMemcpy(&d_Spheres, spheres, sizeSpheres, cudaMemcpyHostToDevice);
}

void FreeSceneData()
{
	cudaFree(d_Lights);
	cudaFree(d_Triangles);
	cudaFree(d_Spheres);
}

// Draws the graphics
void Draw() {
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	float4 **d_PixelData;
	size_t pitch;		
	cudaMallocPitch((void **)&d_PixelData, &pitch, RES_WIDTH * sizeof(float), RES_HEIGHT);
	dim3 gridSize(64, 64);
	dim3 blockSize(width / 64, height / 64);
	trace<<<gridSize, blockSize>>>(d_PixelData, pitch);

	//float4 *pixelData;
	//cudaMemcpy(&pixelData, &d_PixelData, pitch, cudaMemcpyDeviceToHost);
	//glDrawPixels(width, height, GL_RGB, GL_FLOAT, pixelData);

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
	SetSceneData();
	UpdateViewMatrix();
		
	glutMainLoop();

	FreeSceneData();

	return 0;
}


// kernel functions


__device__
float4 Illuminate(Ray ray, int depth) {
	float x, y, z, w;
	x = 0.5;
	y = 0;
	z = 0;
	w = 1.0;
	return make_float4(x, y, z, w);
}

__global__
void trace(float4 **d_PixelData, size_t pitch) {
	int x, y;
	x = 0;
	y = 0;
	Ray ray;
	d_PixelData[x][y] = Illuminate(ray, 1);
}
/*
    float3 intersectPoint;
    RTObject *rt = getClosestIntersection(ray, &intersectPoint);

    if (rt)
    {
        float3 intersectNormal = rt->GetIntersectNormal(intersectPoint);

        //float3 viewVector = (ray.Position - intersectPoint).Normalize();
        float3 viewVector = -ray.Direction;
        float4 totalLight = rt->calculateAmbient(ambientLight, intersectPoint);
        totalLight = totalLight + spawnShadowRay(intersectPoint, rt, intersectNormal, viewVector, depth);

        if (depth < recursionDepth)
        {
            float3 incidentVector = (intersectPoint - ray.Position).Normalize();

            // Material is reflective
            Material *m = rt->GetMaterial();
            if (m->kR > 0)
            {
                float3 dir = incidentVector.Reflect(intersectNormal);
                Ray reflectionRay;
                reflectionRay.Position = intersectPoint;
                reflectionRay.Direction = dir;
                totalLight = totalLight + (Illuminate(reflectionRay, depth + 1) * m->kR);
            }

            // Material is transparent
            if (m->kT > 0)
            {
                totalLight = totalLight + spawnTransmissionRay(depth, intersectPoint, rt, intersectNormal, incidentVector);
            }
        }

        return totalLight;
    }
    else
    {
        return backgroundColor;
    }
}

/// <summary>
/// Spawns a recursive, transmitted (refracted) ray.
/// </summary>
/// <param name="depth">Current recursion depth</param>
/// <param name="intersectPoint">Origin of the ray</param>
/// <param name="intersectedObject">World object that was intersected</param>
/// <param name="intersectNormal">Normal of the world object at the intersection point</param>
/// <param name="totalLight">Total light to contribute to.</param>
/// <param name="incidentVector">Ray direction incident to intersection.</param>
__device__
float4 spawnTransmissionRay(int depth, float3 intersectPoint, RTObject *intersectedObject, float3 intersectNormal, float3 incidentVector)
{
	float n;

	// Parity check
	Material *m = intersectedObject->GetMaterial();
	if (depth % 2 == 0)
	{
		// assuming outside to inside
		n = m->n;
	}
	else
	{
		// assuming inside to outside
		n = 1 / m->n;
		intersectNormal = intersectNormal * -1;
	}

	double dot = incidentVector.Dot(intersectNormal);
	double discriminant = 1 + ((n * n) * ((dot * dot) - 1));

	if (discriminant < 0)
	{
		// simulate total internal reflection
		float3 dir = incidentVector.Reflect(intersectNormal);
		Ray reflectionRay;
		reflectionRay.Position = intersectPoint;
		reflectionRay.Direction = dir;
		return Illuminate(reflectionRay, depth + 1) * m->n;
	}
	else
	{
		float3 dir = incidentVector * n + (intersectNormal * (n * dot - sqrt(discriminant)));
		Ray transRay;
		transRay.Position = intersectPoint;
		transRay.Direction = dir;
		return Illuminate(transRay, depth + 1) * intersectedObject->GetMaterial()->kT;
	}
}

/// <summary>
/// Spawns a shadow ray.
/// </summary>
/// <param name="intersectPoint">Origin of the ray</param>
/// <param name="intersectedObject">World object that was intersected</param>
/// <param name="intersectNormal">Normal of the world object at the intersection point</param>
/// <param name="viewVector">Camera view vector.</param>
/// <param name="depth">current recursion depth.</param>
/// <returns></returns>
__device__
float4 spawnShadowRay(float3 intersectPoint, RTObject *intersectedObject, float3 intersectNormal, float3 viewVector, int depth)
{
	float4 diffuseTotal;
	float4 specularTotal;

	unsigned int i;
	for(i = 0; i < lights.size(); ++i)
	{
		Light light = lights[i];

		// Spawn a shadow ray from the intersection point to the light source
		float3 lightVector = (light.Position - intersectPoint).Normalize();

		// but only if the intersection is facing the light source
		float facing = intersectNormal.Dot(lightVector);
		if (facing < 0)
		{
			Ray shadowRay;
			shadowRay.Position = intersectPoint;
			shadowRay.Direction = lightVector;

			// Check if the shadow ray reaches the light before hitting any other object
			float dist = intersectPoint.Distance(light.Position);
			bool shadowed = false;

			float4 shadowLight;

			unsigned int k;
			for(k = 0; k < worldObjects.size(); ++k)
			{
				RTObject *rt = worldObjects[k];
				if (*rt != intersectedObject)
				{
					float curDist = rt->Intersects(shadowRay);
					if (curDist > 0 && curDist < dist)
					{
						dist = curDist;
						shadowed = true;

#if !TRANSMIT_SHADOW
						break;
#else
						Material *m = rt->GetMaterial();
						if (m->kT > 0)
						{
							float3 incidentVector = (intersectPoint - shadowRay.Position).Normalize();
							float3 shadowIntersect = shadowRay.Position + (shadowRay.Direction * curDist);
							float3 shadowNormal = rt->GetIntersectNormal(shadowIntersect);

							shadowLight = (shadowLight + spawnTransmissionRay(depth, shadowIntersect, rt, shadowNormal, incidentVector)) * m->kT;
						}
						else
						{
							shadowLight.x = 0;
							shadowLight.y = 0;
							shadowLight.z = 0;
							shadowLight.w = 0;
							break;
						}
#endif
					}
				}
			}

			if (shadowed)
			{
				diffuseTotal = diffuseTotal + intersectedObject->calculateDiffuse(intersectPoint, intersectNormal, light, lightVector) * shadowLight;
				specularTotal = specularTotal + intersectedObject->calculateSpecular(intersectPoint, intersectNormal, light, lightVector, viewVector) * shadowLight;
			}
			else
			{
				diffuseTotal = diffuseTotal + intersectedObject->calculateDiffuse(intersectPoint, intersectNormal, light, lightVector);
				specularTotal = specularTotal + intersectedObject->calculateSpecular(intersectPoint, intersectNormal, light, lightVector, viewVector);
			}

		}
	}

	Material *m = intersectedObject->GetMaterial();
	return diffuseTotal * m->diffuseStrength + specularTotal * m->specularStrength;
}

/// <summary>
/// Finds the closest intersected RTObjectand sets the intersectPoint float3.
/// </summary>
/// <param name="ray">The ray to test RTObjectintersections.</param>
/// <param name="intersectPoint">The float3 to hold the intersection data.</param>
/// <returns>The closest intersected RTObject, or null if no RTObject is intersected.</returns>
__device__
RTObject *getClosestIntersection(Ray ray, float3 *intersectPoint)
{
	float minDist = FLT_MAX;
	float curDist;
	RTObject *intersected = NULL;

	unsigned int i;
	for(i = 0; i < worldObjects.size(); ++i)
	{
		RTObject *rt = worldObjects[i];
		curDist = rt->Intersects(ray);
		if (curDist > 0 && curDist < minDist)
		{
			minDist = curDist;
			intersected = rt;
		}
	}

	if(intersected)
		*intersectPoint = ray.Position + ray.Direction * minDist;

	return intersected;
}
*/

