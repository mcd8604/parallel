#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.h>
//#include <device_types.h>
#include "RayTracer.h"
#include <stdlib.h>
#include <time.h>

__constant__ unsigned int d_width;
__constant__ unsigned int d_height;
__constant__ float3x4 d_invViewMatrix;
__constant__ float4 d_AmbientLight;
__constant__ float4 d_BackgroundColor;
__constant__ unsigned int d_numLights;
__constant__ unsigned int d_numTriangles;
__constant__ unsigned int d_numSpheres;
__device__ Light *d_Lights;
__device__ Triangle *d_Triangles;
__device__ Sphere *d_Spheres;

// Kernel functions
__global__ void trace(float4 **d_PixelData, size_t pitch);

// Copies the view matrix to device memory
void SetViewMatrix(float invViewMatrix[12])
{
    cudaMemcpy(&d_invViewMatrix, invViewMatrix, sizeof(float) * 12, cudaMemcpyHostToDevice);
}

// Copy scene data to device
void SetSceneData(float width, float height, float4 ambientLight, float4 backgroundColor,
		unsigned int numLights, Light *lights,
		unsigned int numTriangles, Triangle *triangles,
		unsigned int numSpheres, Sphere *spheres)
{
	cudaMemcpy(&d_width, &width, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_height, &height, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_AmbientLight, &ambientLight, sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_BackgroundColor, &backgroundColor, sizeof(float4), cudaMemcpyHostToDevice);

	size_t sizeLights = numLights * sizeof(Light);
	cudaMalloc((void **)&d_Lights, sizeLights);
	cudaMemcpy(&d_Lights, lights, sizeLights, cudaMemcpyHostToDevice);
	
	cudaMemcpy(&d_numTriangle, &numTriangles, sizeof(unsigned int), cudaMemcpyHostToDevice);
	size_t sizeTriangles = numTriangles * sizeof(Triangle);
	cudaMalloc((void **)&d_Triangles, sizeTriangles);
	cudaMemcpy(&d_Triangles, triangles, sizeTriangles, cudaMemcpyHostToDevice);
	
	cudaMemcpy(&d_numSpheres, &numSpheres, sizeof(unsigned int), cudaMemcpyHostToDevice);
	size_t sizeSpheres = numSpheres * sizeof(Sphere);
	cudaMalloc((void **)&d_Spheres, sizeSpheres);
	cudaMemcpy(&d_Spheres, spheres, sizeSpheres, cudaMemcpyHostToDevice);
}

void FreeSceneData()
{
	cudaFree(d_Lights);
	cudaFree(d_Triangles);
	cudaFree(d_Spheres);
}

void GetPixelData(float4** pixelData) {
	size_t pitch;		
	cudaMallocPitch((void **)&d_PixelData, &pitch, RES_WIDTH * sizeof(float), RES_HEIGHT);
	dim3 gridSize(64, 64);
	dim3 blockSize(width / 64, height / 64);
	trace<<<gridSize, blockSize>>>(d_PixelData, pitch);
}

__global__
void trace(float4 **d_PixelData, size_t pitch) {
	int x, y;
	x = 0;
	y = 0;
	Ray ray;
	d_PixelData[x][y] = Illuminate(ray, 1);
}

__device__
float intersects(Triangle *t, Ray r) {
	//http://www.siggraph.org/education/materials/HyperGraph/raytrace/rapolygon_intersection.htm
	float3 n = t->n;
	float d = Dot(r.Direction, n);
	if(d == 0)
		return -1;
	return Dot((r.Position - v1), n) / d;
}

__device__
float intersects(Sphere *s, Ray ray) {
	// quadratic equation: t = (-b +/= sqrt(b * b - 4 * a * c)) / 2 * a

	// float a = Dot(ray.Direction, ray.Direction);
	// since ray direction is normalized, this will always = (1 += round error)
	// omitting a will save on calculations and reduce error

	float r = s->r;
	float p = s->p;
	float3 diff = ray.Position - p;
	float b = 2.0 * Dot(ray.Direction, diff);
	float c = Dot(diff, diff) - r * r;

	// approximate if below precision quantum
	if (c < .001)//std::numeric_limits<float>::epsilon())
        return 0;

	float d = b * b - 4.0 * c;
	// unreal, no root
	if(d < 0)
		return -1;

	float e = sqrt(d);

	// first root
	float t1 = (-b - e) / 2.0;
	if(t1 >= 0)
		return t1;

	// second root
	float t2 = (-b + e) / 2.0;
	if(t2 >= 0)
		return t2;

	return -1;
}


// kernel and device functions

/// <summary>
/// Finds the closest intersected RTObjectand sets the intersectPoint float3.
/// </summary>
/// <param name="ray">The ray to test RTObjectintersections.</param>
/// <param name="intersectPoint">The float3 to hold the intersection data.</param>
/// <returns>The closest intersected RTObject, or null if no RTObject is intersected.</returns>
__device__
void *getClosestIntersection(Ray ray, float3 *intersectPoint)
{
	float minDist = FLT_MAX;
	float curDist;
	void *isectTri = NULL;

	unsigned int i;
	for(i = 0; i < d_numTriangles; ++i)
	{
		Triangle *t = &d_Triangles[i];
		curDist = intersects(t, ray);
		if (curDist > 0 && curDist < minDist)
		{
			minDist = curDist;
			intersected = (void *)t;
		}
	}

	for(i = 0; i < d_numSpheres; i++)
	{
		Sphere *s = d_Spheres[i];
		curDist = intersects(s, ray);
		if (curDist > 0 && curDist < minDist)
		{
			minDist = curDist;
			intersected = (void *)s;
		}
	}

	if(intersected)
		*intersectPoint = ray.Position + ray.Direction * minDist;

	return intersected;
}

__device__
float4 calculateAmbient(Material *m)
{
	float4 ambientLight = d_ambientLight;
	if(m) ambientLight = ambientLight * m->ambientColor * m->ambientStrength;
	return ambientLight;
}

__device__
float4 calculateDiffuse(Material *m, float3 worldCoords, Light l, float3 normal, float3 lightVector) {
	float4 diffuseLight = l.LightColor;
	if (m)
		diffuseLight = diffuseLight *
			fabs(Dot(lightVector, normal)) * 
			m->getDiffuseColor() * 
			m->diffuseStrength;
	return diffuseLight;
}

__device__
float4 calculateSpecular(Material *m, float3 worldCoords, Light l, float3 normal, float3 lightVector, float3 viewVector) {
	float3 reflectedVector = Reflect(lightVector, normal);
	float dot = Dot(reflectedVector, viewVector);

	if (dot >= 0)
	    return make_float4(0, 0, 0, 0);

	float4 specularLight = l.LightColor;
	
	if (m)
	{
		specularLight = specularLight *
			fabs(Dot(lightVector, normal) * pow(dot, m->exponent)) *
			m->specularColor *
			m->specularStrength;
	}

	return specularLight;
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
float4 spawnShadowRay(float3 intersectPoint, void *intersectedObject, Material *m, /*ObjectType t,*/ float3 intersectNormal, float3 viewVector, int depth)
{
	float4 diffuseTotal;
	float4 specularTotal;

	unsigned int i;
	for(i = 0; i < numLights; ++i)
	{
		Light light = lights[i];

		// Spawn a shadow ray from the intersection point to the light source
		float3 lightVector = Normalize(light.Position - intersectPoint);

		// but only if the intersection is facing the light source
		float facing = Dot(intersectNormal, lightVector);
		if (facing < 0)
		{
			Ray shadowRay;
			shadowRay.Position = intersectPoint;
			shadowRay.Direction = lightVector;

			// Check if the shadow ray reaches the light before hitting any other object
			float dist = Distance(intersectPoint, light.Position);
			/*bool shadowed = false;

			float4 shadowLight;

			unsigned int k;
			for(k = 0; k < numTriangles; ++k)
			{
				Triangle t* = &d_triangles[k];
				if (*t != intersectedObject)
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
			}*/

			/*if (shadowed)
			{
				diffuseTotal = diffuseTotal + intersectedObject->calculateDiffuse(intersectPoint, intersectNormal, light, lightVector) * shadowLight;
				specularTotal = specularTotal + intersectedObject->calculateSpecular(intersectPoint, intersectNormal, light, lightVector, viewVector) * shadowLight;
			}
			else
			{*/
				diffuseTotal = diffuseTotal + calculateDiffuse(m, intersectPoint, light, intersectNormal, lightVector);
				specularTotal = specularTotal + calculateSpecular(m, intersectPoint, light, intersectNormal, lightVector, viewVector);
			//}

		}
	}

	return diffuseTotal * m->diffuseStrength + specularTotal * m->specularStrength;
}

__device__
float4 Illuminate(Ray ray, int depth) {
    float3 intersectPoint;
    void *rt = getClosestIntersection(ray, &intersectPoint);

    if (rt)
    {
        float3 intersectNormal;
	Material *m;
	if(type == T_Sphere)	
	{
		Sphere *s = (Sphere *)rt;
		intersectNormal = (intersectPoint - s.p).Normalize();
		m = &s.m;
	} else {
		Triangle *t = (Triangle *)rt;
		intersectNormal = t.n;	
		m = &t.m;
	}

        //float3 viewVector = (ray.Position - intersectPoint).Normalize();
        float3 viewVector = -ray.Direction;
        float4 totalLight = calculateAmbient(m);
        totalLight = totalLight + spawnShadowRay(intersectPoint, rt, m, intersectNormal, viewVector, depth);

        /*if (depth < recursionDepth)
        {
            float3 incidentVector = (intersectPoint - ray.Position).Normalize();

            // Material is reflective
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
        }*/

        return totalLight;
    }
    else
    {
        return backgroundColor;
    }
}
/*
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

	float dot = incidentVector.Dot(intersectNormal);
	float discriminant = 1 + ((n * n) * ((dot * dot) - 1));

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
*/
