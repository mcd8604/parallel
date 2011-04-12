/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <cutil_inline.h>
#include <cutil_math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <stdlib.h>
#include <time.h>
#include "RayTracer.h"

#include <stdio.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

//size_t pitch;
//__constant__ uint d_width;
//__constant__ uint d_height;
//__constant__ float4 d_ambientLight;
//__constant__ float4 d_backgroundColor;
//__constant__ uint d_numLights;
//__constant__ uint d_numTriangles;
//__constant__ uint d_numSpheres;
//__constant__ Light *d_lights;
//__constant__ Triangle *d_triangles;
//__constant__ Sphere *d_spheres;

__device__ bool operator ==(float3 a, float3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
__device__ bool operator !=(float3 a, float3 b) { return a.x != b.x && a.y != b.y && a.z != b.z; }
//__device__ float3 operator +(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
//__device__ float3 operator -(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
//__device__ float3 operator *(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
//__device__ float3 operator -(float3 a, float s) { return make_float3(a.x - s, a.y - s, a.z - s); }
//__device__ float3 operator -(float3 a) { return make_float3(-a.x , -a.y, -a.z); }
//__device__ float3 operator *(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
//__device__ float3 operator /(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
__device__ float Dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float Dot(float3 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float Dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ float Distance(float3 a, float3 b) { int dX, dY, dZ; dX = b.x - a.x; dY = b.y - a.y; dZ = b.z - a.z; return sqrtf(dX * dX + dY * dY + dZ * dZ); }
__device__ float3 Reflect(float3 v, float3 n) { return v - n * Dot(v, n) * 2; }
__device__ float3 Normalize(float3 v) { float xx, yy, zz, d; xx = v.x * v.x; yy = v.y * v.y; zz = v.z * v.z; d = sqrt(xx + yy + zz); return make_float3( v.x / d, v.y / d, v.z / d); }

__device__ bool operator ==(float4 a, float4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
__device__ bool operator !=(float4 a, float4 b) { return a.x != b.x && a.y != b.y && a.z != b.z && a.w != b.w; }
//__device__ float4 operator +(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
//__device__ float4 operator -(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
//__device__ float4 operator *(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
//__device__ float4 operator *(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }

__device__ float3 operator *(const float3x4 M, const float3 v) {
    float3 r;
    r.x = Dot(v, M.m[0]);
    r.y = Dot(v, M.m[1]);
    r.z = Dot(v, M.m[2]);
    return r;
}

__device__ float4 operator *(const float3x4 M, const float4 v) {
    float4 r;
    r.x = Dot(v, M.m[0]);
    r.y = Dot(v, M.m[1]);
    r.z = Dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ bool operator ==(Sphere s1, Sphere s2) { return s1.r == s2.r && s1.p == s2.p; }
__device__ bool operator !=(Sphere s1, Sphere s2) { return s1.r != s2.r && s1.p != s2.p; }

__device__ bool operator ==(Triangle t1, Triangle t2) { return t1.v1 == t2.v1 && t1.v2 == t2.v2 && t1.v3 == t2.v3 && t1.n == t2.n; }
__device__ bool operator !=(Triangle t1, Triangle t2) { return t1.v1 != t2.v1 && t1.v2 != t2.v2 && t1.v3 != t2.v3 && t1.n != t2.n;  }

__device__ bool operator ==(Material m1, Material m2) {
                return m1.ambientColor == m2.ambientColor &&
                        m1.diffuseColor == m2.diffuseColor &&
                        m1.specularColor == m2.specularColor &&
                        m1.kR == m2.kR && m1.kT == m2.kT && m1.n == m2.n &&
                        m1.ambientStrength == m2.ambientStrength &&
                        m1.diffuseStrength == m2.diffuseStrength &&
                        m1.specularStrength == m2.specularStrength; }
        bool operator !=(Material m1, Material m2) {
                return m1.ambientColor != m2.ambientColor &&
                        m1.diffuseColor != m2.diffuseColor &&
                        m1.specularColor != m2.specularColor &&
                        m1.kR != m2.kR && m1.kT != m2.kT && m1.n != m2.n &&
                        m1.ambientStrength != m2.ambientStrength &&
                        m1.diffuseStrength != m2.diffuseStrength &&
			m1.specularStrength != m2.specularStrength; }
// Kernel functions
//__global__ void trace(uint *d_pixelData);
/*__device__ float4 illuminate(Ray ray, int depth,
		uint d_width, uint d_height,
		float4 d_ambientLight, float4 d_backgroundColor,
		uint d_numLights, Light *d_lights,
		uint d_numTriangles, Triangle *d_triangles,
		uint d_numSpheres, Sphere *d_spheres);
__device__ float intersects(Sphere *s, Ray r);
__device__ float intersects(Triangle *t, Ray r);
__device__ void *getClosestIntersection(Ray r, float3 *intersectPoint, ObjectType *type,
		float4 d_ambientLight, float4 d_backgroundColor,
		uint d_numLights, Light *d_lights,
		uint d_numTriangles, Triangle *d_triangles,
		uint d_numSpheres, Sphere *d_spheres);
__device__ float4 calculateAmbient(Material *m, float4 d_ambientLight);*/

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    cutilSafeCall( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix) );
}

// Copy scene data to device
//void SetSceneData(uint width, uint height, float4 ambientLight, float4 backgroundColor,
//		uint numLights, Light *lights,
//		uint numTriangles, Triangle *triangles,
//		uint numSpheres, Sphere *spheres)
//{
//	cutilSafeCall( cudaMemcpyToSymbol(d_width, &width, sizeof(uint), cudaMemcpyHostToDevice ));
//	cutilSafeCall( cudaMemcpyToSymbol(d_height, &height, sizeof(uint), cudaMemcpyHostToDevice ));
//	cudaMemcpyToSymbol(d_ambientLight, &ambientLight, sizeof(float4));
//	cudaMemcpyToSymbol(d_backgroundColor, &backgroundColor, sizeof(float4));
//
//	//cudaMallocPitch((void **)&d_pixelData, &pitch, d_width * sizeof(float4), d_height);
//	//cudaMalloc((void **)&d_pixelData, d_width * d_height * sizeof(float4));
//
//	/*cudaMemcpyToSymbol(d_numLights, &numLights, sizeof(uint));
//	size_t sizeLights = numLights * sizeof(Light);
//	cudaMalloc((void **)&d_lights, sizeLights);
//	cudaMemcpyToSymbol(d_lights, lights, sizeLights);
//
//	cudaMemcpyToSymbol(d_numTriangles, &numTriangles, sizeof(uint));
//	size_t sizeTriangles = numTriangles * sizeof(Triangle);
//	cudaMalloc((void **)d_triangles, sizeTriangles);
//	cudaMemcpyToSymbol(d_triangles, triangles, sizeTriangles);
//
//	cudaMemcpyToSymbol(d_numSpheres, &numSpheres, sizeof(uint));
//	size_t sizeSpheres = numSpheres * sizeof(Sphere);
//	cudaMalloc((void **)d_spheres, sizeSpheres);
//	cudaMemcpyToSymbol(d_spheres, spheres, sizeSpheres);*/
//}
//
//void FreeSceneData()
//{
//	cudaFree(d_lights);
//	cudaFree(d_triangles);
//	cudaFree(d_spheres);
//}

__device__
float intersects(Triangle *t, Ray r) {
	//http://www.siggraph.org/education/materials/HyperGraph/raytrace/raypolygon_intersection.htm
	float3 n = t->n;
	float d = Dot(r.Direction, n);
	if(d == 0)
		return -1;

	/*float3 ri = r.Position + r.Direction * d;

	// project to 2D
	float2 a, b, c, i;

	float m = max(max(abs(n.x), abs(n.y)),abs(n.z);
	int i = 0;
	if(m == n.y)
		i = 1;
	else if(m == n.y)
		i = 2;

	a = make_float2(t->v1[0], t->v1[1]);*/

	return Dot((r.Position - t->v1), n) / d;
}

__device__
float intersects(Sphere *s, Ray ray) {
	// quadratic equation: t = (-b +/= sqrt(b * b - 4 * a * c)) / 2 * a

	// float a = Dot(ray.Direction, ray.Direction);
	// since ray direction is normalized, this will always = (1 += round error)
	// omitting a will save on calculations and reduce error

	float r = s->r;
	float3 p = s->p;
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
void *getClosestIntersection(Ray ray, float3 *intersectPoint, ObjectType *type,
		float4 d_ambientLight, float4 d_backgroundColor,
				uint d_numLights, Light *d_lights,
				uint d_numTriangles, Triangle *d_triangles,
				uint d_numSpheres, Sphere *d_spheres)
{
	float minDist = CUDART_INF_F;
	float curDist;
	void *intersected = NULL;

	uint i;
	for(i = 0; i < d_numTriangles; ++i)
	{
		Triangle *t = &d_triangles[i];
		curDist = intersects(t, ray);
		if (curDist > 0 && curDist < minDist)
		{
			minDist = curDist;
			intersected = (void *)t;
			*type = T_Triangle;
		}
	}

	for(i = 0; i < d_numSpheres; i++)
	{
		Sphere *s = &d_spheres[i];
		curDist = intersects(s, ray);
		if (curDist > 0 && curDist < minDist)
		{
			minDist = curDist;
			intersected = (void *)s;
			*type = T_Sphere;
		}
	}

	if(intersected)
		*intersectPoint = ray.Position + ray.Direction * minDist;

	return intersected;
}

__device__
float4 calculateAmbient(Material *m, float4 d_ambientLight)
{
	float4 ambientLight = d_ambientLight;
	if(m) ambientLight = ambientLight * m->ambientColor * m->ambientStrength;
	return ambientLight;
}

__device__
float4 calculateDiffuse(Material *m, float3 worldCoords, Light l, float3 normal, float3 lightVector) {
	float4 diffuseLight = l.Color;
	if (m)
		diffuseLight = diffuseLight *
			fabs(Dot(lightVector, normal)) *
			m->diffuseColor *
			m->diffuseStrength;
	return diffuseLight;
}

__device__
float4 calculateSpecular(Material *m, float3 worldCoords, Light l, float3 normal, float3 lightVector, float3 viewVector) {
	float3 reflectedVector = Reflect(lightVector, normal);
	float dot = Dot(reflectedVector, viewVector);

	if (dot >= 0)
	    return make_float4(0, 0, 0, 0);

	float4 specularLight = l.Color;

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
float4 spawnShadowRay(float3 intersectPoint, void *intersectedObject, Material *m, /*ObjectType t,*/
		float3 intersectNormal, float3 viewVector, int depth,
			float4 d_ambientLight, float4 d_backgroundColor,
						uint d_numLights, Light *d_lights,
						uint d_numTriangles, Triangle *d_triangles,
						uint d_numSpheres, Sphere *d_spheres)
{
	float4 diffuseTotal;
	float4 specularTotal;

	uint i;
	for(i = 0; i < d_numLights; ++i)
	{
		Light light = d_lights[i];

		// Spawn a shadow ray from the intersection point to the light source
		float3 lightVector = Normalize(light.Position - intersectPoint);

		// but only if the intersection is facing the light source
		float facing = Dot(intersectNormal, lightVector);
		if (facing < 0)
		{
			/*Ray shadowRay;
			shadowRay.Position = intersectPoint;
			shadowRay.Direction = lightVector;

			// Check if the shadow ray reaches the light before hitting any other object
			float dist = Distance(intersectPoint, light.Position);
			bool shadowed = false;

			float4 shadowLight;

			uint k;
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
				//diffuseTotal = diffuseTotal + calculateDiffuse(m, intersectPoint, light, intersectNormal, lightVector);
				specularTotal = specularTotal + calculateSpecular(m, intersectPoint, light, intersectNormal, lightVector, viewVector);
			//}

		}
	}

	return diffuseTotal /** m->diffuseStrength*/ + specularTotal /** m->specularStrength*/;
}

__device__
float4 illuminate(Ray ray, int depth,
			float4 d_ambientLight, float4 d_backgroundColor,
					uint d_numLights, Light *d_lights,
					uint d_numTriangles, Triangle *d_triangles,
					uint d_numSpheres, Sphere *d_spheres) {
    float3 intersectPoint;
    ObjectType type;
    void *rt = getClosestIntersection(ray, &intersectPoint, &type,
    		d_ambientLight, d_backgroundColor,
			d_numLights, d_lights,
			d_numTriangles, d_triangles,
			d_numSpheres, d_spheres);

    if (rt)
    {
        float3 intersectNormal;
		Material *m;
		if(type == T_Sphere)
		{
			Sphere *s = (Sphere *)rt;
			intersectNormal = Normalize(intersectPoint - s->p);
			m = &(s->m);
		} else {
			Triangle *t = (Triangle *)rt;
			intersectNormal = t->n;
			m = &(t->m);
		}

        //float3 viewVector = Normalize(ray.Position - intersectPoint);
        float3 viewVector = -ray.Direction;
        float4 totalLight = calculateAmbient(m, d_ambientLight);
        totalLight = totalLight + spawnShadowRay(intersectPoint, rt, m, intersectNormal, viewVector, depth,
    			d_ambientLight, d_backgroundColor,
    			d_numLights, d_lights,
    			d_numTriangles, d_triangles,
    			d_numSpheres, d_spheres);

        /*if (depth < 2)
        {
            float3 incidentVector = Normalize(intersectPoint - ray.Position);

            // Material is reflective
            if (m->kR > 0)
            {
                float3 dir = Reflect(incidentVector, intersectNormal);
                Ray reflectionRay;
                reflectionRay.Position = intersectPoint;
                reflectionRay.Direction = dir;
                totalLight = totalLight + (illuminate(reflectionRay, depth + 1,
            			d_ambientLight, d_backgroundColor,
            			d_numLights, d_lights,
            			d_numTriangles, d_triangles,
            			d_numSpheres, d_spheres) * m->kR);
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
        return d_backgroundColor;
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
		return illuminate(reflectionRay, depth + 1) * m->n;
	}
	else
	{
		float3 dir = incidentVector * n + (intersectNormal * (n * dot - sqrt(discriminant)));
		Ray transRay;
		transRay.Position = intersectPoint;
		transRay.Direction = dir;
		return illuminate(transRay, depth + 1) * intersectedObject->GetMaterial()->kT;
	}
}
*/

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ void
d_render(uint *d_output, uint d_width, uint d_height,
		float4 ambientLight, float4 backgroundColor,
		uint numLights, Light *lights,
		uint numTriangles, Triangle *triangles,
		uint numSpheres, Sphere *spheres)
{
	uint x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= d_width) || (y >= d_height)) return;

    float u = (x / (float) d_width)*2.0f-1.0f;
    float v = (y / (float) d_height)*2.0f-1.0f;

    float4 p = c_invViewMatrix * make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    Ray ray;
    ray.Position.x = p.x;
    ray.Position.y = p.y;
    ray.Position.z = p.z;
    ray.Direction = Normalize(make_float3(u, v, -2.0f));
    ray.Direction = c_invViewMatrix * ray.Direction;

    //d_output[y*d_width + x] = rgbaFloatToInt(backgroundColor);
    //d_output[y*d_width + x] = rgbaFloatToInt(make_float4(1,0,0,1));
    d_output[y*d_width + x] = rgbaFloatToInt(illuminate(ray, 1,
			ambientLight, backgroundColor,
			numLights, lights,
			numTriangles, triangles,
			numSpheres, spheres));
}

__global__ void test_intersect(uint *d_output, uint d_width, uint d_height, Sphere *s)
{
	uint x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= d_width) || (y >= d_height)) return;

    float u = (x / (float) d_width)*2.0f-1.0f;
    float v = (y / (float) d_height)*2.0f-1.0f;

    float4 p = c_invViewMatrix * make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    Ray ray;
    ray.Position.x = p.x;
    ray.Position.y = p.y;
    ray.Position.z = p.z;
    ray.Direction = Normalize(make_float3(u, v, -2.0f));
    ray.Direction = c_invViewMatrix * ray.Direction;

    Sphere sphere1;
	sphere1.p = make_float3(3, 0, -70);
	sphere1.r = 1;

    float d = intersects(&sphere1, ray);
    uint o;
    if(d == -1)
    	o = rgbaFloatToInt(make_float4(0,0,0,0));
    else
    	o = rgbaFloatToInt(make_float4(1,0,0,1));
    d_output[y*d_width + x] = o;
}

extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output,
		uint width, uint height, float4 ambientLight, float4 backgroundColor,
				uint numLights, Light *lights,
				uint numTriangles, Triangle *triangles,
				uint numSpheres, Sphere *spheres)
{
//	uint d_width;
//	uint d_height;
//	float4 d_ambientLight;
//	float4 d_backgroundColor;
//	uint d_numLights;
//	uint d_numTriangles;
//	uint d_numSpheres;
	Light *d_lights;
	Triangle *d_triangles;
	Sphere *d_spheres;

	//cutilSafeCall( cudaMemcpyToSymbol(d_width, &width, sizeof(uint), cudaMemcpyHostToDevice ));
	//cutilSafeCall( cudaMemcpyToSymbol(d_height, &height, sizeof(uint), cudaMemcpyHostToDevice ));
	//cudaMemcpyToSymbol(d_ambientLight, &ambientLight, sizeof(float4));
	//cudaMemcpyToSymbol(d_backgroundColor, &backgroundColor, sizeof(float4));

	//cudaMallocPitch((void **)&d_pixelData, &pitch, d_width * sizeof(float4), d_height);
	//cudaMalloc((void **)&d_pixelData, d_width * d_height * sizeof(float4));

	//cudaMemcpyToSymbol(d_numLights, &numLights, sizeof(uint));
	size_t sizeLights = numLights * sizeof(Light);
	cutilSafeCall(cudaMalloc((void **)&d_lights, sizeLights));
	cutilSafeCall(cudaMemcpy(d_lights, lights, sizeLights, cudaMemcpyHostToDevice));

	//cudaMemcpyToSymbol(d_numTriangles, &numTriangles, sizeof(uint));
	size_t sizeTriangles = numTriangles * sizeof(Triangle);
	cutilSafeCall(cudaMalloc((void **)&d_triangles, sizeTriangles));
	cutilSafeCall(cudaMemcpy(d_triangles, triangles, sizeTriangles, cudaMemcpyHostToDevice));

	//cudaMemcpyToSymbol(d_numSpheres, &numSpheres, sizeof(uint));
	size_t sizeSpheres = numSpheres * sizeof(Sphere);
	cutilSafeCall(cudaMalloc((void **)&d_spheres, sizeSpheres));
	cutilSafeCall(cudaMemcpy(d_spheres, spheres, sizeSpheres, cudaMemcpyHostToDevice));

	d_render<<<gridSize, blockSize>>>( d_output, width, height,
			ambientLight, backgroundColor,
			numLights, d_lights,
			numTriangles, d_triangles,
			numSpheres, d_spheres);

	cutilSafeCall(cudaFree(d_lights));
	cutilSafeCall(cudaFree(d_triangles));
	cutilSafeCall(cudaFree(d_spheres));
	//test_intersect<<<gridSize, blockSize>>>( d_output, width, height, &d_spheres[0]);
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
