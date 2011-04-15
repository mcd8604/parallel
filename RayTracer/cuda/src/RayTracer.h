/*
 * RayTracer.h
 *
 *  Created on: Mar 12, 2011
 *      Author: mike
 */

#ifndef RAYTRACER_H_
#define RAYTRACER_H_

#include <vector_types.h>
//#include <math.h>

//inline bool operator ==(float3 a, float3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
//inline bool operator !=(float3 a, float3 b) { return a.x != b.x && a.y != b.y && a.z != b.z; }

//inline bool operator ==(float4 a, float4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
//inline bool operator !=(float4 a, float4 b) { return a.x != b.x && a.y != b.y && a.z != b.z && a.w != b.w; }

//inline float3 operator +(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
//inline float3 operator -(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
//inline float3 operator *(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
//inline float3 operator -(float3 a, float s) { return make_float3(a.x - s, a.y - s, a.z - s); }
//inline float3 operator -(float3 a) { return make_float3(-a.x , -a.y, -a.z); }
//inline float3 operator *(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
//inline float3 operator /(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
//inline float Dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
//inline float Distance(float3 a, float3 b) { int dX, dY, dZ; dX = b.x - a.x; dY = b.y - a.y; dZ = b.z - a.z; return sqrt(dX * dX + dY * dY + dZ * dZ); }
//inline float3 Reflect(float3 v, float3 n) { return v - n * Dot(v, n) * 2; }
//inline float3 Normalize(float3 v) { float xx, yy, zz, d; xx = v.x * v.x; yy = v.y * v.y; zz = v.z * v.z; d = sqrt(xx + yy + zz); return make_float3( v.x / d, v.y / d, v.z / d); }

//inline float4 operator +(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
//inline float4 operator -(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
//inline float4 operator *(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
//inline float4 operator *(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }


struct float3x4
{
	float4 m[3];
};

struct float4x4
{
	float4 m[4];
};

struct Ray {
	float3 Position;
	float3 Direction;
};

struct Light {
	float3 Position;
	float4 Color;
};

struct Material {
	float4 ambientColor;
	float4 diffuseColor;
	float4 specularColor;
	float kR; // Reflectivity coeffecient.
	float kT; // Transparency coeffecient.
	float n; // Index of refraction
	float ambientStrength;
	float diffuseStrength;
	float specularStrength;
	float exponent;
};

enum ObjectType { T_Sphere, T_Triangle };

struct Sphere {
	float r;
	float3 p;
	Material m;
};

struct Triangle {
	float3 v1, v2, v3, n;
	Material m;
};	

#endif /* RAYTRACER_H_ */
