/*
 * RayTracer.h
 *
 *  Created on: Mar 12, 2011
 *      Author: mike
 */

#ifndef RAYTRACER_H_
#define RAYTRACER_H_

#include <vector_types.h>

inline bool operator ==(float3 a, float3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
inline bool operator !=(float3 a, float3 b) { return a.x != b.x && a.y != b.y && a.z != b.z; }

inline bool operator ==(float4 a, float4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
inline bool operator !=(float4 a, float4 b) { return a.x != b.x && a.y != b.y && a.z != b.z && a.w != b.w; }

__device__ float3 operator +(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 operator -(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 operator *(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ float3 operator -(float3 a, float s) { return make_float3(a.x - s, a.y - s, a.z - s); }
__device__ float3 operator -(float3 a) { return make_float3(-a.x , -a.y, -a.z); }
__device__ float3 operator *(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float3 operator /(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
__device__ float Dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float Distance(float3 a, float3 b) { int dX, dY, dZ; dX = b.x - a.x; dY = b.y - a.y; dZ = b.z - a.z; return sqrt(dX * dX + dY * dY + dZ * dZ); }
__device__ float3 Reflect(float3 v, float3 n) { return v - n * Dot(v, n) * 2; }
__device__ float3 Normalize(float3 v) { float xx, yy, zz, d; xx = v.x * v.x; yy = v.y * v.y; zz = v.z * v.z; d = sqrt(xx + yy + zz); return make_float3( v.x / d, v.y / d, v.z / d); }

struct float3x4
{
	float4 m[3];
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
        bool operator ==(Material m) {
                return m.ambientColor == ambientColor &&
                        m.diffuseColor == diffuseColor &&
                        m.specularColor == specularColor &&
                        m.kR == kR && m.kT == kT && m.n == n &&
                        m.ambientStrength == ambientStrength &&
                        m.diffuseStrength == diffuseStrength &&
                        m.specularStrength == specularStrength; }
        bool operator !=(Material m) {
                return m.ambientColor != ambientColor &&
                        m.diffuseColor != diffuseColor &&
                        m.specularColor != specularColor &&
                        m.kR != kR && m.kT != kT && m.n != n &&
                        m.ambientStrength != ambientStrength &&
                        m.diffuseStrength != diffuseStrength &&
			m.specularStrength != specularStrength; }
};

enum ObjectType { T_Sphere, T_Triangle };

struct Sphere {
	float r;
	float3 p;
	Material m;
	bool operator ==(Sphere s) { return r == s.r && p == s.p; }
	bool operator !=(Sphere s) { return r != s.r && p != s.p; }
};

struct Triangle {
	float3 v1, v2, v3, n;
	Material m;
	bool operator ==(Triangle t) { return v1 == t.v1 && v2 == t.v2 && v3 == t.v3 && n == t.n; }
	bool operator !=(Triangle t) { return v1 != t.v1 && v2 != t.v2 && v3 != t.v3 && n != t.n;  }
};	

#endif /* RAYTRACER_H_ */
