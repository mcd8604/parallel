/*
 * Geom.h
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#ifndef GEOM_H_
#define GEOM_H_

#include <math.h>

namespace RayTracer {

struct Vector3 {
	double x, y, z;
	Vector3() { x = 0; y = 0; z = 0; }
	Vector3(double a, double b, double c) { x = a; y = b; z = c; }
	Vector3 operator +(Vector3 v) { return Vector3(x + v.x, y + v.y, z + v.z); }
	Vector3 operator -(Vector3 v) { return Vector3(x - v.x, y - v.y, z - v.z); }
	Vector3 operator *(Vector3 v) { return Vector3(x * v.x, y * v.y, z * v.z); }
	Vector3 operator -(double s) { return Vector3(x - s, y - s, z - s); }
	Vector3 operator -() { return Vector3(-x , -y, -z); }
	Vector3 operator *(double s) { return Vector3(x * s, y * s, z * s); }
	Vector3 operator /(double s) { return Vector3(x / s, y / s, z / s); }
	bool operator ==(Vector3 v) { return x == v.x && y == v.y && z == v.z; }
	bool operator !=(Vector3 v) { return x != v.x || y != v.y || z != v.z; }
	Vector3 Reflect(Vector3 n) { Vector3 v(x, y, z); return v - n * v.Dot(n) * 2; }
	Vector3 Normalize() { double xx, yy, zz, d;	xx = x * x; yy = y * y; zz = z * z; d = sqrt(xx + yy + zz); return Vector3( x / d, y / d, z / d); }
	double Dot(Vector3 v) { return x * v.x + y * v.y + z * v.z; }
	double Distance(Vector3 v) { double dX, dY, dZ; dX = v.x - x; dY = v.y -y; dZ = v.z - z; return sqrt(dX * dX + dY * dY + dZ * dZ); }
	double DistanceSq(Vector3 v) { double dX, dY, dZ; dX = v.x - x; dY = v.y -y; dZ = v.z - z; return dX * dX + dY * dY + dZ * dZ; }
};

struct Vector4 {
	double x, y, z, w;
	Vector4() { x = 0; y = 0; z = 0; w = 0; }
	Vector4(Vector3 v, double _w) { x = v.x; y = v.y; z = v.z; w = _w; }
	Vector4(double a, double b, double c, double d) { x = a; y = b; z = c; w = d; }
	double operator [](int i) { return (&x)[i]; }
	Vector4 operator +(Vector4 v) { return Vector4(x + v.x, y + v.y, z + v.z, w + v.w); }
	Vector4 operator -(Vector4 v) { return Vector4(x - v.x, y - v.y, z - v.z, w - v.w); }
	Vector4 operator *(Vector4 v) { return Vector4(x * v.x, y * v.y, z * v.z, w * v.w); }
	Vector4 operator -(double s) { return Vector4(x - s, y - s, z - s, w - s); }
	Vector4 operator *(double s) { return Vector4(x * s, y * s, z * s, w * s); }
	Vector4 operator /(double s) { return Vector4(x / s, y / s, z / s, w / s); }
	bool operator ==(Vector4 v) { return x == v.x && y == v.y && z == v.z && w == v.w; }
	bool operator !=(Vector4 v) { return x != v.x || y != v.y || z != v.z || w != v.w; }
	Vector4 Reflect(Vector4 n) { Vector4 v(x, y, z, w); return v - n * v.Dot(n) * 2; }
	Vector4 Normalize() { double xx, yy, zz, ww, d; xx = x * x;	yy = y * y;	zz = z * z; ww = w * w; sqrt(d = xx + yy + zz + ww); return Vector4(x / d, y / d, z / d, w / d); }
	double Dot(Vector4 v) { return x * v.x + y * v.y + z * v.z + w * v.w; }
	double Distance(Vector4 v) { double dX, dY, dZ, dW; dX = v.x - x; dY = v.y -y; dZ = v.z - z; dW = v.w - w; return sqrt(dX * dX + dY * dY + dZ * dZ + dW * dW); }
	double DistanceSq(Vector4 v) { double dX, dY, dZ, dW; dX = v.x - x; dY = v.y -y; dZ = v.z - z; dW = v.w - w; return dX * dX + dY * dY + dZ * dZ + dW * dW; }
};

struct Ray {
	Vector3 Position;
	Vector3 Direction;
};

struct Light {
	Vector3 Position;
	Vector4 LightColor;
};

}

#endif /* GEOM_H_ */
