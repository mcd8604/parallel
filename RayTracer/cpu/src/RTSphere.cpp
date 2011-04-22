/*
 * RTSphere.cpp
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#include "RTSphere.h"
#include <limits>

namespace RayTracer {

RTSphere::RTSphere(Vector3 position, float radius) {
	this->r = radius;
	this->p = position;
}

bool RTSphere::operator ==(RTObject *o) {
	RTSphere *s = (RTSphere *)o;
	if(s)
		return this->GetMaterial() == o->GetMaterial() &&
				r == s->GetRadius() &&
				p == s->GetPosition();
	else
		return false;
}

bool RTSphere::operator !=(RTObject *o) {
	return this->GetMaterial() != o->GetMaterial();
	RTSphere *s = (RTSphere *)o;
	if(s)
		return this->GetMaterial() != o->GetMaterial() ||
				r != s->GetRadius() ||
				p != s->GetPosition();
	else
		return true;
}

float RTSphere::GetRadius() {
	return r;
}

Vector3 RTSphere::GetPosition() {
	return p;
}

double RTSphere::Intersects(Ray ray) {
	// quadratic equation: t = (-b +/= sqrt(b * b - 4 * a * c)) / 2 * a

	// double a = ray.Direction.Dot(ray.Direction);
	// since ray direction is normalized, this will always = (1 += round error)
	// omitting a will save on calculations and reduce error

	Vector3 diff = ray.Position - p;
	double b = 2.0 * ray.Direction.Dot(diff);
	double c = diff.Dot(diff) - r * r;

	// approximate if below precision quantum
	if (c < .001)//std::numeric_limits<double>::epsilon())
        return 0;

	double d = b * b - 4.0 * c;
	// unreal, no root
	if(d < 0)
		return -1;

	double e = sqrt(d);

	// first root
	double t1 = (-b - e) / 2.0;
	if(t1 >= 0)
		return t1;

	// second root
	double t2 = (-b + e) / 2.0;
	if(t2 >= 0)
		return t2;

	return -1;
}

Vector3 RTSphere::GetIntersectNormal(Vector3 intersectionPoint)
{
	return (intersectionPoint - p).Normalize();
}

}
