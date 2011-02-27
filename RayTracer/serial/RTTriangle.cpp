/*
 * RTTriangle.cpp
 *
 *  Created on: Feb 26, 2011
 *      Author: mike
 */

#include "RTTriangle.h"

namespace RayTracer {

RTTriangle::RTTriangle(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 n) {
	this->v1 = v1;
	this->v2 = v2;
	this->v3 = v3;
	this->n = n;
}

RTTriangle::~RTTriangle() {
	// TODO Auto-generated destructor stub
}

bool RTTriangle::operator ==(RTObject *o) {
	RTTriangle *t = (RTTriangle *)o;
	if(t)
		return this->GetMaterial() == o->GetMaterial() &&
				v1 == t->GetV1() &&
				v2 == t->GetV2() &&
				v3 == t->GetV3() &&
				n == t->GetN();
	else
		return false;
}

bool RTTriangle::operator !=(RTObject *o) {
	RTTriangle *t = (RTTriangle *)o;
	if(t)
		return this->GetMaterial() != o->GetMaterial() ||
				v1 != t->GetV1() ||
				v2 != t->GetV2() ||
				v3 != t->GetV3() ||
				n != t->GetN();
	else
		return true;
}

Vector3 RTTriangle::GetV1() { return v1; }
Vector3 RTTriangle::GetV2() { return v2; }
Vector3 RTTriangle::GetV3() { return v3; }
Vector3 RTTriangle::GetN() { return n; }
Vector3 RTTriangle::GetIntersectNormal(Vector3 intersectPoint) { return n; }

double RTTriangle::Intersects(Ray ray) {
	double d = (ray.Direction).Dot(n);
	if(d == 0)
		return -1;

	return (ray.Position - v1).Dot(n) / d;
}

}
