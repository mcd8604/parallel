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

//http://www.siggraph.org/education/materials/HyperGraph/raytrace/raypolygon_intersection.htm
double RTTriangle::Intersects(Ray ray) {
	double d = (ray.Direction).Dot(n);
	if(d == 0)
		return -1;
    // get triangle edge vectors and plane normal
	Vector3 u, v;
	u = v2 - v1;
    v = v3 - v1;

    // intersect point of ray and plane
    float dist = (ray.Position - v1).Dot(n) / d;
    Vector3 i = ray.Position + ray.Direction * dist;

    // check if i inside t
    float    uu, uv, vv, wu, wv, D;
    uu = u.Dot(u);
    uv = u.Dot(v);
    vv = v.Dot(v);
    Vector3 w = i - v1;
    wu = w.Dot(u);
    wv = w.Dot(v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)        // I is outside T
        return -1;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
        return -1;

	return dist;
}

}
