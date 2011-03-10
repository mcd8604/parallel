/*
 * RTTriangle.h
 *
 *  Created on: Feb 26, 2011
 *      Author: mike
 */

#ifndef RTTRIANGLE_H_
#define RTTRIANGLE_H_

#include "RTObject.h"

namespace RayTracer {

class RTTriangle: public RTObject {
protected:
	Vector3 v1, v2, v3, n;
public:
	RTTriangle(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 n);
	virtual ~RTTriangle();
	Vector3 GetV1();
	Vector3 GetV2();
	Vector3 GetV3();
	Vector3 GetN();
	bool operator == (RTObject *o);
	bool operator != (RTObject *o);
	double Intersects(Ray ray);
	Vector3 GetIntersectNormal(Vector3 intersectPoint);
};

}

#endif /* RTTRIANGLE_H_ */
