/*
 * RTSphere.h
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#ifndef RTSPHERE_H_
#define RTSPHERE_H_

#include "RTObject.h"

namespace RayTracer {

class RTSphere: public RTObject {
protected:
	float r;
	Vector3 p;
public:
	float GetRadius();
	Vector3 GetPosition();
	bool operator == (RTObject *o);
	bool operator != (RTObject *o);
	RTSphere(Vector3 position, float radius);
	double Intersects(Ray ray);
	Vector3 GetIntersectNormal(Vector3 intersectPoint);
};

}

#endif /* RTSPHERE_H_ */
