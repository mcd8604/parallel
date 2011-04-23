/*
 * Scene.cpp
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#include "Scene.h"
#include <iostream>
#include <GL/glut.h>

#define TRANSMIT_SHADOW 0

namespace RayTracer {

Scene::Scene() { }

Scene::~Scene() {
	unsigned int i;
	for (i = 0; i < worldObjects.size(); ++i)
		delete worldObjects[i];
}

void Scene::SetRecursionDepth(int rDepth) {
	recursionDepth = rDepth;
}

int Scene::GetRecursionDepth() {
	return recursionDepth;
}

void Scene::updateViewProj() {
	if(width > 0 && height > 0) {
		//rayTable.reserve(width * height);
		rayTable.resize(width * height);

		// use OpenGL to unproject
		GLdouble model[16];
		GLdouble proj[16];
		GLint view[4];
		view[0] = 0;
		view[1] = 0;
		view[2] = width;
		view[3] = height;

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		gluPerspective(fovy, GLdouble(width)/height, nearPlaneDistance, farPlaneDistance);
		glGetDoublev(GL_PROJECTION_MATRIX, proj);
		glPopMatrix();

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		gluLookAt(cameraPos.x, cameraPos.y, cameraPos.z,
				cameraTarget.x, cameraTarget.y, cameraTarget.z,
				cameraUp.x, cameraUp.y, cameraUp.z);
		glGetDoublev(GL_MODELVIEW_MATRIX, model);
		//glGetIntegerv(GL_VIEWPORT, view);
		glPopMatrix();

		int x, y;
		for(y = 0; y < height; ++y)
		{
			for(x = 0; x < width; ++x)
			{
				Vector3 rayS;
				Vector3 rayE;

				gluUnProject(x, y, 0, model, proj, view, &rayS.x, &rayS.y, &rayS.z);
				gluUnProject(x, y, 1, model, proj, view, &rayE.x, &rayE.y, &rayE.z);

				Ray r;
				r.Position = rayS;
				r.Direction = (rayE - rayS).Normalize();
				rayTable[x + y * width] = r;
			}
		}
	}
}

Vector3 Scene::GetCameraPosition() {
	return cameraPos;
}

void Scene::SetCameraPosition(Vector3 p) {
	cameraPos = p;
	updateViewProj();
}

void Scene::SetViewProjection(Vector3 pos, Vector3 tar, Vector3 up,
		double fovy, unsigned int width, unsigned int height, double near, double far) {
	cameraPos = pos;
	cameraTarget = tar;
	cameraUp = up;
	this->fovy = fovy;
	this->width = width;
	this->height = height;
	nearPlaneDistance = near;
	farPlaneDistance = far;

	//if(!rayTable)
		//rayTable = new Ray[width * height];
	updateViewProj();
}

void Scene::SetAmbient(Vector4 color) {
	ambientLight = color;
}

void Scene::SetBackground(Vector4 color) {
	backgroundColor = color;
}

void Scene::AddLight(Vector3 pos, Vector4 clr) {
	Light l;
	l.Position = pos;
	l.LightColor = clr;
	lights.push_back(l);
}

void Scene::AddObject(RTObject *rt) {
	worldObjects.push_back(rt);
}

void Scene::trace(float *colorData, int id, int t) {
    //colorData = new Vector4[width * height];

    for (int y = id; y < height; y+=t)
    {
        for (int x = 0; x < width; ++x)
        {
            int i= (y * width) + x;
            Vector4 c = Illuminate(rayTable[i], 0);
            colorData[i * 4] = c.x;
            colorData[i * 4 + 1] = c.y;
            colorData[i * 4 + 2] = c.z;
            colorData[i * 4 + 3] = c.w;
        }
    }
    //if(trOp != TROp.None)
    //    applyToneReproduction(colorData);
}

Vector4 Scene::Illuminate(Ray ray, int depth) {

    Vector3 intersectPoint;
    int i = getClosestIntersection(ray, &intersectPoint);

    if(i != -1)
    {
        RTObject *rt = worldObjects[i];
        Vector3 intersectNormal = rt->GetIntersectNormal(intersectPoint);

        //Vector3 viewVector = (ray.Position - intersectPoint).Normalize();
        Vector3 viewVector = -ray.Direction;
        Vector4 totalLight = rt->calculateAmbient(ambientLight, intersectPoint);
        //totalLight = totalLight + spawnShadowRay(intersectPoint, rt, intersectNormal, viewVector, depth);
        totalLight = totalLight + spawnShadowRay(intersectPoint, i, intersectNormal, viewVector, depth);

        if (depth < recursionDepth)
        {
            Vector3 incidentVector = (intersectPoint - ray.Position).Normalize();

            // Material is reflective
            Material *m = rt->GetMaterial();
            if (m->kR > 0)
            {
                Vector3 dir = incidentVector.Reflect(intersectNormal);
                Ray reflectionRay;
                reflectionRay.Position = intersectPoint;
                reflectionRay.Direction = dir;
                totalLight = totalLight + (Illuminate(reflectionRay, depth + 1) * m->kR);
            }

            // Material is transparent
            //if (m->kT > 0)
            //{
            //    totalLight = totalLight + spawnTransmissionRay(depth, intersectPoint, rt, intersectNormal, incidentVector);
            //}
        }

        return totalLight;
    }
    else
    {
        return backgroundColor;
    }
}

/// <summary>
/// Spawns a recursive, transmitted (refracted) ray.
/// </summary>
/// <param name="depth">Current recursion depth</param>
/// <param name="intersectPoint">Origin of the ray</param>
/// <param name="intersectedObject">World object that was intersected</param>
/// <param name="intersectNormal">Normal of the world object at the intersection point</param>
/// <param name="totalLight">Total light to contribute to.</param>
/// <param name="incidentVector">Ray direction incident to intersection.</param>
Vector4 Scene::spawnTransmissionRay(int depth, Vector3 intersectPoint,
		RTObject *intersectedObject, Vector3 intersectNormal, Vector3 incidentVector)
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
		intersectNormal = -intersectNormal;
	}

	double dot = incidentVector.Dot(intersectNormal);
	double discriminant = 1 + ((n * n) * ((dot * dot) - 1));

	if (discriminant > 0)
	{
		// simulate total internal reflection
		Vector3 dir = incidentVector.Reflect(intersectNormal);
		Ray reflectionRay;
		reflectionRay.Position = intersectPoint;
		reflectionRay.Direction = dir;
		return Illuminate(reflectionRay, depth + 1) * m->n;
	}
	else
	{
		Vector3 dir = incidentVector * n + (intersectNormal * (n * dot - sqrt(discriminant)));
		Ray transRay;
		transRay.Position = intersectPoint;
		transRay.Direction = dir;
		return Illuminate(transRay, depth + 1) * m->kT;
	}
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
Vector4 Scene::spawnShadowRay(Vector3 intersectPoint, //RTObject *intersectedObject,
		int objI, Vector3 intersectNormal, Vector3 viewVector, int depth)
{
	Vector4 diffuseTotal;
	Vector4 specularTotal;
	RTObject *intersectedObject = worldObjects[objI];

	unsigned int i;
	for(i = 0; i < lights.size(); ++i)
	{
		Light light = lights[i];

		// Spawn a shadow ray from the intersection point to the light source
		Vector3 lightVector = (light.Position - intersectPoint).Normalize();

		// but only if the intersection is facing the light source
		float facing = intersectNormal.Dot(lightVector);
		if (facing > 0)
		{
			Ray shadowRay;
			shadowRay.Position = intersectPoint;
			shadowRay.Direction = lightVector;

			// Check if the shadow ray reaches the light before hitting any other object
			float dist = intersectPoint.Distance(light.Position);
			bool shadowed = false;

			Vector4 shadowLight = Vector4(0, 0, 0, 0);

			unsigned int k;
			for(k = 0; k < worldObjects.size(); ++k)
			{
				//if (*rt != intersectedObject)
				if (k != objI)
				{
					RTObject *rt = worldObjects[k];
				    //Vector3 intersectPoint2;
				    //int objI2 = getClosestIntersection(ray, &intersectPoint2);
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
							Vector3 incidentVector = (intersectPoint - shadowRay.Position).Normalize();
							Vector3 shadowIntersect = shadowRay.Position + (shadowRay.Direction * curDist);
							Vector3 shadowNormal = rt->GetIntersectNormal(shadowIntersect);

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
			}

			if (shadowed)
			{
//				diffuseTotal = diffuseTotal + intersectedObject->calculateDiffuse(intersectPoint, intersectNormal, light, lightVector) * shadowLight;
//				specularTotal = specularTotal + intersectedObject->calculateSpecular(intersectPoint, intersectNormal, light, lightVector, viewVector) * shadowLight;
			}
			else
			{
				diffuseTotal = diffuseTotal + intersectedObject->calculateDiffuse(intersectPoint, intersectNormal, light, lightVector);
				specularTotal = specularTotal + intersectedObject->calculateSpecular(intersectPoint, intersectNormal, light, lightVector, viewVector);
			}

		}
	}

	Material *m = intersectedObject->GetMaterial();
	return diffuseTotal * m->diffuseStrength + specularTotal * m->specularStrength;
}

/// <summary>
/// Finds the closest intersected RTObjectand sets the intersectPoint Vector3.
/// </summary>
/// <param name="ray">The ray to test RTObjectintersections.</param>
/// <param name="intersectPoint">The Vector3 to hold the intersection data.</param>
/// <returns>The closest intersected RTObject, or null if no RTObject is intersected.</returns>
int Scene::getClosestIntersection(Ray ray, Vector3 *intersectPoint)
{
	float minDist = FLT_MAX;
	float curDist;
	//RTObject *intersected = NULL;
	int intersected = -1;

	unsigned int i;
	for(i = 0; i < worldObjects.size(); ++i)
	{
		RTObject *rt = worldObjects[i];
		curDist = rt->Intersects(ray);
		if (curDist > 0 && curDist < minDist)
		{
			minDist = curDist;
			//intersected = rt;
			intersected = i;
		}
	}

	if(intersected != -1)
		*intersectPoint = ray.Position + ray.Direction * minDist;

	return intersected;
}

}
