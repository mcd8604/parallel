/*
 * Scene.cpp
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#include "Scene.h"

#include <iostream>

//#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

void Scene::SetViewProjection(Vector3 pos, Vector3 tar, Vector3 up,
		double fovy, double width, double height, double near, double far) {
	cameraPos = pos;
	cameraTarget = tar;
	cameraUp = up;
	this->width = width;
	this->height = height;
	nearPlaneDistance = near;
	farPlaneDistance = far;

	// create ray table

	/*Matrixd4x4 proj;
	double range = tan((fovy / 2.0)) * near;
	double left = -range * aspect;
	double right = range * aspect;
	double bottom = -range;
	double top = range;

	proj.data[0][0] = (2.0 * near) / (right - left);
	proj.data[1][1] = (2.0 * near) / (top - bottom);
	proj.data[2][2] = - (far + near) / (far - near);
	proj.data[2][3] = - 1.0;
	proj.data[3][2] = - (2.0 * far * near) / (far - near);

	Matrixd4x4 model;
	model.data[0][0] = cameraPos.x;
	model.data[1][1] = cameraPos.y;
	model.data[2][2] = cameraPos.z;
	model.data[3][3] = 1;*/

	//Matrixd4x4 camRot;
	//model = model * camRot;

	//Vector4 viewport = Vector4(0, 0, width, height);

	glm::dmat4x4 proj = glm::gtc::matrix_transform::perspectiveFov(fovy, width, height, near, far);

	glm::dvec3 vEye = glm::dvec3(cameraPos.x, cameraPos.y, cameraPos.z);
	glm::dvec3 vCenter = glm::dvec3(cameraTarget.x, cameraTarget.y, cameraTarget.z);
	glm::dvec3 vUp = glm::dvec3(cameraUp.x, cameraUp.y, cameraUp.z);
	glm::dmat4x4 lookat = glm::gtc::matrix_transform::lookAt(vEye, vCenter, vUp);

	glm::dvec4 viewport = glm::dvec4(0, 0, width, height);

	//if(!rayTable)
		//rayTable = new Ray[width * height];
	rayTable.reserve(width * height);

	int x, y;
	for(y = 0; y < height; ++y)
	{
		for(x = 0; x < width; ++x)
		{
			//Vector3 s = unProject(x, y, 0, model, proj, viewport);
			//Vector3 e = unProject(x, y, 1, model, proj, viewport);

			glm::dvec3 winS = glm::dvec3(x, y, 0);
			glm::dvec3 s = glm::unProject(winS, lookat, proj, viewport);

			glm::dvec3 winE = glm::dvec3(x, y, 1);
			glm::dvec3 e = glm::unProject(winE, lookat, proj, viewport);

			Vector3 rayS = Vector3(s.x, s.y, s.z);
			Vector3 rayE = Vector3(e.x, e.y, e.z);

			Ray r;
			r.Position = rayS;
			r.Direction = (rayE - rayS).Normalize();
			rayTable[x + y * width] = r;
		}
	}
}

/*Vector3 Scene::unProject(double winX, double winY, double winZ, Matrixd4x4 model, Matrixd4x4 proj, Vector4 viewport) {
	Matrixd4x4 inverse = (proj * model).inverse();
	Vector4 v1 = Vector4(
			(winX - viewport[0]) / viewport[2],
			(winY - viewport[1]) / viewport[3],
			winZ,
			1);
	v1 = v1 * 2 - 1;
	Vector4 obj = inverse * v1;
	obj = obj / obj.w;

	return Vector3(obj.x, obj.y, obj.z);
}*/

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

void Scene::trace(Vector4 *colorData) {
    //colorData = new Vector4[width * height];

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int i= (y * width) + x;
            colorData[i] = Illuminate(rayTable[i], 0);
        }
    }
    //if(trOp != TROp.None)
    //    applyToneReproduction(colorData);
}

Vector4 Scene::Illuminate(Ray ray, int depth) {

    Vector3 intersectPoint;
    RTObject *rt = getClosestIntersection(ray, &intersectPoint);

    if (rt)
    {
        Vector3 intersectNormal = rt->GetIntersectNormal(intersectPoint);

        //Vector3 viewVector = (ray.Position - intersectPoint).Normalize();
        Vector3 viewVector = -ray.Direction;
        Vector4 totalLight = rt->calculateAmbient(ambientLight, intersectPoint);
        totalLight = totalLight + spawnShadowRay(intersectPoint, rt, intersectNormal, viewVector, depth);

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
            if (m->kT > 0)
            {
                totalLight = totalLight + spawnTransmissionRay(depth, intersectPoint, rt, intersectNormal, incidentVector);
            }
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
Vector4 Scene::spawnTransmissionRay(int depth, Vector3 intersectPoint, RTObject *intersectedObject, Vector3 intersectNormal, Vector3 incidentVector)
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

	double dot = incidentVector.Dot(intersectNormal);
	double discriminant = 1 + ((n * n) * ((dot * dot) - 1));

	if (discriminant < 0)
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
		return Illuminate(transRay, depth + 1) * intersectedObject->GetMaterial()->kT;
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
Vector4 Scene::spawnShadowRay(Vector3 intersectPoint, RTObject *intersectedObject, Vector3 intersectNormal, Vector3 viewVector, int depth)
{
	Vector4 diffuseTotal;
	Vector4 specularTotal;

	unsigned int i;
	for(i = 0; i < lights.size(); ++i)
	{
		Light light = lights[i];

		// Spawn a shadow ray from the intersection point to the light source
		Vector3 lightVector = (light.Position - intersectPoint).Normalize();

		// but only if the intersection is facing the light source
		float facing = intersectNormal.Dot(lightVector);
		if (facing < 0)
		{
			Ray shadowRay;
			shadowRay.Position = intersectPoint;
			shadowRay.Direction = lightVector;

			// Check if the shadow ray reaches the light before hitting any other object
			float dist = intersectPoint.Distance(light.Position);
			bool shadowed = false;

			Vector4 shadowLight;

			unsigned int k;
			for(k = 0; k < worldObjects.size(); ++k)
			{
				RTObject *rt = worldObjects[k];
				if (*rt != intersectedObject)
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
				diffuseTotal = diffuseTotal + intersectedObject->calculateDiffuse(intersectPoint, intersectNormal, light, lightVector) * shadowLight;
				specularTotal = specularTotal + intersectedObject->calculateSpecular(intersectPoint, intersectNormal, light, lightVector, viewVector) * shadowLight;
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
RTObject *Scene::getClosestIntersection(Ray ray, Vector3 *intersectPoint)
{
	float minDist = FLT_MAX;
	float curDist;
	RTObject *intersected = NULL;

	unsigned int i;
	for(i = 0; i < worldObjects.size(); ++i)
	{
		RTObject *rt = worldObjects[i];
		curDist = rt->Intersects(ray);
		if (curDist > 0 && curDist < minDist)
		{
			minDist = curDist;
			intersected = rt;
		}
	}

	if(intersected)
		*intersectPoint = ray.Position + ray.Direction * minDist;

	return intersected;
}

}
