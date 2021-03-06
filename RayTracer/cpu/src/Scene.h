/*
 * Scene.h
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#ifndef SCENE_H_
#define SCENE_H_

#include "RTObject.h"
#include "Geom.h"
//#include "Matrix.h"
#include "float.h"
#include <vector>

namespace RayTracer {

#define PI 3.14159265
#define DEF_L_MAX 100.0;
#define numerator (1.219 + 4.781762499)// Used in scale factor for tone reproduction
#define TRANSMIT_SHADOW true

/// <summary>
/// Tone Reproduction Operators
/// </summary>
enum TROp
{
	None,
	Ward,
	Reinhard
};

class Scene {
private:
	unsigned int width;
	unsigned int height;
	//Matrix world;
	//Matrix view;
	//Matrix proj;
	int recursionDepth;
	Vector3 cameraPos;
	Vector3 cameraTarget;
	Vector3 cameraUp;
	double fovy;
	double nearPlaneDistance;
	double farPlaneDistance;
	std::vector<Ray> rayTable;
	Vector4 ambientLight;
	Vector4 backgroundColor;
	//float lMax = DEF_L_MAX;
	//TROp trOp = TROp.None;
	std::vector<Light> lights;
	std::vector<RTObject *> worldObjects;

	void initViewProj();
	void updateViewProj();
	void createRayTable();
	//Vector3 unProject(double winX, double winY, double winZ, Matrixd4x4 model, Matrixd4x4 proj, Vector4 view);
	Vector4 Illuminate(Ray ray, int depth);
	Vector4 spawnTransmissionRay(
			int depth,
			Vector3 intersectPoint,
			RTObject *intersectObject,
			Vector3 intersectNormal,
			Vector3 incidentVector);
	Vector4 spawnShadowRay(
			Vector3 intersectPoint,
			//RTObject *intersectedObject,
			int objI,
			Vector3 intersectNormal,
			Vector3 viewVector,
			int depth);
	//RTObject *getClosestIntersection(
	int getClosestIntersection(
			Ray ray,
			Vector3 *intersectPt);
	//void applyToneReproduction();
	//void reinhardOp(Vector4 colorData[], float logAvg);
	//void wardOp(Vector4 colorData[], float logAvg);
public:
	Scene();
	virtual ~Scene();
	void SetRecursionDepth(int rDepth);
	int GetRecursionDepth();
	void SetViewProjection(Vector3 pos, Vector3 tar, Vector3 up,
			double fovy, unsigned int width, unsigned int height, double near, double far);
	Vector3 GetCameraPosition();
	void SetCameraPosition(Vector3 p);
	void SetAmbient(Vector4 color);
	void SetBackground(Vector4 color);
	void AddLight(Vector3 pos, Vector4 clr);
	void AddObject(RTObject *rt);
	void trace(float *colorData, int id, int t);
};

}

#endif /* SCENE_H_ */
