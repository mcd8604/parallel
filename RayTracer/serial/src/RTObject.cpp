/*
 * RTObject.cpp
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#include "RTObject.h"
#include <math.h>

namespace RayTracer {

float RTObject::getU(Vector3 worldCoords) { return 0; }
float RTObject::getV(Vector3 worldCoords) { return 0; }

Material *RTObject::GetMaterial() {
	return material1;
}

void RTObject::SetMaterial(Material *material) {
	material1 = material;
	isTextured = false;
}

void RTObject::SetMaterial(Material *material, float maxU, float maxV) {
	material1 = material;
	this->maxU = maxU;
	this->maxV = maxV;
	isTextured = true;
}

Vector4 RTObject::calculateAmbient(Vector4 worldAmbient, Vector3 worldCoords) {
	Vector4 ambientLight = worldAmbient;

	if (material1)
	{
		Vector4 color;
		if (isTextured)
		    color = material1->getTextureColor(getU(worldCoords), getV(worldCoords));
		else
			color = material1->getAmbientColor();
		ambientLight = ambientLight * color * material1->ambientStrength;
	}

	return ambientLight;
}

/// <summary>
/// Returns the diffuse color of the object at the specified world coordinates for a given light source.
/// </summary>
/// <param name="worldCoords">The world coordinates.</param>
/// <param name="normal">Normal of the intersection.</param>
/// <param name="l">Light source.</param>
/// <param name="lightVector">Vector to light source.</param>
/// <returns>The diffuse color of the object at the specified world coordinates for a given light source.</returns>
Vector4 RTObject::calculateDiffuse(Vector3 worldCoords, Vector3 normal, Light l, Vector3 lightVector)
{
	Vector4 diffuseLight = l.LightColor;

	if (material1)
	{
		Vector4 color;
		if (isTextured)
		    color = material1->getTextureColor(getU(worldCoords), getV(worldCoords));
		else
			color = material1->getDiffuseColor();
		diffuseLight = diffuseLight * color * fabs(lightVector.Dot(normal)) * material1->diffuseStrength;
	}

	return diffuseLight;
}

/// <summary>
/// Returns the specular color of the object at the specified world coordinates for a given light source.
/// </summary>
/// <param name="intersection">Point to find diffuse color.</param>
/// <param name="normal">Normal of the intersection.</param>
/// <param name="l">Light source.</param>
/// <param name="lightVector">Vector to light source.</param>
/// <param name="viewVector">Vector of the camera view.</param>
/// <returns>The specular color of the object at the specified point for a given light source.</returns>
Vector4 RTObject::calculateSpecular(Vector3 worldCoords, Vector3 normal, Light l, Vector3 lightVector, Vector3 viewVector)
{
	Vector3 reflectedVector = lightVector.Reflect(normal);
	double dot = reflectedVector.Dot(viewVector);

	if (dot >= 0)
	    return Vector4(0, 0, 0, 0);

	Vector4 specularLight = l.LightColor;

	if (material1)
	{
		Vector4 color;
		if (isTextured)
		    color = material1->getTextureColor(getU(worldCoords), getV(worldCoords));
		else
			color = material1->getSpecularColor();
		specularLight = specularLight * color * material1->specularStrength *
				fabs(lightVector.Dot(normal) * pow(dot, material1->exponent));
	}

	return specularLight;
}

}

