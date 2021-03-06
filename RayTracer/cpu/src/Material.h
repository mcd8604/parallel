﻿/*
 * Material.h
 *
 * Defines properties of a material. Implements the basic Phong illumination model.
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "Geom.h"

namespace RayTracer {

class Material {

private:
	Vector4 ambientColor;
	Vector4 diffuseColor;
	Vector4 specularColor;

public:
	double kR; // Reflectivity coeffecient.
	double kT; // Transparency coeffecient.
	double n; // Index of refraction
	double ambientStrength;
	double diffuseStrength;
	double specularStrength;
	double exponent;

	virtual Vector4 getAmbientColor();
	virtual void setAmbientColor(Vector4 color);
	virtual Vector4 getDiffuseColor();
	virtual void setDiffuseColor(Vector4 color);
	virtual Vector4 getSpecularColor();
	virtual void setSpecularColor(Vector4 color);
	//Vector4 (*getTextureColor)(float, float);
	//NOTE: hardcoded texture function, should be a function pointer
	virtual Vector4 getTextureColor(float u, float v);
};

}

#endif /* MATERIAL_H_ */
