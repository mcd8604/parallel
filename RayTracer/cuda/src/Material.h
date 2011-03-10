/*
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
	float kR; // Reflectivity coeffecient.
	float kT; // Transparency coeffecient.
	float n; // Index of refraction
	float ambientStrength;
	float diffuseStrength;
	float specularStrength;
	float exponent;

	virtual Vector4 getAmbientColor();
	virtual void setAmbientColor(Vector4 color);
	virtual Vector4 getDiffuseColor();
	virtual void setDiffuseColor(Vector4 color);
	virtual Vector4 getSpecularColor();
	virtual void setSpecularColor(Vector4 color);
};

}

#endif /* MATERIAL_H_ */
