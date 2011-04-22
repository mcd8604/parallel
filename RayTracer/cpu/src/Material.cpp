/*
 * Material.cpp
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#include "Material.h"
#include <math.h>

namespace RayTracer {

Vector4 Material::getAmbientColor() { return ambientColor; }
void Material::setAmbientColor(Vector4 color) { ambientColor = color; }
Vector4 Material::getDiffuseColor() { return diffuseColor; }
void Material::setDiffuseColor(Vector4 color) { diffuseColor = color; }
Vector4 Material::getSpecularColor() { return specularColor; }
void Material::setSpecularColor(Vector4 color) { specularColor = color; }
//NOTE: hardcoded checkered texture
Vector4 Material::getTextureColor(float u, float v) {
	//NOTE: hardcoded static properties for hardcoded function
	Vector4 red = Vector4(1, 0, 0, 1);
	Vector4 yellow = Vector4(1, 1, 0, 1);
    if (fmod(u, 1) < 0.5f)
    {
        if (fmod(v, 1) < 0.5f)
            return red;// * ambientStrength;
        else
            return yellow;// * ambientStrength;
    }
    else
    {
        if (fmod(v, 1) < 0.5f)
            return yellow;// * ambientStrength;
        else
        	return red;// * ambientStrength;
    }
}

}
