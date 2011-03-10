/*
 * Material.cpp
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#include "Material.h"

namespace RayTracer {

Vector4 Material::getAmbientColor() { return ambientColor; }
void Material::setAmbientColor(Vector4 color) { ambientColor = color; }
Vector4 Material::getDiffuseColor() { return diffuseColor; }
void Material::setDiffuseColor(Vector4 color) { diffuseColor = color; }
Vector4 Material::getSpecularColor() { return specularColor; }
void Material::setSpecularColor(Vector4 color) { specularColor = color; }

}
