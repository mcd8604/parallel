/*
 * RTObject.h
 *
 *  Created on: Feb 19, 2011
 *      Author: mike
 */

#ifndef RTOBJECT_H_
#define RTOBJECT_H_

#include "Material.h"

namespace RayTracer {

class RTObject {
protected:
	Material *material1;
	bool isTextured;
	float maxU;
	float maxV;
public:

	virtual bool operator ==(RTObject *o) =0;
	virtual bool operator !=(RTObject *o) =0;

    /// <summary>
    /// Set the Material of the object
    /// </summary>
	void SetMaterial(Material* mat);
	void SetMaterial(Material* mat, float maxU, float maxV);

	virtual float getU(Vector3 worldCoords);
	virtual float getV(Vector3 worldCoords);

    /// <summary>
    /// Set the Material of the object
    /// </summary>
	Material *GetMaterial();

    /// <summary>
    /// Tests a ray for intersection against the object.
    /// </summary>
    /// <param name="ray">The ray</param>
    /// <returns>The distance of the closest positive intersection, or null if no intersection exists.</returns>
    virtual double Intersects(Ray ray) =0;

    /// <summary>
    /// Gets the normal of the object at the specified point.
    /// </summary>
    /// <param name="intersectPoint">Point to find normal.</param>
    /// <returns>The normal of the object at the specified point.</returns>
    virtual Vector3 GetIntersectNormal(Vector3 intersectPoint) =0;

    /// <summary>
    /// Returns the ambient color of the object at the specified world coordinates.
    /// </summary>
    /// <param name="worldAmbient">World ambient color.</param>
    /// <param name="worldCoords">The world coordinates.</param>
    /// <returns>The ambient color of the object at the specified world coordinates.</returns>
	virtual Vector4 calculateAmbient(Vector4 worldAmbient, Vector3 worldCoords);

    /// <summary>
    /// Returns the diffuse color of the object at the specified world coordinates for a given light source.
    /// </summary>
    /// <param name="worldCoords">The world coordinates.</param>
    /// <param name="normal">Normal of the intersection.</param>
    /// <param name="l">Light source.</param>
    /// <param name="lightVector">Vector to light source.</param>
    /// <returns>The diffuse color of the object at the specified world coordinates for a given light source.</returns>
	virtual Vector4 calculateDiffuse(Vector3 worldCoords, Vector3 normal, Light l, Vector3 lightVector);

    /// <summary>
    /// Returns the specular color of the object at the specified world coordinates for a given light source.
    /// </summary>
    /// <param name="intersection">Point to find diffuse color.</param>
    /// <param name="normal">Normal of the intersection.</param>
    /// <param name="l">Light source.</param>
    /// <param name="lightVector">Vector to light source.</param>
    /// <param name="viewVector">Vector of the camera view.</param>
    /// <returns>The specular color of the object at the specified point for a given light source.</returns>
	Vector4 calculateSpecular(Vector3 worldCoords, Vector3 normal, Light l, Vector3 lightVector, Vector3 viewVector);

};

}

#endif /* RTOBJECT_H_ */
