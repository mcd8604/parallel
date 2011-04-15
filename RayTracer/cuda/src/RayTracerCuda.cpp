/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.

    Note - this is intended to be an example of using 3D textures
    in CUDA, not an optimized volume renderer.

    Changes
    sgg 22/3/2010
    - updated to use texture for display instead of glDrawPixels.
    - changed to render from front-to-back rather than back-to-front.
*/

// Graphics includes
#include <GL/glew.h>
#include <GL/glut.h>

// Utilities and System includes
#include <shrUtils.h>
#include <cutil_math.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>
#include <cuda_gl_interop.h>

#include "RayTracer.h"

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "volume.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_volume.ppm",
    NULL
};

const char *sSDKsample = "CUDA 3D Volume Render";

char *volumeFilename = "Bucky.raw";
cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
typedef unsigned char VolumeType;

//char *volumeFilename = "mrt16_angio.raw";
//cudaExtent volumeSize = make_cudaExtent(416, 512, 112);
//typedef unsigned short VolumeType;

uint width = 800, height = 600;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];
Ray *rayTable;

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

bool unprojectGPU = true;
float3 camPos;
float3 camTar;
float3 camUp;
float fovy;
float near;
float far;

int numTriangles;
Triangle *triangles;
int numSpheres;
Sphere *spheres;
int numLights;
Light *lights;
float4 backgroundColor;
float4 ambientLight;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

unsigned int timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bQAGLVerify = false;
bool g_bFBODisplay = false;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)

extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output,
		//float3 camPos, float3 camTar, float3 camUp, float fovy, float near, float far,
		uint width, uint height, float4 ambientLight, float4 backgroundColor,
		uint numLights, Light *lights,
		uint numTriangles, Triangle *triangles,
		uint numSpheres, Sphere *spheres);
extern "C" void render_kernel2(dim3 gridSize, dim3 blockSize, uint *d_output,
		Ray *rayTable,
		uint width, uint height, float4 ambientLight, float4 backgroundColor,
		uint numLights, Light *lights,
		uint numTriangles, Triangle *triangles,
		uint numSpheres, Sphere *spheres);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void initPixelBuffer();
void updateView();

//void SetSceneData(uint width, uint height, float4 backgroundColor, float4 ambientLight,
		//unsigned int numLights, Light *lights, unsigned int numTriangles, Triangle *triangles, unsigned int numSpheres, Sphere *spheres);
//void FreeSceneData();

void GetSceneData()
{
	// TODO: read initialization data from file, data source, or user input

	camPos = make_float3(3, 4, 15);
	viewTranslation = make_float3(-3, -4, -15);
	camTar = make_float3(3, 0, -70);
	camUp = make_float3(0, 1, 0);
	fovy = 45.0;
	near = 0.1;
	far = 1000;

	backgroundColor = make_float4(.5, .7, .9, 1);
	ambientLight = make_float4(.6, .6, .6, 1);

	numLights = 2;
	lights = (Light *)malloc(numLights * sizeof(Light));
	lights[0].Position = make_float3(5, 8, 15);
	lights[0].Color = make_float4(1, 1, 1, 1);
	lights[1].Position = make_float3(-5, 8, 15);
	lights[1].Color = make_float4(1, 1, 1, 1);

	Triangle floor1;
	floor1.v1 = make_float3(8, 0, 16);
	floor1.v2 = make_float3(-8, 0, -16);
	floor1.v3 = make_float3(8, 0, -16);
	floor1.n = make_float3(0, 1, 0);

	Triangle floor2;
	floor2.v1 = make_float3(8, 0, 16);
	floor2.v2 = make_float3(-8, 0, -16);
	floor2.v3 = make_float3(-8, 0, 16);
	floor2.n = make_float3(0, 1, 0);

	Material floorM;
	floorM.ambientStrength = 1;
	floorM.diffuseStrength = 1;
    floorM.ambientColor = make_float4(0.2, 1, 0.2, 1);
    floorM.diffuseColor = make_float4(0.2, 1, 0.2, 1);
    floorM.specularColor = make_float4(0.2, 1, 0.2, 1);
	floor1.m = floorM;
	floor2.m = floorM;

	numTriangles = 2;
	triangles = (Triangle *)malloc(numTriangles * sizeof(Triangle));
	triangles[0] = floor1;
	triangles[1] = floor2;

    Sphere sphere1;
	sphere1.p = make_float3(3, 4, 11);
	sphere1.r = 1;
	// glass material
    Material glass;
    glass.ambientStrength = 0.075;
    glass.diffuseStrength = 0.075;
    glass.specularStrength = 0.2;
    glass.exponent = 20;
    glass.ambientColor = make_float4(1, 1, 1, 1);
    glass.diffuseColor = make_float4(1, 1, 1, 1);
    glass.specularColor = make_float4(1, 1, 1, 1);
    glass.kR = .01;
    glass.kT = .99;
    glass.n = .99;
    sphere1.m = glass;

    Sphere sphere2;
	sphere2.p = make_float3(1.5, 3, 9);
	sphere2.r = 1;
	// mirror material
    Material mirror;
    mirror.ambientStrength = 0.15;
    mirror.diffuseStrength = 0.25;
    mirror.specularStrength = 1;
    mirror.exponent = 20;
    mirror.ambientColor = make_float4(.7, .7, .7, .7);
    mirror.diffuseColor = make_float4(.7, .7, .7, .7);
    mirror.specularColor = make_float4(1, 1, 1, 1);
    mirror.kR = .75;
    sphere2.m = mirror;

	numSpheres = 2;
	spheres = (Sphere *)malloc(numSpheres * sizeof(Sphere));
	spheres[0] = sphere1;
	spheres[1] = sphere2;
}

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "AutoTest: Volume Render");
	    glutSetWindowTitle(temp);

		exit(0);
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "%sVolume Render: %3.1f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        cutilCheckError(cutResetTimer(timer));  

        AutoQATest();
    }
}

// render image using CUDA
void render()
{

    // map PBO to get CUDA device pointer
    uint *d_output;
	// map PBO to get CUDA device pointer
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes; 
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  
						       cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    cutilSafeCall(cudaMemset(d_output, 0, width*height*4));


    // call CUDA kernel, writing results to PBO
    if(unprojectGPU)
		render_kernel(gridSize, blockSize, d_output,
				//camPos, camTar, camUp, fovy, near, far,
				width, height,
				ambientLight, backgroundColor,
				numLights, lights,
				numTriangles, triangles,
				numSpheres, spheres);
    else
    {
        Ray *d_rayTable;
        cutilSafeCall(cudaMalloc((void **)&d_rayTable, width * height * sizeof(Ray)));
        cutilSafeCall(cudaMemcpy(d_rayTable, rayTable, width * height * sizeof(Ray), cudaMemcpyHostToDevice));
		render_kernel2(gridSize, blockSize, d_output, d_rayTable,
				width, height,
				ambientLight, backgroundColor,
				numLights, lights,
				numTriangles, triangles,
				numSpheres, spheres);
	    cutilSafeCall(cudaFree(d_rayTable));
    }

    cutilCheckMsg("kernel failed");


    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    cutilCheckError(cutStartTimer(timer));  

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
    // draw using glDrawPixels (slower)
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // draw using texture

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
#endif

    if (g_CheckRender && g_CheckRender->IsQAReadback() && g_Verify) {
        // readback for QA testing
        shrLog("\n> (Frame %d) Readback BackBuffer\n", frameCount);
        g_CheckRender->readback( width, height );
        g_CheckRender->savePPM(sOriginal[g_Index], true, NULL);
        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Index], sReference[g_Index], MAX_EPSILON_ERROR, THRESHOLD)) {
            g_TotalErrors++;
        }
        g_Verify = false;
    }
    glutSwapBuffers();
    glutReportErrors();

    cutilCheckError(cutStopTimer(timer));  

    computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            exit(0);
            break;
        case 'w':
        	camPos.z -= 1;
            break;
        case 's':
        	camPos.z += 1;
            break;
        case 'a':
        	camPos.x -= 0.1;
            break;
        case 'd':
        	camPos.x += 0.1;
            break;
        case 'q':
        	unprojectGPU = !unprojectGPU;
        	break;
        default:
            break;
    }
    //shrLog("spheres[0].p.z = %.2f, \n", spheres[0].p.z);
    updateView();
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 4) {
        // right = zoom
        //viewTranslation.z += dy / 100.0;
    	//camPos.z += dy / 100.0;
    } 
    else if (buttonState == 2) {
        // middle = translate
        //viewTranslation.x += dx / 100.0;
        //viewTranslation.y -= dy / 100.0;
    	//camPos.x += dx / 100.0;
    	//camPos.y -= dy / 100.0;
    }
    else if (buttonState == 1) {
        // left = rotate
        //viewRotation.x += dy / 5.0;
        //viewRotation.y += dx / 5.0;
        //camTar.x += dy / 5.0;
        //camTar.y += dx / 5.0;
    }

    ox = x; oy = y;
    updateView();
    glutPostRedisplay();
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
    width = w; height = h;
    initPixelBuffer();

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
    cutilCheckError( cutDeleteTimer( timer));

    //FreeSceneData();

    if (pbo) {
	    cutilSafeCall(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    if (g_CheckRender) {
        delete g_CheckRender; g_CheckRender = NULL;
    }
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        shrLog("Required OpenGL extensions missing.");
        exit(-1);
    }
}

void initPixelBuffer()
{
    if (pbo) {
		// unregister this buffer object from CUDA C
		cutilSafeCall(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));	

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

	void *data = malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    shrLog("Read '%s', %d bytes\n", filename, read);

    return data;
}

#define SWAP_ROWS_DOUBLE(a, b) { double *_tmp = a; (a)=(b); (b)=_tmp; }
#define SWAP_ROWS_FLOAT(a, b) { float *_tmp = a; (a)=(b); (b)=_tmp; }
#define MAT(m,r,c) (m)[(c)*4+(r)]
//This code comes directly from GLU except that it is for float
int glhInvertMatrixf2(float *m, float *out)
{
float wtmp[4][8];
float m0, m1, m2, m3, s;
float *r0, *r1, *r2, *r3;
r0 = wtmp[0], r1 = wtmp[1], r2 = wtmp[2], r3 = wtmp[3];
r0[0] = MAT(m, 0, 0), r0[1] = MAT(m, 0, 1),
  r0[2] = MAT(m, 0, 2), r0[3] = MAT(m, 0, 3),
  r0[4] = 1.0, r0[5] = r0[6] = r0[7] = 0.0,
  r1[0] = MAT(m, 1, 0), r1[1] = MAT(m, 1, 1),
  r1[2] = MAT(m, 1, 2), r1[3] = MAT(m, 1, 3),
  r1[5] = 1.0, r1[4] = r1[6] = r1[7] = 0.0,
  r2[0] = MAT(m, 2, 0), r2[1] = MAT(m, 2, 1),
  r2[2] = MAT(m, 2, 2), r2[3] = MAT(m, 2, 3),
  r2[6] = 1.0, r2[4] = r2[5] = r2[7] = 0.0,
  r3[0] = MAT(m, 3, 0), r3[1] = MAT(m, 3, 1),
  r3[2] = MAT(m, 3, 2), r3[3] = MAT(m, 3, 3),
  r3[7] = 1.0, r3[4] = r3[5] = r3[6] = 0.0;
/* choose pivot - or die */
if (fabsf(r3[0]) > fabsf(r2[0]))
  SWAP_ROWS_FLOAT(r3, r2);
if (fabsf(r2[0]) > fabsf(r1[0]))
  SWAP_ROWS_FLOAT(r2, r1);
if (fabsf(r1[0]) > fabsf(r0[0]))
  SWAP_ROWS_FLOAT(r1, r0);
if (0.0 == r0[0])
  return 0;
/* eliminate first variable     */
m1 = r1[0] / r0[0];
m2 = r2[0] / r0[0];
m3 = r3[0] / r0[0];
s = r0[1];
r1[1] -= m1 * s;
r2[1] -= m2 * s;
r3[1] -= m3 * s;
s = r0[2];
r1[2] -= m1 * s;
r2[2] -= m2 * s;
r3[2] -= m3 * s;
s = r0[3];
r1[3] -= m1 * s;
r2[3] -= m2 * s;
r3[3] -= m3 * s;
s = r0[4];
if (s != 0.0) {
  r1[4] -= m1 * s;
  r2[4] -= m2 * s;
  r3[4] -= m3 * s;
}
s = r0[5];
if (s != 0.0) {
  r1[5] -= m1 * s;
  r2[5] -= m2 * s;
  r3[5] -= m3 * s;
}
s = r0[6];
if (s != 0.0) {
  r1[6] -= m1 * s;
  r2[6] -= m2 * s;
  r3[6] -= m3 * s;
}
s = r0[7];
if (s != 0.0) {
  r1[7] -= m1 * s;
  r2[7] -= m2 * s;
  r3[7] -= m3 * s;
}
/* choose pivot - or die */
if (fabsf(r3[1]) > fabsf(r2[1]))
  SWAP_ROWS_FLOAT(r3, r2);
if (fabsf(r2[1]) > fabsf(r1[1]))
  SWAP_ROWS_FLOAT(r2, r1);
if (0.0 == r1[1])
  return 0;
/* eliminate second variable */
m2 = r2[1] / r1[1];
m3 = r3[1] / r1[1];
r2[2] -= m2 * r1[2];
r3[2] -= m3 * r1[2];
r2[3] -= m2 * r1[3];
r3[3] -= m3 * r1[3];
s = r1[4];
if (0.0 != s) {
  r2[4] -= m2 * s;
  r3[4] -= m3 * s;
}
s = r1[5];
if (0.0 != s) {
  r2[5] -= m2 * s;
  r3[5] -= m3 * s;
}
s = r1[6];
if (0.0 != s) {
  r2[6] -= m2 * s;
  r3[6] -= m3 * s;
}
s = r1[7];
if (0.0 != s) {
  r2[7] -= m2 * s;
  r3[7] -= m3 * s;
}
/* choose pivot - or die */
if (fabsf(r3[2]) > fabsf(r2[2]))
  SWAP_ROWS_FLOAT(r3, r2);
if (0.0 == r2[2])
  return 0;
/* eliminate third variable */
m3 = r3[2] / r2[2];
r3[3] -= m3 * r2[3], r3[4] -= m3 * r2[4],
  r3[5] -= m3 * r2[5], r3[6] -= m3 * r2[6], r3[7] -= m3 * r2[7];
/* last check */
if (0.0 == r3[3])
  return 0;
s = 1.0 / r3[3];		/* now back substitute row 3 */
r3[4] *= s;
r3[5] *= s;
r3[6] *= s;
r3[7] *= s;
m2 = r2[3];			/* now back substitute row 2 */
s = 1.0 / r2[2];
r2[4] = s * (r2[4] - r3[4] * m2), r2[5] = s * (r2[5] - r3[5] * m2),
  r2[6] = s * (r2[6] - r3[6] * m2), r2[7] = s * (r2[7] - r3[7] * m2);
m1 = r1[3];
r1[4] -= r3[4] * m1, r1[5] -= r3[5] * m1,
  r1[6] -= r3[6] * m1, r1[7] -= r3[7] * m1;
m0 = r0[3];
r0[4] -= r3[4] * m0, r0[5] -= r3[5] * m0,
  r0[6] -= r3[6] * m0, r0[7] -= r3[7] * m0;
m1 = r1[2];			/* now back substitute row 1 */
s = 1.0 / r1[1];
r1[4] = s * (r1[4] - r2[4] * m1), r1[5] = s * (r1[5] - r2[5] * m1),
  r1[6] = s * (r1[6] - r2[6] * m1), r1[7] = s * (r1[7] - r2[7] * m1);
m0 = r0[2];
r0[4] -= r2[4] * m0, r0[5] -= r2[5] * m0,
  r0[6] -= r2[6] * m0, r0[7] -= r2[7] * m0;
m0 = r0[1];			/* now back substitute row 0 */
s = 1.0 / r0[0];
r0[4] = s * (r0[4] - r1[4] * m0), r0[5] = s * (r0[5] - r1[5] * m0),
  r0[6] = s * (r0[6] - r1[6] * m0), r0[7] = s * (r0[7] - r1[7] * m0);
MAT(out, 0, 0) = r0[4];
MAT(out, 0, 1) = r0[5], MAT(out, 0, 2) = r0[6];
MAT(out, 0, 3) = r0[7], MAT(out, 1, 0) = r1[4];
MAT(out, 1, 1) = r1[5], MAT(out, 1, 2) = r1[6];
MAT(out, 1, 3) = r1[7], MAT(out, 2, 0) = r2[4];
MAT(out, 2, 1) = r2[5], MAT(out, 2, 2) = r2[6];
MAT(out, 2, 3) = r2[7], MAT(out, 3, 0) = r3[4];
MAT(out, 3, 1) = r3[5], MAT(out, 3, 2) = r3[6];
MAT(out, 3, 3) = r3[7];
return 1;
}

void unproject() {
	// use OpenGL to unproject

	if(!rayTable)
		rayTable = (Ray *)malloc(sizeof(Ray) * width * height);

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
	gluPerspective(fovy, GLdouble(width)/height, near, far);
	glGetDoublev(GL_PROJECTION_MATRIX, proj);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	gluLookAt(camPos.x, camPos.y, camPos.z,
			camTar.x, camTar.y, camTar.z,
			camUp.x, camUp.y, camUp.z);
	glGetDoublev(GL_MODELVIEW_MATRIX, model);
	//glGetIntegerv(GL_VIEWPORT, view);
	glPopMatrix();

	int x, y;
	for(y = 0; y < height; ++y)
	{
		for(x = 0; x < width; ++x)
		{
			GLdouble coords[3];
			gluUnProject(x, y, 0, model, proj, view, &coords[0], &coords[1], &coords[2]);
			float3 rayS = make_float3(coords[0],coords[1],coords[2]);
			gluUnProject(x, y, 1, model, proj, view, &coords[0], &coords[1], &coords[2]);
			float3 rayE = make_float3(coords[0],coords[1],coords[2]);

			Ray r;
			r.Position = rayS;
			r.Direction = normalize(rayE - rayS);
			rayTable[x + y * width] = r;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void updateView()
{
    GLfloat mvp[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(fovy, GLdouble(width)/height, near, far);
    gluLookAt(camPos.x, camPos.y, camPos.z,
    		camTar.x, camTar.y, camTar.z,
    		camUp.x, camUp.y, camUp.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, mvp);
    glPopMatrix();
    glLoadIdentity();

//	shrLog("%f %f %f %f\n", mvp[0], mvp[1], mvp[2], mvp[3]);
//	shrLog("%f %f %f %f\n", mvp[4], mvp[5], mvp[6], mvp[7]);
//	shrLog("%f %f %f %f\n", mvp[8], mvp[9], mvp[10], mvp[11]);
//	shrLog("%f %f %f %f\n\n", mvp[12], mvp[13], mvp[14], mvp[15]);

    if(unprojectGPU)
    {
		float inverse[16];
		glhInvertMatrixf2(mvp, inverse);
		float transpose[16];
		transpose[0] = inverse[0];
		transpose[1] = inverse[4];
		transpose[2] = inverse[8];
		transpose[3] = inverse[12];
		transpose[4] = inverse[1];
		transpose[5] = inverse[5];
		transpose[6] = inverse[9];
		transpose[7] = inverse[13];
		transpose[8] = inverse[2];
		transpose[9] = inverse[6];
		transpose[10] = inverse[10];
		transpose[11] = inverse[14];
		transpose[12] = inverse[3];
		transpose[13] = inverse[7];
		transpose[14] = inverse[11];
		transpose[15] = inverse[15];
		copyInvViewMatrix(transpose, sizeof (float4) * 4);
    } else {
    	unproject();
    }
}

int
main( int argc, char** argv) 
{
    //start logs
    shrSetLogFileName ("volumeRender.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest") ||
		cutCheckCmdLineFlag(argc, (const char **)argv, "noprompt")) 
	{
        g_bQAReadback = true;
        fpsLimit = frameCheckNumber;
    }

    if (cutCheckCmdLineFlag(argc, (const char **)argv, "glverify")) 
	{
        g_bQAGLVerify = true;
        fpsLimit = frameCheckNumber;
    }

    if (g_bQAReadback) {
	    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
            cutilDeviceInit(argc, argv);
        } else {
            cudaSetDevice( cutGetMaxGflopsDeviceId() );
        }

    } else {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL( &argc, argv );

	    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
            cutilGLDeviceInit(argc, argv);
        } else {
            cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
        }
/*
        int device;
        struct cudaDeviceProp prop;
        cudaGetDevice( &device );
        cudaGetDeviceProperties( &prop, device );
        if( !strncmp( "Tesla", prop.name, 5 ) ) {
            shrLog("This sample needs a card capable of OpenGL and display.\n");
            shrLog("Please choose a different device with the -device=x argument.\n");
            cutilExit(argc, argv);
        }
*/
	}

    // parse arguments
    char *filename;
    if (cutGetCmdLineArgumentstr( argc, (const char**) argv, "file", &filename)) {
        volumeFilename = filename;
    }
    int n;
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "size", &n)) {
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "xsize", &n)) {
        volumeSize.width = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "ysize", &n)) {
        volumeSize.height = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "zsize", &n)) {
         volumeSize.depth = n;
    }

    // load volume data
    char* path = shrFindFilePath(volumeFilename, argv[0]);
    if (path == 0) {
        shrLog("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
    void *h_volume = loadRawFile(path, size);
    
    free(h_volume);

    cutilCheckError( cutCreateTimer( &timer));

    shrLog("Press '=' and '-' to change density\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n\n");

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    if (g_bQAReadback) {
        g_CheckRender = new CheckBackBuffer(width, height, 4, false);
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);

        uint *d_output;
        cutilSafeCall(cudaMalloc((void**)&d_output, width*height*sizeof(uint)));
        cutilSafeCall(cudaMemset(d_output, 0, width*height*sizeof(uint)));

        float modelView[16] = 
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 1.0f
        };

        invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
        invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
        invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

        // call CUDA kernel, writing results to PBO
	    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
        
        // Start timer 0 and process n loops on the GPU 
        int nIter = 10;
        for (int i = -1; i < nIter; i++)
        {
            if( i == 0 ) {
                cudaThreadSynchronize();
                cutStartTimer(timer); 
            }
            
            //render_kernel(gridSize, blockSize, d_output);
        }
        cudaThreadSynchronize();
        cutStopTimer(timer);
        // Get elapsed time and throughput, then log to sample and master logs
        double dAvgTime = cutGetTimerValue(timer)/(nIter * 1000.0);
        shrLogEx(LOGBOTH | MASTER, 0, "volumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n", 
               (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y); 
        

        cutilCheckMsg("Error: render_kernel() execution FAILED");
        cutilSafeCall( cudaThreadSynchronize() );

        cutilSafeCall( cudaMemcpy(g_CheckRender->imageData(), d_output, width*height*4, cudaMemcpyDeviceToHost) );
        g_CheckRender->savePPM(sOriginal[g_Index], true, NULL);

        if (!g_CheckRender->PPMvsPPM(sOriginal[g_Index], sReference[g_Index], MAX_EPSILON_ERROR, THRESHOLD)) {
            shrLog("\nFAILED\n\n");
        } else {
            shrLog("\nPASSED\n\n");
        }

        cudaFree(d_output);

        if (g_CheckRender) {
            delete g_CheckRender; g_CheckRender = NULL;
        }

    } else {
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutReshapeFunc(reshape);
		glutIdleFunc(idle);
		GetSceneData();
		updateView();

        initPixelBuffer();

        if (g_bQAGLVerify) {
            g_CheckRender = new CheckBackBuffer(width, height, 4);
            g_CheckRender->setPixelFormat(GL_RGBA);
            g_CheckRender->setExecPath(argv[0]);
            g_CheckRender->EnableQAReadback(true);
        }
        atexit(cleanup);

        glutMainLoop();
    }

    cudaThreadExit();
    shrEXIT(argc, (const char**)argv);
}
