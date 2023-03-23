#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#define FREEGLUT_STATIC
#include "gl_core_3_3.h"
#include <GL/glut.h>
#include <GL/freeglut_ext.h>

#define TW_STATIC
#include <AntTweakBar.h>


#include <ctime>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>  
#include<fstream>
#include<iostream>

#include "objloader.h"
#include "glprogram.h"
#include "MyImage.h"
#include "VAOImage.h"
#include "VAOMesh.h"
#include "trackball.h"

#include"Loop/MyLoop.h"
#include"Loop/MatInfo.h"
#include"Loop/MeshInfo.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "functions.h"

using namespace std;
using namespace chrono;

// mesh variables
MyLoop loop;
OBJ::Data obj;
std::string path = (".\\monsterfrogT.obj");
int target_level = 1;

//
GLProgram MyMesh::prog, MyMesh::pickProg, MyMesh::pointSetProg;
GLTexture MyMesh::colormapTex;

//
MyMesh M;
MyMesh subdivM;

// vbo variables
struct cudaGraphicsResource* cuda_vbo_resource;
struct cudaGraphicsResource* cuda_vbo_resource_2;

//
int viewport[4] = { 0, 0, 1280, 960 };
int actPrimType = MyMesh::PE_VERTEX;

bool showATB = true;

void deform_preprocess(){

}

void meshLoop2() {

    std::vector<int> P2PVtxIds = M.getConstrainVertexIds();
    std::vector<float> p2pDsts = M.getConstrainVertexCoords();


    if (P2PVtxIds.size() > 0) {
        loop.cmesh_refine(P2PVtxIds, p2pDsts);
    }
    loop.cmesh_subdivison();
    //////////////////verts///////////////////

    unsigned nv = loop.rmesh.nverts;
    unsigned nf = loop.rmesh.nfaces;

    subdivM.allocateStorage(loop.rmesh.nverts, loop.rmesh.nfaces);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, subdivM.gpuX.handle, cudaGraphicsMapFlagsWriteDiscard);

    float3* dptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_resource);

    launch_array << <nv / 256 + 1, 256 >> > (dptr, loop.rmesh.verts, nv);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    ////////////////faces////////////////////

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_2, subdivM.gpuT.handle, cudaGraphicsMapFlagsWriteDiscard);

    int3* dptr_2;
    size_t num_bytes_2;
    cudaGraphicsMapResources(1, &cuda_vbo_resource_2, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr_2, &num_bytes_2, cuda_vbo_resource_2);

    launch_faces_trans << <nf / 256 + 1, 256 >> > (dptr_2, loop.rmesh.ids, nf);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource_2, 0);
    cudaGraphicsUnregisterResource(cuda_vbo_resource_2);

    ////////////////free////////////////////

    float time = loop.get_time();
    printf("used time: %f (ms)\n\n", time);

    loop.rmesh.free();
}

int mousePressButton;
int mouseButtonDown;
int mousePos[2];

bool msaa = true;

float myRotation[4] = { 1, 0, 0, 0 };

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    if (msaa) glEnable(GL_MULTISAMPLE);
    else glDisable(GL_MULTISAMPLE);

    auto temp = viewport[2];
    glViewport(0, 0, temp >> 1, viewport[3]);

    viewport[2] /= 2;
    M.draw(viewport);

    glViewport(temp >> 1, 0, temp - (temp >> 1), viewport[3]);
    viewport[0] = viewport[2];
    subdivM.draw(viewport);


    viewport[0] = 0;
    viewport[2] = temp;

    if (showATB) TwDraw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char code, int x, int y)
{
    if (!TwEventKeyboardGLUT(code, x, y)) {
        switch (code) {
        case 17:
            exit(0);
        case 'f':
            glutFullScreenToggle();
            break;
        case ' ':
            showATB = !showATB;
            break;
        case 'l':
            meshLoop2();
        }
    }
    glutPostRedisplay();
}

void onMouseButton(int button, int updown, int x, int y)
{
    if (!showATB || !TwEventMouseButtonGLUT(button, updown, x, y)) {
        mousePressButton = button;
        mouseButtonDown = updown;

        if (updown == GLUT_DOWN) {
            if (button == GLUT_LEFT_BUTTON) {
                if (glutGetModifiers() & GLUT_ACTIVE_CTRL) {
                }
                else {
                    auto temp = viewport[2];
                    glViewport(0, 0, temp >> 1, viewport[3]);
                    viewport[2] /= 2;
                    int r = M.pick(x, y, viewport, M.PE_VERTEX, M.PO_ADD);
                    viewport[2] = temp;
                }
            }
            else if (button == GLUT_RIGHT_BUTTON) {
                auto temp = viewport[2];
                glViewport(0, 0, temp >> 1, viewport[3]);
                viewport[2] /= 2;
                M.pick(x, y, viewport, M.PE_VERTEX, M.PO_REMOVE);
                viewport[2] = temp;
            }
        }
        else { // updown == GLUT_UP
            if (button == GLUT_LEFT_BUTTON);
        }
        mousePos[0] = x;
        mousePos[1] = y;
    }
    glutPostRedisplay();
}

void onMouseMove(int x, int y)
{
    if (!showATB || !TwEventMouseMotionGLUT(x, y)) {
        if (mouseButtonDown == GLUT_DOWN) {
            if (mousePressButton == GLUT_MIDDLE_BUTTON) {
                M.moveInScreen(mousePos[0], mousePos[1], x, y, viewport);
                subdivM.moveInScreen(mousePos[0], mousePos[1], x, y, viewport);

            }
            else if (mousePressButton == GLUT_LEFT_BUTTON) {
                auto temp = viewport[2];
                glViewport(0, 0, temp >> 1, viewport[3]);
                viewport[2] /= 2;

                if (!M.moveCurrentVertex(x, y, viewport)) {
                    meshLoop2();
                }
                else {
                    const float s[2] = { 2.f / viewport[2], 2.f / viewport[3] };
                    auto r = Quat<float>(M.mQRotate) * Quat<float>::trackball(x * s[0] - 1, 1 - y * s[1], s[0] * mousePos[0] - 1, 1 - s[1] * mousePos[1]);
                    std::copy_n(r.q, 4, M.mQRotate);
                    std::copy_n(r.q, 4, subdivM.mQRotate);

                }
                viewport[2] = temp;
            }
        }
    }
    mousePos[0] = x; mousePos[1] = y;
    glutPostRedisplay();
}


void onMouseWheel(int wheel_number, int direction, int x, int y)
{
    M.mMeshScale *= direction > 0 ? 1.1f : 0.9f;
    subdivM.mMeshScale *= direction > 0 ? 1.1f : 0.9f;

    glutPostRedisplay();
}

int initGL(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
    glutInitWindowSize(960, 960);
    glutInitWindowPosition(200, 50);
    glutCreateWindow(argv[0]);

    // !Load the OpenGL functions. after the opengl context has been created
    if (ogl_LoadFunctions() == ogl_LOAD_FAILED)
        return -1;

    glClearColor(1.f, 1.f, 1.f, 0.f);

    glutReshapeFunc([](int w, int h) { viewport[2] = w; viewport[3] = h; TwWindowSize(w, h); });
    glutDisplayFunc(display);
    glutKeyboardFunc(onKeyboard);
    glutMouseFunc(onMouseButton);
    glutMotionFunc(onMouseMove);
    glutMouseWheelFunc(onMouseWheel);
    glutCloseFunc([]() {exit(0); });

    return 0;
}


void createTweakbar()
{
    TwBar* bar = TwGetBarByName("MeshViewer");
    if (bar)    TwDeleteBar(bar);

    //Create a tweak bar
    bar = TwNewBar("MeshViewer");
    TwDefine(" MeshViewer size='220 100' color='0 128 255' text=dark alpha=128 position='5 5'"); // change default tweak bar size and color

    TwAddVarRO(bar, "#Vertex", TW_TYPE_INT32, &M.nVertex, " group='Mesh View'");
    TwAddVarRO(bar, "#Face", TW_TYPE_INT32, &M.nFace, " group='Mesh View'");

    TwAddVarRW(bar, "Point Size", TW_TYPE_FLOAT, &M.pointSize, " group='Mesh View' ");
    TwAddVarRW(bar, "Vertex Color", TW_TYPE_COLOR4F, M.vertexColor.data(), " group='Mesh View' help='mesh vertex color' ");


    TwAddVarRW(bar, "Edge Width", TW_TYPE_FLOAT, &M.edgeWidth, " group='Mesh View' ");
    TwAddVarRW(bar, "Edge Color", TW_TYPE_COLOR4F, M.edgeColor.data(), " group='Mesh View' help='mesh edge color' ");

    TwAddVarRW(bar, "Face Color", TW_TYPE_COLOR4F, M.faceColor.data(), " group='Mesh View' help='mesh face color' ");

    TwAddVarRW(bar, "subEdge Width", TW_TYPE_FLOAT, &subdivM.edgeWidth, " group='Mesh View' ");
    TwAddVarRW(bar, "subEdge Color", TW_TYPE_COLOR4F, subdivM.edgeColor.data(), " group='Mesh View' help='mesh edge color' ");

    TwAddVarRW(bar, "subFace Color", TW_TYPE_COLOR4F, subdivM.faceColor.data(), " group='Mesh View' help='mesh face color' ");

    TwDefine(" MeshViewer/'Mesh View' opened=false ");

    TwAddVarRW(bar, "Rotation", TW_TYPE_QUAT4F, myRotation, " group='Modeling' open help='can be used to specify the rotation for current handle' ");
}

int main(int argc, char* argv[])
{

    if (initGL(argc, argv)) {
        fprintf(stderr, "!Failed to initialize OpenGL!Exit...");
        exit(-1);
    }
    MyMesh::buildShaders();

    //////////////////////////////////////////////////////////////////////////
    loop.set_cmesh(path);
    loop.set_level(target_level);

    /// /////////////////////////////////////////////////////////////////////

    OBJ::read(obj, path.c_str());

    M.upload(obj.f_v.data(), obj.v.size(), obj.f_vi.data(), obj.f_vi.size() / 3, nullptr);
    subdivM.upload(obj.f_v.data(), obj.v.size(), obj.f_vi.data(), obj.f_vi.size() / 3, nullptr);

    //////////////////////////////////////////////////////////////////////////

    TwInit(TW_OPENGL_CORE, NULL);
    TwGLUTModifiersFunc(glutGetModifiers);
    glutSpecialFunc([](int key, int x, int y) { TwEventSpecialGLUT(key, x, y); glutPostRedisplay(); }); // important for special keys like UP/DOWN/LEFT/RIGHT ...
    TwCopyStdStringToClientFunc([](std::string& dst, const std::string& src) {dst = src; });

    createTweakbar();

    ////////////////////////////////////////////////////////////////////////

    atexit([] { TwDeleteAllBars();  TwTerminate(); });

    glutTimerFunc(1, [](int) {deform_preprocess(); }, 0);

    ////////////////////////////////////////////////////////////////////////////
    glutMainLoop();

    return 0;
}





