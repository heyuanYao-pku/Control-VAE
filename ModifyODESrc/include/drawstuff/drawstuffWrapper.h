#include <ode/ode.h>
#ifdef WIN32
#include "drawstuff/drawstuff.h"
#endif
// extern float color[3];
// extern float campos[3];
// extern float camrot[3];
// extern float JointRadius;

// extern dWorldID dsWorld;
// extern int pauseTime;
extern int DRAWBACKGROUND;

void dsWorldSetter(dWorldID value);

dWorldID dsWorldGetter();

void dsTrackBodyWrapper(dBodyID target, int track_character, int sync_y);

void dsAssignJointRadius(float x);

void dsAssignAxisLength(float x);

void dsCameraLookAtWrapper(float pos_x, float pos_y, float pos_z, float target_x, float target_y, float target_z, float up_x, float up_y, float up_z);

void dsAssignColor(float red, float green, float blue);

void dsAssignPauseTime(int value);

void dsAssignBackground(int x);

void dsWhetherHingeAxis(int x);

void dsWhetherLocalAxis(int x);

void drawStart();

void dsDrawStep(int pause);

void dsDrawWorld();

void dsDrawWorldinThread();

void dsKillThread();

void dsSlowforRender();

// Add by Zhenhua Song
void dsSetWindowWidth(int value);
void dsSetWindowHeight(int value);
int dsGetWindowWidth();
int dsGetWindowHeight();
void dsGetScreenBuffer(unsigned char * data);
void dsStartRecordVideo();
void dsPauseRecordVideo();
size_t dsGetVideoFrame();
void dsEndRecordVideo(unsigned char * output_data, size_t num_frame);
