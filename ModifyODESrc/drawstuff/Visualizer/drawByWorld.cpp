//draw stuff on second thread
#include <stdio.h>
#include <math.h>
#include <drawstuff/drawstuffWrapper.h>
#include <ode/ode.h>
#include <thread>
#include <mutex>
#include <chrono>

#define s_RADIUS 0.1f
#define BOXLEN 0.15f
#define BUFFER_LEN 100

std::mutex mtx;

static dSpaceID dsSpace;
static dBodyID sphbody;
static dBodyID boxbody;
static dGeomID sphgeom;
static dGeomID boxgeom;
static dGeomID ground;
static dJointGroupID contactgroup;
/*
void start()
{
	// adjust the starting viewpoint a bit
    static float xyz[3] = {2.6117f,-1.4433f,2.3700f};
    static float hpr[3] = {151.5000f,-30.5000f,0.0000f};
	dsSetViewpoint(xyz, hpr);
}


void simLoop(int pause)
{
	static float pos[3];
	static float rot[12];
	dGeomID g;
	dBodyID b = world->firstbody;
	dReal l, r;
	int type;
	for(int body_cnt=0; body_cnt<world->nb; body_cnt++){
		g = b->geom;
		b = dWorldGetNextBody(b);
		while(g != NULL){
			for (int i = 0; i < 3; i++)
				pos[i] = dGeomGetPosition(g)[i];
			for (int j = 0; j < 12; j++)
				rot[j] = dGeomGetRotation(g)[j];
			dsSetColor(1, 1, 0);
			type = dGeomGetClass(g);
			switch (type){
				case dSphereClass:
					dsDrawSphere(pos, rot, dGeomSphereGetRadius(g));
					break;
				case dBoxClass:
					float sides[3];
					dVector3 box_size;
					dGeomBoxGetLengths(g, box_size);
					sides[0] = box_size[0];
					sides[1] = box_size[1];
					sides[2] = box_size[2];
					dsDrawBox(pos, rot, sides);
					break;
				case dCapsuleClass:
					dGeomCapsuleGetParams(g, &r, &l);
					dsDrawCapsule(pos, rot, l, r);
					break;
				case dCylinderClass:
					dGeomCylinderGetParams(g,&r, &l);
					dsDrawCylinder(pos, rot, l, r);
					break;
			}
			g = dGeomGetBodyNext(g);
		}
	}
}

void command(int cmd)
{
	dsPrint("received command %d (`%c')\n", cmd, cmd);
}

void drawThread(int argc, char* argv[]) {
	// setup pointers to callback functions
	dsFunctions fn;
	fn.version = DS_VERSION;
	fn.start = &start;
	fn.step = &simLoop;
	fn.command = command;
	fn.stop = 0;
	fn.path_to_textures = "D:\\ode-scene\\ModifyODE\\drawstuff\\textures";	// uses default

	// run simulation
	dsSimulationLoop(argc, argv, 1280, 700, &fn);
}
*/
static void nearCallback(void *data, dGeomID o1, dGeomID o2){
    dBodyID b1 = dGeomGetBody(o1);
    dBodyID b2 = dGeomGetBody(o2);
    
    const int MAX_CONTACTS = 4;
    dContact contact[MAX_CONTACTS];

    int num_contact = dCollide(o1, o2, MAX_CONTACTS, &contact[0].geom, sizeof(dContact));
    for (int i=0;i<num_contact; i++){
        contact[i].surface.mode = dContactBounce;
        contact[i].surface.mu = 5;
		dJointID c = dJointCreateContact(dsWorld, contactgroup, contact+i);
		dJointAttach(c, b1, b2);
    }
}
void resetPos(){
	dBodySetPosition(sphbody, 0.0f, 15.0f * s_RADIUS, 0.0f);
	dBodySetPosition(boxbody, 0.0f, 6.0f * BOXLEN, 0.0f);
}
void simuThread(){
	float timestep = 0.01f;
	int i=0;
	static float simuTime = 0.0f;
	while(true){
		mtx.lock();
        dSpaceCollide(dsSpace, 0, &nearCallback);
        dWorldStep(dsWorld, timestep);
		dJointGroupEmpty(contactgroup);
		simuTime += timestep;
		if (simuTime > 2.0f){
			resetPos();
			simuTime = 0.0f;
		}
		mtx.unlock();
        dsSlowforRender();
	}
}
int main(int argc, char* argv[])
{
	dInitODE2(0);
	dsWorld = dWorldCreate();
	dWorldSetGravity(dsWorld, 0.0f, -9.81f, 0.0f);
	dsSpace = dSimpleSpaceCreate(0);
	contactgroup = dJointGroupCreate(0);

	sphbody = dBodyCreate(dsWorld);
	sphgeom = dCreateSphere(dsSpace, s_RADIUS);
    boxbody = dBodyCreate(dsWorld);
    boxgeom = dCreateBox(dsSpace, BOXLEN, BOXLEN, BOXLEN);
	dGeomSetBody(sphgeom, sphbody);
    dGeomSetBody(boxgeom, boxbody);
	resetPos();
	dMass m;
	dMassSetSphere(&m, 2.0f, s_RADIUS);
	dBodySetMass(sphbody, &m);
    dMass m_b;
	dMassSetBox(&m_b, 1.0f, BOXLEN, BOXLEN, BOXLEN);
	dBodySetMass(boxbody, &m_b);
	ground = dCreatePlane(dsSpace, 0, 1, 0, 0);
	//std::thread dsThread(dsDrawWorld, argc, argv);
	dsDrawWorldinThread();
	simuThread();
	//dsThread.join();
	printf("main...\n");

	dWorldDestroy(dsWorld);
	dBodyDestroy(sphbody);
	dGeomDestroy(sphgeom);
    dGeomDestroy(ground);
    dSpaceDestroy(dsSpace);
	dJointGroupDestroy(contactgroup);
    dCloseODE();
	return 0;
}