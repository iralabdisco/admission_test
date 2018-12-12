// import libraries
#include <fstream>
#include <math.h>
#include <iostream>
#include <vector> 
#include <sstream>


using namespace std;

// main code here
int main(){

    // Configuraitons
    string configCamera = "./Camera.config";
    string configPositions = "./Positions.config";

    // read the parameters of the camera
    ifstream inConfigCamera(configCamera.c_str(), ifstream::in); // handle
    string line; // temporary line 
    // focal length
    double f;
    // z
    double z;
    while(getline(inConfigCamera,line)){ // until the last line do your best :D 
        istringstream iss(line);
        iss >> f; 
        iss >> z; 
    }

    // read the list of motion actions
    vector<double> motionActionsD;
    vector<double> motionActionsTheta;
    ifstream inConfigPositions(configPositions.c_str(), ifstream::in); // handle
    while(getline(inConfigPositions,line)){ // until the last line do your best :D 
        istringstream iss(line);
        // translation
        double d;
        // heading
        double theta;
        iss >> d; 
        iss >> theta; 
        motionActionsD.push_back(d);
        motionActionsTheta.push_back(theta);
    }

    // initial pose by command line 
    double xInit;
    double yInit;
    double thetaInit;
    double xFinal;
    double yFinal;
    double thetaFinal;
    cout << "Please enter the initial position x: ";
    cin >> xInit; 
    cout << "Please enter the initial position y: ";
    cin >> yInit; 
    cout << "Please enter the initial rotation theta: ";
    cin >> thetaInit; 

    // homogeneous transformation matrix 
    // T = (cosTheta, -sinTheta, xt)
    //     (sinTheta,  cosTheta, yt)
    //     (0,            0,      1)
    // p' = T*p roto-translation
    // #TODO: vectorized form is easier, change it to matrix later
    // #TODO: Check the perforrrrrrrrrrrrrrrrrrrrrrrmance later
    int allMotions = motionActionsD.size();
    double x = xInit;
    double y = yInit;
    for (int i = 0; i < allMotions ; ++i){
        double d = motionActionsD[i];
        double theta = motionActionsTheta[i];
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);
        xFinal =  x*cosTheta - y*sinTheta + d; 
        yFinal =  x*sinTheta + y*cosTheta + d; 
        x = xFinal;
        y = yFinal;
    }

    // calculate the initial and final position projection 
    // (x,y,f) = f/z (x,y,z) image projection, the pinhole camera
    double m = f/z; // camera parameter
    double xProjectedI = m * xInit;
    double xProjectedF = m * xFinal;
    double yProjectedI = m * yInit;
    double yProjectedF = m * yFinal;
    double thetaProjectedI = m * thetaInit;
    double thetaProjectedF = m * thetaFinal;

    // print the results
    cout << "Projected Initial Position: " << xProjectedI << ", "<< yProjectedI << ", " << thetaProjectedI << endl; 
    cout << "Projected Final Position: " << xProjectedF << ", "<< yProjectedF << ", " << thetaProjectedF << endl; 
    
    return 0;
}