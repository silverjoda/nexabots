#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np

#hexapod robot 
import src.envs.hexapod_trossen.hexapod_sim.RobotHAL as robothal
#import hexapod_real.RobotHAL as robothal

from src.envs.hexapod_trossen.RobotConst import *


class Robot:
    def __init__(self, logfile=None):
        """
        Robot class constructor

        Parameters
        ----------
        logfile: string (optional)
            name of the log file to store all the simulation data
        """
        self.robot = robothal.RobotHAL(TIME_FRAME)
        
       
        #mined data
        self.position_ = None
        self.orientation_ = None
        self.acceleration_ = None
        self.joint_position_set_ = None
        self.joint_position_real_ = None
        self.joint_torques_ = None
        self.leg_contacts_ = None

        #logfile
        self.logfile = None
        if not logfile == None:
            self.start_log(logfile)
    
    
    def start_log(self, logfile):
        """
        method to open the logfile and write data header into it
        """
        self.logfile = open(logfile,'w')
        if self.logfile == -1:
            self.logfile = None
            print("ERROR opening the log file")
        else:
            #cat the header into the file
            self.logfile.write("#timestamp, " 
                "position_x, position_y, position_z, "
                "orientation_yaw, orientation_pitch, orientation_roll, "
                "acceleration_x, acceleration_y, acceleration_z, "
                "joint_position_set x18, "
                "joint_position_real x18, "
                "joint_torques x18, "
                "leg contacts x6\n")
    

    def turn_on(self):
        """
        Method to drive the robot into the default position
        """
        #read out the current pose of the robot
        pose = self.robot.get_all_servo_position()

        #interpolate to the default position
        INTERPOLATION_TIME = 3000 #ms
        interpolation_steps = int(INTERPOLATION_TIME/TIME_FRAME)

        speed = np.zeros(18)
        for i in range(0,18):
            speed[i] = (SERVOS_BASE[i]-pose[i])/interpolation_steps
        
        #execute the motion
        for t in range(0, interpolation_steps):
            self.robot.set_all_servo_position(pose + t*speed)
            pass


    def turn_off(self):
        self.robot.stop_simulation()

    
if __name__=="__main__":
    robot = Robot()
    robot.turn_on()
    time.sleep(3)

    for i in range(100):
        vec = [0] * 18
        robot.robot.set_all_servo_position(vec)

        time.sleep(2)
        robot.robot.set_all_servo_position([1] * 18)
        time.sleep(2)
        


