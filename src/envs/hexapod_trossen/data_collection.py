#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIS DEMO SHOWS HOW TO SIMPLY COLLECT ALL THE AVAILABLE DATA WITH THE SIMULATED ROBOT
"""

import sys
sys.path.append("robot")
import Robot as rob

import math
import time
import numpy as np
import threading as thread

if __name__=="__main__":

    #parametrization of individual experiments
    #gait, swing legth, swing height
    experiments = [("tripod", 1, 1),
                   ("tripod", 0.5, 1), #short swing
                   ("tripod", 1.5, 1), #long swing
                   ("tetrapod", 1, 1),
                   ("tetrapod", 0.5, 1), #short swing
                   ("tetrapod", 1.5, 1), #long swing
                   ("pentapod", 1, 1),
                   ("pentapod", 0.5, 1), #short swing
                   ("pentapod", 1.5, 1)]

    simulation_time = 30 #[s] the length of a single experiment

    
    first_run = True
    for experiment in experiments:
        
        #wait for finishing previous run
        if not first_run:
            time.sleep(20)
        else:
            first_run = False

        #extract parametrization of individual experiments
        used_gait = experiment[0] #get the used gait
        swing_length = experiment[1] #get the swing length
        swing_height = experiment[2] #get the swing height

        #define the log file name for the data
        logfile = "logs/experiment_" + used_gait + "_" + str(swing_length) + "_" + str(swing_height) + ".log"
        print("Collecting data to " + logfile)
        
        #instantiate the robot - define the logfile for data
        robot = rob.Robot(logfile=logfile)

        #set the gait parametrization
        robot.set_locomotion_parameters(swing_length, swing_height, used_gait)

        #start the locomotion thread - the data are collected with each step of the simulation
        robot.start_locomotion()
        
        #locomote forward - diferential steering (v_left, v_right)
        v_left = 1
        v_right = 1
        robot.move(v_left, v_right)

        #after some time stop the simulation
        time.sleep(simulation_time)
        robot.stop_locomotion()
        
        #turn off the robot
        robot.turn_off()
