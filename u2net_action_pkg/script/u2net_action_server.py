#!/usr/bin/env python3

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import cv2
import os
import math
import pickle as pkl

import torch
import torchvision
import PIL

import rospy
import rospkg
import actionlib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import getCvType

import rospkg
rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path("u2net_action_pkg"), "script/imports"))
from u2net import U2NET
from u2net_action_msgs.msg import U2NETAction, U2NETResult, U2NETFeedback 

class U2NETServer(object):
    def __init__(self)-> None:
        rospy.loginfo("Start Initialize")
        self.bridge  = CvBridge()
        self.model = U2NET()
        self.action_server = actionlib.SimpleActionServer("u2net_action_server", 
                                                            U2NETAction,
                                                            execute_cb=self.callback,
                                                            auto_start=False)
        self.action_server.start()

    def callback(self, goal):
        rospy.loginfo("Get image")
        feedback = U2NETFeedback()
        result   = U2NETResult()
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(goal.bgr_img , "bgr8")
        except Exception as err:
            rospy.loginfo(err)

        try:
            cv_mask = self.model(cv_bgr)
            feedback.feedback = True
            self.action_server.publish_feedback(feedback)
        except:
            feedback.feedback = False
            self.action_server.publish_feedback(feedback)
            return

        result.mask_img = self.bridge.cv2_to_imgmsg(cv_mask, "bgr8")
        self.action_server.set_succeeded(result)

if __name__ == "__main__":
    rospy.init_node("u2net_action_node")
    action_server = U2NETServer()
    rospy.spin()

