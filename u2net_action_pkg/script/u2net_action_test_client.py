#!/usr/bin/env python3

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import cv2
import os

import rospy
import rospkg
import actionlib
from cv_bridge import CvBridge

import rospkg
rospack = rospkg.RosPack()
sys.path.append(os.path.join(rospack.get_path("u2net_action_pkg"), "script/imports"))
from u2net_action_msgs.msg import U2NETGoal, U2NETAction

TEST_IMAGE_PATH = os.path.join(rospack.get_path("u2net_action_pkg"), "u2net/test_data/test_images" )

if __name__ == "__main__":
    rospy.init_node("u2net_action_test_client")
    bridge  = CvBridge()
    goal = U2NETGoal()

    cv_test_img  = cv2.imread(os.path.join(TEST_IMAGE_PATH, "0003.jpg"))
    goal.bgr_img = bridge.cv2_to_imgmsg(cv_test_img, "bgr8")

    client = actionlib.SimpleActionClient("u2net_action_server", U2NETAction)

    rospy.loginfo("Wait for server")
    client.wait_for_server()
    rospy.loginfo("Detect action-server")

    rospy.loginfo("Send the goal")
    client.send_goal(goal)
    rospy.loginfo("Wait for result")
    client.wait_for_result()

    cv_mask = bridge.imgmsg_to_cv2(client.get_result().mask_img, "bgr8")

    cv2.imshow("orig", cv_test_img)
    cv2.imshow("mask", cv_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rospy.loginfo("SUCCESS")

