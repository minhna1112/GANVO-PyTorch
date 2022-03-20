#!/home/minh/.conda/envs/torch-clone/bin python
import rospy
import rosbag

import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys

from tqdm import tqdm

import torch

def main(args):
    bag = rosbag.Bag('/home/minh/data.bag')
    # message_generator = bag.read_messages(topics=['/uav/camera/left_rgb_blurred/image_rect_color'])

    # bridge = CvBridge()

    # for i, (topic, msg, t) in enumerate(tqdm(message_generator)):
    #     im = bridge.imgmsg_to_cv2(msg, "bgr8")
    #     #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     #print(im.shape)
    #     #cv2.imwrite(f'/home/minh/data_0/{i}.jpg', im)
    #     cv2.imshow("Window", im)
    #     cv2.waitKey(3)

    #     nim = torch.rand(480, 640, 3).numpy() + im

    #     cv2.imshow("Next Window", nim)
    #     cv2.waitKey(3)
    #     #break

    odom_generator = bag.read_messages(topics=['/uav/odometry'])
    
    for i, (topic, msg, t) in enumerate(tqdm(odom_generator)):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        z = float(msg.pose.pose.position.z)

if __name__ == '__main__':
    main(sys.argv)
