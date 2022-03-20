import rospy

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys


class image_converter:

    def __init__(self):
        """
        This is a node that subscribes to the RGB image topic of rosout, 
        convert each sensor_msg.msg.Image message at each timestamp to cv2 compatible numpy.ndarray
        """
        self.image_pub = rospy.Publisher("image_topic_2",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/uav/camera/left_rgb_blurred/image_rect_color',Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows,cols,channels) = cv_image.shape

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    #Step 1: Initialize Python3 Object
    ic = image_converter()
    #Step 2: Initialize ROS node
    rospy.init_node('image_converter', anonymous=True)
    #Step 3: Run
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    #Step 4: Finish
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
