import rospy
import rosbag

import cv2
import numpy as np


from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
from tqdm import tqdm
from path import Path
import os

from test_interpolation import sync_pose


class BagConverter(object):
    def __init__(self, bag_path, dst_folder):
        """
        Args: bag_path: Path to bag file
              dst_folder: Folder path to holds converted results 
        """
        self.dst_folder = Path(dst_folder)
        self.bag = rosbag.Bag(bag_path)
        self.bridge = CvBridge()
        self.valid_tstamps = []

    def save_intrinsics(self):
        intrinsic_generator = self.bag.read_messages(topics='/uav/camera/left_rgb_blurred/camera_info')
        print("Reading Intrinsic Matrix......")
        for i, (topic, msg, t) in enumerate(intrinsic_generator):
            K_msg = msg.K # a tuple of (fx, 0, cx, 0, fy, cy, 0, 0, 1)
            K_matrix = np.array(K_msg).reshape([3,3]) # 3x3 intrinsic matrix
            np.savetxt(self.dst_folder / 'cam.txt', K_matrix)            
            break

    def save_rgb(self):
        self.bag.read_messages()
        image_generator = self.bag.read_messages(topics='/uav/camera/left_rgb_blurred/image_rect_color')
        print("Reading Images  ......")
        
  
        for i, (topic, msg, t) in enumerate(tqdm(image_generator)):
            im = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            im_path = self.dst_folder / (str(i).zfill(6) + '.jpg')
            self.valid_tstamps.append(t.to_sec())
            # print(type(t.to_nsec()))
            
            # cv2.imwrite(im_path, im)

        
                

    def quat2mat(self, w, x, y, z):
        """
        Args:  4 quarternion coefficients
        Return: Corresponing 3x3 Rotation matrix 
        """
        ww, wx, wy, wz = w*w, w*x, w*y, w*z
        xx, xy, xz = x*x, x*y, x*z
        yy, yz = y*y, y*z
        zz = z*z

        n = ww + xx + yy + zz

        s = 0 if n < 1e-8 else 2 / n
        
        R = np.array([1 - s*(yy+zz),  s*(xy-wz)   ,  s*(xz+wy), 
                      s*(xy+wz)    ,  1 - s*(xx+zz), s*(yz-wx),
                      s*(xz-wy),      s*(yz+wx),     1-s*(xx+yy)]).reshape([3,3])

        return R

    def save_pose(self):
        pose_generator = self.bag.read_messages('/uav/odometry')
        print("Reading Poses  ......")
        out = []
        odom_ts = []
        with open('./Timestamps_pose.txt', 'w') as f:
            for i, (topic, msg, t) in enumerate(tqdm(pose_generator)):
                px = float(msg.pose.pose.position.x) # float
                py = float(msg.pose.pose.position.y)
                pz = float(msg.pose.pose.position.z)

                ox = float(msg.pose.pose.orientation.x) # float
                oy = float(msg.pose.pose.orientation.y)
                oz = float(msg.pose.pose.orientation.z)
                ow = float(msg.pose.pose.orientation.w)

                t_vec = np.array([px,py,pz]).reshape([3,1]) # 3x1
                rot_mat = self.quat2mat(ow, ox, oy, oz) # 3x3
                T_mat = np.concatenate([rot_mat, t_vec], axis=1) # 3x4
                out.append(T_mat.reshape([-1,12]))
                odom_ts.append(t.to_sec())
                f.write(f"Timestaps pose {t}\n")
                
        out = np.concatenate(out, axis=0)
        odom_ts = np.array(odom_ts)
        print("syncing poses")
        final_poses = sync_pose(valid_timestamps=np.array(self.valid_tstamps), odom_timestamps=odom_ts, odom=out)
        
        np.savetxt(self.dst_folder / 'poses_new.txt', final_poses)


class BagDataReader(object):
    def __init__(self, raw_folder: str, 
                       tgt_folder: str,
                       get_depth: False,
                       get_pose: True
                       ):
        """
        Args: raw_folder: Path to folder containing bag files for each sequences
              tgt_folder: Path to destination folder, which hosts results subfolders arranged as
                    data_0
                           |0000000.jpg
                           |0000000.npy (if get_depth is True)
                           |0000001.jpg
                           |0000001.npy
                           |...
                           |cam.txt
                           |poses.txt (if get_pose is True)
                    data_1
                    data_2
                    .....
        """
        self.raw_folder = Path(raw_folder)
        self.tgt_folder = Path(tgt_folder)
        self.scene_names = [n[:-4] for n in os.listdir(raw_folder)] 
        self.get_depth = get_depth
        self.get_pose = get_pose

    def collect_single_scene(self, scene_name):
        """
        Args: Scene name in scene list
        """
        bag_path = self.raw_folder / (scene_name + '.bag')
        dst_folder = self.tgt_folder / scene_name
        if os.path.exists(dst_folder):
            os.mkdir(dst_folder)

        converter = BagConverter(bag_path, dst_folder)
        converter.save_intrinsics()
        converter.save_rgb()
        converter.save_pose()
        if self.get_pose:
            converter.save_pose()
        if self.get_depth:
            converter.save_depth()

    def read_multiple_scenes(self):
        for scene_name in self.scene_names:
            self.collect_single_scene(scene_name)
        

def main(args):

    converter = BagConverter('/home/minh/data.bag', '/home/minh/a')
    converter.save_intrinsics()
    converter.save_rgb()
    converter.save_pose()

if __name__ == '__main__':
    main(sys.argv)
