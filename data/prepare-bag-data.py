import rospy
import rosbag
import numpy as np

import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
from tqdm import tqdm
from path import Path
import os

from pose_syncing import sync_pose
import argparse
import matplotlib.pyplot as plt


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
        image_generator = self.bag.read_messages(topics='/rgb')
        print("Reading Images  ......")
        
  
        for i, (topic, msg, t) in enumerate(tqdm(image_generator)):
            im = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            im_path = self.dst_folder / (str(i).zfill(6) + '.jpg')
            # Append every timestamps that has an image message published
            self.valid_tstamps.append(t.to_sec())
            # print(type(t.to_nsec()))
            cv2.imwrite(im_path, im)       
                
    @staticmethod
    def quat2mat(w, x, y, z):
        """
        Args:  w,x,y,z: 4 quarternion coefficients
        Return: Corresponing 3x3 Rotation matrix 
        https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
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
        pose_generator = self.bag.read_messages('/hummingbird/ground_truth/pose_with_covariance')
        print("Reading Poses  ......")
        full_odom = []
        odom_ts = []
       
        for i, (topic, msg, t) in enumerate(tqdm(pose_generator)):
            px = float(msg.pose.pose.position.x) # float
            py = float(msg.pose.pose.position.y)
            pz = float(msg.pose.pose.position.z)

            ox = float(msg.pose.pose.orientation.x) # float
            oy = float(msg.pose.pose.orientation.y)
            oz = float(msg.pose.pose.orientation.z)
            ow = float(msg.pose.pose.orientation.w)

            t_vec = np.array([px,py,pz]).reshape([3,1]) # 3x1 Translational vector
            rot_mat = self.quat2mat(ow, ox, oy, oz) # 3x3 Rotation matrix
            T_mat = np.concatenate([rot_mat, t_vec], axis=1) # 3x4 Transformation matrix
            # Append the flattened Transformation (Pose vector) and the corresponding timestamps
            full_odom.append(T_mat.reshape([-1,12])) 
            odom_ts.append(t.to_sec())
            
        # Array with full number Gt Poses before syncing with images timestamps, save it for later comparison
        full_odom = np.concatenate(full_odom, axis=0)  # [N , 12]
        np.savetxt(self.dst_folder / 'poses_full.txt', full_odom)
        
        # Array of every timestamps that a Gt odometry message is published
        odom_ts = np.array(odom_ts) # [N,]
            
        print("syncing poses ..... ")
        # New Array with  GT Poses after syncing with images timestamps (Interpolated with neighbor GT poses)
        final_poses = sync_pose(valid_timestamps=np.array(self.valid_tstamps), odom_timestamps=odom_ts, odom=full_odom) # [M, 12]
        np.savetxt(self.dst_folder / 'poses.txt', final_poses)

    def save_depth(self):
        
        # image_generator = self.bag.read_messages(topics='/airsim_node/PX4/camera_1/DepthPlanar')
        image_generator = self.bag.read_messages(topics='/depth')
        print("Reading Depth Images  ......")
     
        for i, (topic, msg, t) in enumerate(tqdm(image_generator)):

            h, w = msg.height, msg.width
            dtype = np.dtype("float32") 
            # Depth message from Flightmare is encoded as 32FC1
            print(len(msg.data))
            im = np.ndarray(shape=(h, w),
                           dtype=dtype, buffer=msg.data)  
            print(im.max())
            out = np.copy(im)
            # out.setflags(write=1)
            # out = out / 1000.
            # Clip depth values that exceesds 100 (m)
            # out[im>=100] = 100
            # Save to binary file
            depth_path = self.dst_folder / (str(i).zfill(6) + '.npy')
            # np.save(depth_path, out)

            if i == 550:
                plt.imshow(out, 'gray')
                plt.show()
                print(out.max())
                break


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

        if not os.path.exists(self.tgt_folder):
            os.mkdir(self.tgt_folder)

    def collect_single_scene(self, scene_name):
        """
        Args: Scene name in scene list
        """
        bag_path = self.raw_folder / (scene_name + '.bag')
        dst_folder = self.tgt_folder / scene_name
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)

        converter = BagConverter(bag_path, dst_folder)
        # converter.save_intrinsics()
        converter.save_rgb()
        converter.save_pose()
        if self.get_pose:
            converter.save_pose()
        if self.get_depth:
            converter.save_depth()

    def read_multiple_scenes(self):
        for scene_name in self.scene_names:
            print("-------------------------------------------------------")
            print(f"Processing Sequence {scene_name}")
            self.collect_single_scene(scene_name)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", metavar='DIR', type=Path,
                    help='path to original dataset')
    parser.add_argument("--dst-dir", metavar='DIR', type=Path,
                    help='path to the destination folder for saving converted data')
    parser.add_argument("--with-depth", action='store_true',
                        help="If available (e.g. with KITTI), will store depth ground truth along with images, for validation")
    parser.add_argument("--with-pose", action='store_true',
                        help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
    
    args = parser.parse_args()

    reader = BagDataReader(
        raw_folder=args.raw_dir,
        tgt_folder=args.dst_dir,
        get_depth=args.with_depth,
        get_pose=args.with_pose
    )

    reader.read_multiple_scenes()


if __name__ == '__main__':
    main()
