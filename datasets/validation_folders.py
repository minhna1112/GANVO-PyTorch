import torch.utils.data as data
import torch
import numpy as np
from imageio import imread
from path import Path
import random
import argparse
import custom_transforms

def crawl_folders(folders_list):
        imgs = []
        depth = []
        for folder in folders_list:
            current_imgs = sorted(folder.files('*.jpg'))
            current_depth = []
            for img in current_imgs:
                d = img.dirname()/(img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                depth.append(d)
            imgs.extend(current_imgs)
            depth.extend(current_depth)
        return imgs, depth


def load_as_float(path):
    return imread(path).astype(np.float32)


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.imgs, self.depth = crawl_folders(self.scenes)
        self.transform = transform

    def __getitem__(self, index):
        img = load_as_float(self.imgs[index])
        depth = np.load(self.depth[index]).astype(np.float32)
        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]
        return img, depth

    def __len__(self):
        return len(self.imgs)


class ValidationSetWithPose(data.Dataset):
    """A sequence validation data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_1/cam.txt
        root/scene_1/pose.txt
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .
    """

    def __init__(self, root, seed=None, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi: the middle element of the sequence: 1 or 2
        demi_length = (sequence_length-1)//2
        # shifts: [-1, 1] or [-2, -1, 1, 2]
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            # [Image_numbers, 3, 4]
            # First row of poses.txt files is always an identity transformation: R: identity, t: 0,0,0
            poses = np.genfromtxt(scene/'poses.txt').reshape((-1, 3, 4))
            # Homogenous transformation matrices [Image numbers, 4, 4]
            poses_4D = np.zeros((poses.shape[0], 4, 4)).astype(np.float32)
            poses_4D[:, :3] = poses
            poses_4D[:, 3, 3] = 1
            # [3, 3]
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            # [Number of images, ]
            imgs = sorted(scene.files('*.jpg'))
            assert(len(imgs) == poses.shape[0])
            if len(imgs) < sequence_length:
                continue
            # (1, 499) (2, 498) Construc a list of sample where each sample is :
            # Sample length:  sequence length
            # Number of samples =  number of images - 2*demilength = number of images - (seq_length-1)                                    
            for i in range(demi_length, len(imgs)-demi_length):
                # Take the middle image as target view
                tgt_img = imgs[i]
                # REad depth
                d = tgt_img.dirname()/(tgt_img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                sample = {'intrinsics': intrinsics, 'tgt': tgt_img, 'ref_imgs': [], 'poses': [], 'depth': d}
                # First pose of each sample (Odometry) (RElative to first frame of trajectory)
                first_pose = poses_4D[i - demi_length] # aka First pose, [1,4,4]
                # Convert every odometry in the seqence to "First frame of sequence" view
                sample['poses'] = (np.linalg.inv(first_pose) @ poses_4D[i - demi_length: i + demi_length + 1])[:, :3] #[seq_length, 3,4]
                # Append source view images
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sample['poses'] = np.stack(sample['poses'])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        depth = np.load(sample['depth']).astype(np.float32)
        poses = sample['poses']
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, _ = self.transform([tgt_img] + ref_imgs, None)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]

        return tgt_img, ref_imgs, depth, poses

    def __len__(self):
        return len(self.samples)


class ValidationSetWithPoseOnly(data.Dataset):
    """A sequence validation data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_1/pose.txt
        root/scene_2/0000000.jpg
        .
    """

    def __init__(self, root, seed=None, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        # demi: the middle element of the sequence: 1 or 2
        demi_length = (sequence_length-1)//2
        # shifts: [-1, 1] or [-2, -1, 1, 2]
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            # [Image_numbers, 3, 4]
            # First row of poses.txt files is always an identity transformation: R: identity, t: 0,0,0
            poses = np.genfromtxt(scene/'poses.txt').reshape((-1, 3, 4))
            # Homogenous transformation matrices [Image numbers, 4, 4]
            poses_4D = np.zeros((poses.shape[0], 4, 4)).astype(np.float32)
            poses_4D[:, :3] = poses
            poses_4D[:, 3, 3] = 1
            # [3, 3]
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            # [Number of images, ]
            imgs = sorted(scene.files('*.jpg'))
            assert(len(imgs) == poses.shape[0])
            if len(imgs) < sequence_length:
                continue
            # (1, 499) (2, 498) Construc a list of sample where each sample is :
            # Sample length:  sequence length
            # Number of samples =  number of images - 2*demilength = number of images - (seq_length-1)                                    
            for i in range(demi_length, len(imgs)-demi_length):
                # Take the middle image as target view
                tgt_img = imgs[i]
                sample = {'intrinsics': intrinsics, 'tgt': tgt_img, 'ref_imgs': [], 'poses': []}
                # First pose of each sample (Odometry) (RElative to first frame of trajectory)
                first_pose = poses_4D[i - demi_length] # aka First pose, [1,4,4]
                # Convert every odometry in the seqence to "First frame of sequence" view
                sample['poses'] = (np.linalg.inv(first_pose) @ poses_4D[i - demi_length: i + demi_length + 1])[:, :3] #[seq_length, 3,4]
                # Append source view images
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sample['poses'] = np.stack(sample['poses'])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        poses = sample['poses']
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, _ = self.transform([tgt_img] + ref_imgs, None)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]

        return tgt_img, ref_imgs,  poses

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='GANVO training on KITTI-formatted Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size') 
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')   
    
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    args = parser.parse_args()

    val_set = ValidationSetWithPoseOnly(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        sequence_length=args.sequence_length
    )
    #Dataloader
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)