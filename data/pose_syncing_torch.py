from turtle import forward
import torch
import numpy as np


def find_nb_idx(valid_timestamps, odom_timestamps):
    """
    Args: valid_timestamps: array of timestamps in which an image message is published [M, ]
          odom_timestamps: array of timestamps in which an odometry message is published [N, ] N > M   

    Return: [M, 2] Array of pairs of indices of neighbor timestamps w.r.t to every image timestamps 
    """

    time_diff = valid_timestamps[:, None]-odom_timestamps[None, :] # [M, N]

    ref_idx = torch.abs(time_diff).argmin(axis=-1) # [M,]
    ref_idx = ref_idx[:, None] # [M, 1]
    derived_idx = torch.clone(ref_idx) # [M, 1]

    ref_time_diff = torch.gather(input=time_diff, index=ref_idx, dim=1) # [M, 1]
    derived_idx[ref_time_diff>0] +=1 # [M, 1]
    derived_idx[ref_time_diff<0] -=1 # [M, 1]
    derived_idx = torch.clip(derived_idx, 0, len(ref_idx)-1)

    neighbor_idx = torch.cat([ref_idx, derived_idx], axis=-1) # [M, 2]
    neighbor_idx, _ = torch.sort(neighbor_idx, axis=-1) # [M, 2]
    
    
    return neighbor_idx

def linear_interpolation(x, x0, x1, y0, y1):
    """
    Args: x: Array of valid timestamps from image topic [M,]
          x0: Array of left-neighbor timestamps [M, ]
          x1: Array of right-neighbor timestamps [M, ]
          y0: Array of left-neighbor poses [M, 12]
          y1: Array of right-neighbor poses [M, 12]
    Return: y: The interpolated pose [M, 12]
    References: https://en.wikipedia.org/wiki/Linear_interpolation
    """
    
    y = torch.zeros_like(y0) # [M, 12]
    y[x0==x1] = y0[x0==x1] # [M-A, 12] Assign the unbounded values
    
    valid_idx = x0!=x1 # [A, 12]
    w0 = 1 - (x[valid_idx] - x0[valid_idx]) / (x1[valid_idx] - x0[valid_idx]) # [A, 12] 
    w1 = 1 - w0 # [A, 12]
    y[valid_idx] = y0[valid_idx] * w0[:, None] + y1[valid_idx] * w1[:, None] # [A, 12]
    
    return y # [M, 12]

def sync_pose(valid_timestamps, odom_timestamps, odom):
    """
    Args: valid_timestamps: array of timestamps in which an image message is published [M, ]
          odom_timestamps: array of timestamps in which an odometry message is published [N, ] N > M    
          odom: Array of full groundtruth poses recorded from /uav/odometry topic [N, 12]

    Return: [M, 12] New groundtruth after interpolation, has the same length with the number of image messages 
    """
    valid_timestamps = torch.tensor(np.array(valid_timestamps, dtype=np.float32))
    odom_timestamps = torch.tensor(np.array(odom_timestamps, dtype=np.float32))
    odom = torch.tensor(odom)

    neighbor_idx = find_nb_idx(valid_timestamps, odom_timestamps) # [M, 2]
    
    # neighbor_timestamps = torch.gather(input=odom_timestamps[:, None], index=neighbor_idx, dim=0) # [N, 1] & [M, 2] -> [M, 2]
    neighbor_timestamps = torch.take(input=odom_timestamps[:, None], index=neighbor_idx) # [N, 1] & [M, 2] -> [M, 2]
    
    neighbor_poses = odom[:, :] # [N, 12] -> [N, 12]
    neighbor_poses = neighbor_poses[neighbor_idx]  # [N, 1, 12] & [M, 2, 1] -> [M, 2, 12]
    print(neighbor_poses.size())
    synced_pose = linear_interpolation(x = valid_timestamps,
                        x0 = neighbor_timestamps[:, 0],
                        x1 = neighbor_timestamps[:, 1],
                        y0 = neighbor_poses[:, 0, :],
                        y1 = neighbor_poses[:, 1, :])

    return synced_pose.numpy()


class PoseSyncer(torch.nn.Module):
    def __init__(self):
        super(PoseSyncer, self).__init__()

    def forward(self, valid_timestamps, odom_timestamps, odom):
        return sync_pose(valid_timestamps, odom_timestamps, odom)


if __name__=='__main__':
    valid_timestamps = torch.tensor(
        range(0, 10, 1)
    ) # (10,)

    odom_timestamps = torch.linspace(0, 12, 23) #(23,)
    
    rd_odom = torch.ones([len(odom_timestamps), 12]) #(23, 12)
    print(sync_pose(valid_timestamps, odom_timestamps, rd_odom).shape)

    