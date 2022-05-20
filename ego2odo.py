import numpy as np
#path to pose.txt file
ego = '/path'
#path to converted file
odo = '/path'
#from list string to transformation matrix (4x4)
def str2arr(str):
    arr = str.split(" ")
    arr = list(map(float, arr))
    arr = np.array(arr)
    arr = arr.reshape([3,4])
    arr1 = np.array([[0,0,0,1]])
    arr = np.concatenate((arr,arr1))
    return arr
def arr2str(arr):
    arr = arr[:3].flatten()
    arrtolist = arr.tolist()
    arrtostr = ' '.join(str(e) for e in arrtolist)
    return arrtostr
with open(ego) as f:
    read_pose = [s.strip() for s in f.readlines()]
    num_pose = len(read_pose)

odometry = []
for i in range(num_pose):
    odometry.append(str2arr(read_pose[i]))
for i in range(1,len(odometry)):
    odometry[i] = odometry[i]@odometry[i-1]
for i in range(len(odometry)):
    odometry[i] = arr2str(odometry[i])
odo = open(odo,'w')
with odo as f:
    f.write('\n'.join(odometry))
odo.close()
