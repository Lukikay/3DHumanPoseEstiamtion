import os
import re
import glob
import copy
import h5py
import json
import torch
import torch.optim
import numpy as np
import torch.nn as nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



n = 0

fig, ax = plt.subplots()

while(True):

    currentFile = 'webcamOutput/{}_keypoints.json'.format('%012d' % n)

    try:

        if os.path.isfile(currentFile):
            currentData = json.load(open(currentFile))
            n = n + 1
            
            currentPose = currentData["people"][0]["pose_keypoints_2d"]
            currentPose = np.array(currentPose).reshape(-1,3)[:,:2][:15,:]
            index2mpii = [8,9,10,11,12,13,14,1,0,5,6,7,2,3,4]
            currentPose = currentPose[index2mpii]

            width = (np.max(currentPose, axis = 0) - np.min(currentPose, axis = 0))
            widthONE = currentPose/width[0]
            widthONE = widthONE - np.tile( widthONE[0,:], [15, 1] )
            pose_output = np.delete(widthONE, [0], axis=0)*np.array([1,-1])
            
            plt.cla()

            ax.scatter(pose_output[:,0], pose_output[:,1], c='r', s=20, alpha=0.5)
            for i in range(pose_output.shape[0]):
                plt.text(pose_output[i,0], pose_output[i,1], str(i))
            ax.set_aspect('equal')
            plt.pause(.01)

            
    except Exception as e:
        print (e)
    
    