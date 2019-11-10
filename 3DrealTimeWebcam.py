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

device = torch.device('cuda')

print('Load Plot Functions')

def show3Dpose(ax,data3d,is_raw=False,nose=True,view_x=30,view_y=270):

    if nose == True:    
        index_0 = np.array([ 9, 8,14,15, 8,11,12, 8, 7, 0, 1, 2, 0, 4, 5])
        index_1 = np.array([ 8,14,15,16,11,12,13, 7, 0, 1, 2, 3, 4, 5, 6])
        side =             [ 2, 1, 1, 1, 0, 0, 0, 3, 3, 1, 1, 1, 0, 0, 0]      
        if data3d.shape[0] == 16:
            coord = np.zeros((17, 3))
            coord[1:, :] = data3d
        else:
            coord = np.reshape(data3d, (-1, 3))        
        
    else:
        index_0 = np.array([ 9, 8,13,14, 8,10,11, 8, 7, 0, 1, 2, 0, 4, 5])
        index_1 = np.array([ 8,13,14,15,10,11,12, 7, 0, 1, 2, 3, 4, 5, 6])
        side =             [ 2, 1, 1, 1, 0, 0, 0, 3, 3, 1, 1, 1, 0, 0, 0]
        if data3d.shape[0] == 15:
            coord = np.zeros((16, 3))
            coord[1:, :] = data3d
        else:
            coord = np.reshape(data3d, (-1, 3))

    
    X = coord[:,0]
    if is_raw:
        Y = coord[:,1]
        Z = coord[:,2]
    else:
        Y = coord[:,2]
        Z = coord[:,1]
        
    

    for i in np.arange(len(index_0)):
        x = [coord[index_0[i], 0], coord[index_1[i], 0]]
        if is_raw:
            y = [coord[index_0[i], 1], coord[index_1[i], 1]]
            z = [coord[index_0[i], 2], coord[index_1[i], 2]]
        else:
            y = [coord[index_0[i], 2], coord[index_1[i], 2]]
            z = [coord[index_0[i], 1], coord[index_1[i], 1]]
    
        
        if side[i] == 0:
            ax.plot(x, y, z, lw=2, c="#ba2929")
        elif side[i] == 1:
            ax.plot(x, y, z, lw=2, c="#0e8745")
        elif side[i] == 2:
            ax.plot(x, y, z, lw=2, c="#1c7de6")
        else:
            ax.plot(x, y, z, lw=2, c="#6e6e6e")
    
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")

    ax.grid(False) 
#     plt.axis('off') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])   

    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
    
    
    # ax.set_aspect('equal')
    # ax.invert_yaxis()
    
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    ax.view_init(view_x, view_y)
    
    if not is_raw:
        ax.invert_zaxis()




print('Load Model')


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.1):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.1):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  14 * 2
        # 3d joints
        self.output_size = 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y


model = LinearModel()
model.to(device)

# model.load_state_dict(torch.load('6Layers_orth_x-axis(widthONE)_2dRootAlign_NoHead_forDemo_45.7655.pkl', map_location='cpu'))
model.load_state_dict(torch.load('6Layers_orth_x-axis(widthONE)_2dRootAlign_NoHead_forDemo_45.7655.pkl'))

print('Waiting for 2D Detection...')

n = 0
fig = plt.figure(figsize=(8,8))

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

            model.eval()
            pose_tensor = torch.Tensor(pose_output.reshape(1,-1)).to(device)
            poseOut = model(pose_tensor)
            poseOut_cpu = poseOut.data.cpu().numpy()
            poseOut_cpu = poseOut_cpu.reshape((-1,3))

            fig.clf()

            ax = fig.add_subplot(111, projection='3d')
            show3Dpose(ax,poseOut_cpu,is_raw=True,nose=True,view_x=20,view_y=315)
            
            plt.pause(.0001)
            

    except Exception as e:
        print (e)




    
    