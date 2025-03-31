import cv2
import math
import numpy as np
import os
import time
import torch
from os_reconstruction.pix2norm import Pix2NormalNet
from os_reconstruction.rgb2norm import RGB2NormalNet
from scipy import fftpack

class PointCloudReconstuctor:
    '''
    A class for reconstructing point clouds using depth maps.

    Parameters
    ----------
    width : int, optional -
        The width of the image, by default 640
    height : int, optional -
        The height of the image, by default 480
        
    Attributes
    ----------
    device_type : str - 
        The type of device to be used, default is 'cpu'
    weights_path : str - 
        The path to the weights file, default is 'weights/pix2norm.pt'
    depthmap_zero : numpy.ndarray
        A numpy array representing the initial depth map with zeros
    depthmap_zero_frames : int -
        Number of frames for resetting the depth map to zero
    depthmap_zero_counter : int - 
        Counter to keep track of frames for resetting the depth map
        
    References
    ----------
    [1] GelSight Robotics GitHub repository. GelSight Inc.
        https://github.com/gelsightinc/gsrobotics
    '''
    def __init__(self, width=640, height=480):
        self.device_type  = 'cpu'
        self.weights_path = 'weights/pix2norm_v1.2.pt'
        #self.weights_path = 'weights/nnmini.pt'    
        
        self.depthmap_zero         = np.zeros((height, width))
        self.depthmap_zero_frames  = 20
        self.depthmap_zero_counter = 0
        self.load_nn(self.weights_path, self.device_type)
    
    def load_nn(self, net_path, device_type):
        '''
        Load the neural network model.

        Parameters
        ----------
        net_path : str -
            The path to the neural network weights file.
        device_type : str -
            The type of device to load the model onto ('cpu' or 'cuda').

        Returns
        -------
        torch.nn.Module -
            The loaded neural network model.
        '''
        self.device_type = device_type
        device           = torch.device(device_type)

        if not os.path.isfile(net_path):
            print('Error opening ', net_path, ' does not exist')
            return

        net = Pix2NormalNet().float().to(device)
        #net = RGB2NormalNet().float().to(device)
        if device_type=='cuda': 
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(0))
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['model_state_dict'])
            
        self.net = net
        return self.net
    
    def get_depthmap(self, frame):
        '''
        Generate a depth map from the input frame.

        Parameters
        ----------
        frame : numpy.ndarray -
            The input frame.

        Returns
        -------
        numpy.ndarray -
            The generated depth map.
        '''
        height  = frame.shape[0]
        width   = frame.shape[1]
        contactmask = np.ones(frame.shape[:2])

        nx       = np.zeros(frame.shape[:2])
        ny       = np.zeros(frame.shape[:2])
        depthmap = np.zeros(frame.shape[:2])

        rgb                   = frame[np.where(contactmask)] / 255
        pixel_positions       = np.vstack(np.where(contactmask)).T
        pixel_positions[:, 0] = pixel_positions[:, 0] / height
        pixel_positions[:, 1] = pixel_positions[:, 1] / width
        
        features = np.column_stack((rgb, pixel_positions))
        features = torch.from_numpy(features).float().to(self.device_type)
        
        with torch.no_grad():
            self.net.eval()
            out = self.net(features)
            
        nx[np.where(contactmask)] = out[:, 0].cpu().detach().numpy()
        ny[np.where(contactmask)] = out[:, 1].cpu().detach().numpy()

        nz = np.sqrt(1 - nx ** 2 - ny ** 2)
        
        if np.isnan(nz).any():
            print ('nan found')
            nz[np.where(np.isnan(nz))] = np.nanmean(nz)
        
        gx = -nx / nz
        gy = -ny / nz
        gx, gy = np.reshape(gx, (height, width)), np.reshape(gy, (height, width))
        cv2.imshow('gx', cv2.flip(gx, 0))
        cv2.imshow('gy', cv2.flip(gy, 0))

        #depthmap = poisson_dct_neumann(gx, gy)
        depthmap = poisson_dst(gx, gy)
        depthmap = np.reshape(depthmap, (height, width))
                
        if self.depthmap_zero_counter < self.depthmap_zero_frames:
            self.depthmap_zero += depthmap
            print (f'{self.depthmap_zero_counter} - zeroing depth.')
            
            if self.depthmap_zero_counter == self.depthmap_zero_frames - 1:
                self.depthmap_zero /= self.depthmap_zero_counter
                print (f'{self.depthmap_zero_counter + 1} - zeroed depth.')
            
            self.depthmap_zero_counter += 1
            
        depthmap = depthmap - self.depthmap_zero 
        return depthmap
    
    def zero_depth(self, depthmap):
        height, width = depthmap.shape
        
        nx = cv2.Scharr(depthmap, cv2.CV_64F, 1, 0)
        ny = cv2.Scharr(depthmap, cv2.CV_64F, 0, 1)
        nz = np.sqrt(nx**2 + ny**2 + 1)
        
        if np.isnan(nz).any():
            print ('nan found')
            nz[np.where(np.isnan(nz))] = np.nanmean(nz)
        
        gx = -nx / nz
        gy = -ny / nz
        gx, gy = np.reshape(gx, (height, width)), np.reshape(gy, (height, width))
        cv2.imshow('gx', cv2.flip(gx, 0))
        cv2.imshow('gy', cv2.flip(gy, 0))
        
        #depthmap = poisson_dct_neumann(gx, gy)
        depthmap = poisson_dst(gx, gy)
        depthmap = np.reshape(depthmap, (height, width))
            
        if self.depthmap_zero_counter < self.depthmap_zero_frames:
            self.depthmap_zero += depthmap
            print (f'{self.depthmap_zero_counter} - zeroing depth.')
            
            if self.depthmap_zero_counter == self.depthmap_zero_frames - 1:
                self.depthmap_zero /= self.depthmap_zero_counter
                print (f'{self.depthmap_zero_counter + 1} - zeroed depth.')
            
            self.depthmap_zero_counter += 1
            
        depthmap = depthmap - self.depthmap_zero 
        return depthmap
    
def poisson_dct_neumann(gx, gy):
    '''
    Solve Poisson equation using DCT and Neumann boundary conditions.

    Parameters
    ----------
    gx : numpy.ndarray - 
        Gradient in the x-direction
    gy : numpy.ndarray - 
        Gradient in the y-direction

    Returns
    -------
    numpy.ndarray -
        Reconstructed image
    '''
    start_time = time.time()
    # Compute Laplacian
    height, width = gx.shape
    gxx = 1 * (gx[:, (list(range(1, width)) + [width - 1])] - gx[:, ([0] + list(range(width - 1)))])
    gyy = 1 * (gy[(list(range(1,height))+[height-1]), :] - gy[([0]+list(range(height-1))), :])
    f   = gxx + gyy

    # Boundary
    b          = np.zeros(gx.shape)
    b[0,1:-2]  = -gy[0,1:-2]
    b[-1,1:-2] = gy[-1,1:-2]
    b[1:-2,0]  = -gx[1:-2,0]
    b[1:-2,-1] = gx[1:-2,-1]
    b[0,0]     = (1/np.sqrt(2))*(-gy[0,0] - gx[0,0])
    b[0,-1]    = (1/np.sqrt(2))*(-gy[0,-1] + gx[0,-1])
    b[-1,-1]   = (1/np.sqrt(2))*(gy[-1,-1] + gx[-1,-1])
    b[-1,0]    = (1/np.sqrt(2))*(gy[-1,0]-gx[-1,0])

    # Modification near the boundaries to enforce the non-homogeneous Neumann BC
    f[0,1:-2]  = f[0,1:-2] - b[0,1:-2]
    f[-1,1:-2] = f[-1,1:-2] - b[-1,1:-2]
    f[1:-2,0]  = f[1:-2,0] - b[1:-2,0]
    f[1:-2,-1] = f[1:-2,-1] - b[1:-2,-1]

    # Modification near the corners
    f[0,-1]  = f[0,-1] - np.sqrt(2) * b[0,-1]
    f[-1,-1] = f[-1,-1] - np.sqrt(2) * b[-1,-1]
    f[-1,0]  = f[-1,0] - np.sqrt(2) * b[-1,0]
    f[0,0]   = f[0,0] - np.sqrt(2) * b[0,0]

    # Compute Discrete Cosine Transform of f
    tt   = fftpack.dct(f, norm='ortho')
    fcos = fftpack.dct(tt.T, norm='ortho').T

    # Compute Cosine Transform of z
    (x, y)      = np.meshgrid(range(1, width + 1), range(1, height + 1), copy=True)
    denominator = 4 * ((np.sin(0.5 * math.pi * x / (width)))**2 + (np.sin(0.5 * math.pi * y / (height)))**2)
    f = -fcos / denominator
    
    # Compute inverse discrete cosine transform
    tt     = fftpack.idct(f, norm='ortho')
    img_tt = fftpack.idct(tt.T, norm='ortho').T
    img_tt += img_tt.mean()
    img_tt -= img_tt.min()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time (DCT):", elapsed_time)
    return img_tt

def poisson_dst(gx, gy):
    '''
    Solve Poisson equation using DST

    Parameters
    ----------
    gx : numpy.ndarray - 
        Gradient in the x-direction
    gy : numpy.ndarray - 
        Gradient in the y-direction

    Returns
    -------
    numpy.ndarray -
        Reconstructed image
        
    References
    ----------
    [1] Raskar, Ramesh. "Fast Poisson solvers." (2007).
        https://web.media.mit.edu/~raskar/photo/code.pdf
    '''
    start_time = time.time()
    # Compute Laplacian
    height, width = gx.shape
    gxx = 1 * (gx[:, (list(range(1, width)) + [width - 1])] - gx[:, ([0] + list(range(width - 1)))])
    gyy = 1 * (gy[(list(range(1,height))+[height-1]), :] - gy[([0]+list(range(height-1))), :])
    f = gxx + gyy
    
    # Boundary
    b          = np.zeros(gx.shape)
    b[0,1:-2]  = -gy[0,1:-2]
    b[-1,1:-2] = gy[-1,1:-2]
    b[1:-2,0]  = -gx[1:-2,0]
    b[1:-2,-1] = gx[1:-2,-1]
    b[0,0]     = (1/np.sqrt(2))*(-gy[0,0] - gx[0,0])
    b[0,-1]    = (1/np.sqrt(2))*(-gy[0,-1] + gx[0,-1])
    b[-1,-1]   = (1/np.sqrt(2))*(gy[-1,-1] + gx[-1,-1])
    b[-1,0]    = (1/np.sqrt(2))*(gy[-1,0]-gx[-1,0])

    # Modification near the boundaries to enforce the non-homogeneous Neumann BC
    f[0,1:-2]  = f[0,1:-2] - b[0,1:-2]
    f[-1,1:-2] = f[-1,1:-2] - b[-1,1:-2]
    f[1:-2,0]  = f[1:-2,0] - b[1:-2,0]
    f[1:-2,-1] = f[1:-2,-1] - b[1:-2,-1]

    # Modification near the corners
    f[0,-1]  = f[0,-1] - np.sqrt(2) * b[0,-1]
    f[-1,-1] = f[-1,-1] - np.sqrt(2) * b[-1,-1]
    f[-1,0]  = f[-1,0] - np.sqrt(2) * b[-1,0]
    f[0,0]   = f[0,0] - np.sqrt(2) * b[0,0]

    # Compute Discrete Sine Transform of f
    tt   = fftpack.dst(f, norm='ortho')
    fsin = fftpack.dst(tt.T, norm='ortho').T

    # Compute Sine Transform of z
    (x, y) = np.meshgrid(range(1, width + 1), range(1, height + 1))
    denominator = (2 * np.cos(np.pi * x / (width - 1)) - 2) + (2 * np.cos(np.pi * y / (height - 1)) - 2)
    f = fsin / denominator

    # Compute Inverse DST
    tt  = fftpack.idst(f, norm='ortho')
    img_tt = fftpack.idst(tt.T, norm='ortho').T
    img_tt += img_tt.mean()
    img_tt -= img_tt.min()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time (DST):", elapsed_time)
    return img_tt