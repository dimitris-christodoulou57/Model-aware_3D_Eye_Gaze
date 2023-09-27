"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import copy
import math
import cv2
import os
import torch
#import wandb

from PIL import Image
from skimage import draw
from itertools import chain
from skimage.transform import rescale
from skimage.segmentation.boundaries import find_boundaries
from scipy.ndimage import distance_transform_edt as distance

from einops import rearrange
from Visualitation_TEyeD.gaze_estimation import generate_gaze_gt

EPS = 1e-40

# Helper classes
class my_ellipse():
    def __init__(self, param):
        '''
        Accepts parameterized form
        '''
        self.EPS = 1e-3
        if param is not list:
            self.param = param
            self.mat = self.param2mat(self.param)
            self.quad = self.mat2quad(self.mat)
        else:
            if param:
                raise Exception('my_ellipse only accepts numpy arrays')

    def param2mat(self, param):
        cx, cy, a, b, theta = tuple(param)
        H_rot = rotation_2d(-theta)
        H_trans = trans_2d(-cx, -cy)

        A, B = 1/a**2, 1/b**2
        Q = np.array([[A, 0, 0], [0, B, 0], [0, 0, -1]])
        mat = H_trans.T @ H_rot.T @ Q @ H_rot @ H_trans
        return mat

    def mat2quad(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.EPS, 'Conic form incorrect'
        a, b, c, d, e, f = mat[0,0], 2*mat[0, 1], mat[1,1], 2*mat[0, 2], 2*mat[1, 2], mat[-1, -1]
        return np.array([a, b, c, d, e, f])

    def quad2param(self, quad):
        mat = self.quad2mat(quad)
        param = self.mat2param(mat)
        return param

    def quad2mat(self, quad):
        a, b, c, d, e, f = tuple(quad)
        mat = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])
        return mat

    def mat2param(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.EPS, 'Conic form incorrect'
        # Estimate rotation
        theta = self.recover_theta(mat)
        # Estimate translation
        tx, ty = self.recover_C(mat)
        # Invert translation and rotation
        H_rot = rotation_2d(theta)
        H_trans = trans_2d(tx, ty)
        mat_norm = H_rot.T @ H_trans.T @ mat @ H_trans @ H_rot
        major_axis = np.sqrt(1/mat_norm[0,0])
        minor_axis = np.sqrt(1/mat_norm[1,1])
        area = np.pi*major_axis*minor_axis
        return np.array([tx, ty, major_axis, minor_axis, theta, area])

    def phi2param(self, xm, ym):
        '''
        Given phi values, compute ellipse parameters

        Parameters
        ----------
        Phi : np.array [5, ]
            for information on Phi values, please refer to ElliFit.
        xm : int
        ym : int

        Returns
        -------
        param : np.array [5, ].
            Ellipse parameters, [cx, cy, a, b, theta]

        '''
        try:
            x0=(self.Phi[2]-self.Phi[3]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            y0=(self.Phi[0]*self.Phi[3]-self.Phi[2]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            term2=np.sqrt(((1-self.Phi[0])**2+4*(self.Phi[1])**2))
            term3=(self.Phi[4]+(y0)**2+(x0**2)*self.Phi[0]+2*self.Phi[1])
            term1=1+self.Phi[0]
            print(term1, term2, term3)
            b=(np.sqrt(2*term3/(term1+term2)))
            a=(np.sqrt(2*term3/(term1-term2)))
            alpha=0.5*np.arctan2(2*self.Phi[1],1-self.Phi[0])
            model = [x0+xm, y0+ym, a, b, -alpha]
        except:
            print('Inappropriate model generated')
            model = [np.nan, np.nan, np.nan, np.nan, np.nan]
        if np.all(np.isreal(model)) and np.all(~np.isnan(model)) and np.all(~np.isinf(model)):
            model = model
        else:
            model = [-1, -1, -1, -1, -1]
        return model

    def recover_theta(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        #print('a: {}. b: {}. c: {}'.format(a, b, c))
        if abs(b)<=EPS and a<=c:
            theta = 0.0
        elif abs(b)<=EPS and a>c:
            theta=np.pi/2
        elif abs(b)>EPS and a<=c:
            theta=0.5*np.arctan2(b, (a-c))
        elif abs(b)>EPS and a>c:
            #theta = 0.5*(np.pi + np.arctan(b/(a-c)))
            theta = 0.5*np.arctan2(b, (a-c))
        else:
            print('Unknown condition')
        return theta

    def recover_C(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        tx = (2*c*d - b*e)/(b**2 - 4*a*c)
        ty = (2*a*e - b*d)/(b**2 - 4*a*c)
        return (tx, ty)

    def transform(self, H):
        '''
        Given a transformation matrix H, modify the ellipse
        '''
        mat_trans = np.linalg.inv(H.T) @ self.mat @ np.linalg.inv(H)
        return self.mat2param(mat_trans), self.mat2quad(mat_trans), mat_trans

    def recover_Phi(self):
        '''
        Generate Phi
        '''
        x, y = self.generatePoints(50, 'random')
        data_pts = np.stack([x, y], axis=1)
        ellipseFit = ElliFit(**{'data':data_pts})
        return ellipseFit.Phi

    def verify(self, pts):
        '''
        Given an array of points Nx2, verify the ellipse model
        '''
        N = pts.shape[0]
        pts = np.concatenate([pts, np.ones((N, 1))], axis=1)
        err = 0.0
        for i in range(0, N):
            err+=pts[i, :]@self.mat@pts[i, :].T # Note that the transpose here is irrelevant
        return np.inf if (N==0) else err/N

    def generatePoints(self, N, mode):
        '''
        Generates 8 points along the periphery of an ellipse. The mode dictates
        the uniformity between points.
        mode: str
        'equiAngle' - Points along the periphery with angles [0:45:360)
        'equiSlope' - Points along the periphery with tangential slopes [-1:0.5:1)
        'random' - Generate N points randomly across the ellipse
        '''

        a = self.param[2]
        b = self.param[3]

        alpha = (a*np.sin(self.param[-1]))**2 + (b*np.cos(self.param[-1]))**2
        beta = (a*np.cos(self.param[-1]))**2 + (b*np.sin(self.param[-1]))**2
        gamma = (a**2 - b**2)*np.sin(2*self.param[-1])

        if mode == 'equiSlope':
            slope_list = [1e-6, 1, 1000, -1]
            K_fun = lambda m_i:  (m_i*gamma + 2*alpha)/(2*beta*m_i + gamma)

            x_2 = [((a*b)**2)/(alpha + beta*K_fun(m)**2 - gamma*K_fun(m)) for m in slope_list]

            x = [(+np.sqrt(val), -np.sqrt(val)) for val in x_2]
            y = []
            for i, m in enumerate(slope_list):
                y1 = -x[i][0]*K_fun(m)
                y2 = -x[i][1]*K_fun(m)
                y.append((y1, y2))
            y_r = np.array(list(chain(*y))) + self.param[1]
            x_r = np.array(list(chain(*x))) + self.param[0]

        if mode == 'equiAngle':

            T = 0.5*np.pi*np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            N = len(T)
            x = self.param[2]*np.cos(T)
            y = self.param[3]*np.sin(T)
            H_rot = rotation_2d(self.param[-1])
            X1 = H_rot.dot(np.stack([x, y, np.ones(N, )], axis=0))

            x_r = X1[0, :] + self.param[0]
            y_r = X1[1, :] + self.param[1]

        elif mode == 'random':
            T = 2*np.pi*(np.random.rand(N, ) - 0.5)
            x = self.param[2]*np.cos(T)
            y = self.param[3]*np.sin(T)
            H_rot = rotation_2d(self.param[-1])
            X1 = H_rot.dot(np.stack([x, y, np.ones(N, )], axis=0))
            x_r = X1[0, :] + self.param[0]
            y_r = X1[1, :] + self.param[1]

        else:
            print('Mode is not defined')

        return x_r, y_r

class ElliFit():
    def __init__(self, **kwargs):
        self.data = np.array([]) # Nx2
        self.W = np.array([])
        self.Phi = []
        self.pts_lim = 6*2
        for k, v in kwargs.items():
            setattr(self, k, v)
        if np.size(self.W):
            self.weighted = True
        else:
            self.weighted = False
        if np.size(self.data) > self.pts_lim:
            self.model = self.fit()
            self.error = np.mean(self.fit_error(self.data))
        else:
            self.model = [-1, -1, -1, -1, -1]
            self.Phi = [-1, -1, -1, -1, -1]
            self.error = np.inf

    def fit(self):
        # Code implemented from the paper ElliFit
        xm = np.mean(self.data[:, 0])
        ym = np.mean(self.data[:, 1])
        x = self.data[:, 0] - xm
        y = self.data[:, 1] - ym
        X = np.stack([x**2, 2*x*y, -2*x, -2*y, -np.ones((np.size(x), ))], axis=1)
        Y = -y**2
        if self.weighted:
            self.Phi = np.linalg.inv(
                X.T.dot(np.diag(self.W)).dot(X)
                ).dot(
                    X.T.dot(np.diag(self.W)).dot(Y)
                    )
        else:
            try:
                self.Phi = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
            except:
                self.Phi = -1*np.ones(5, )
        try:
            x0=(self.Phi[2]-self.Phi[3]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            y0=(self.Phi[0]*self.Phi[3]-self.Phi[2]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            term2=np.sqrt(((1-self.Phi[0])**2+4*(self.Phi[1])**2))
            term3=(self.Phi[4] + (y0)**2 + (x0**2)*self.Phi[0] + 2*self.Phi[1])
            term1=1+self.Phi[0]
            b=(np.sqrt(2*term3/(term1+term2)))
            a=(np.sqrt(2*term3/(term1-term2)))
            alpha=0.5*np.arctan2(2*self.Phi[1],1-self.Phi[0])
            model = [x0+xm, y0+ym, a, b, -alpha]
        except:
            print('Inappropriate model generated')
            model = [np.nan, np.nan, np.nan, np.nan, np.nan]
        if np.all(np.isreal(model)) and np.all(~np.isnan(model)) and np.all(~np.isinf(model)):
            model = model
        else:
            model = [-1, -1, -1, -1, -1]
        return model

    def fit_error(self, data):
        # General purpose function to find the residual
        # model: xc, yc, a, b, theta
        term1 = (data[:, 0] - self.model[0])*np.cos(self.model[-1])
        term2 = (data[:, 1] - self.model[1])*np.sin(self.model[-1])
        term3 = (data[:, 0] - self.model[0])*np.sin(self.model[-1])
        term4 = (data[:, 1] - self.model[1])*np.cos(self.model[-1])
        res = (1/self.model[2]**2)*(term1 - term2)**2 + \
            (1/self.model[3]**2)*(term3 + term4)**2 - 1
        return np.abs(res)

class ransac():
    def __init__(self, data, model, n_min, mxIter, Thres, n_good):
        self.data = data
        self.num_pts = data.shape[0]
        self.model = model
        self.n_min = n_min
        self.D = n_good if n_min < n_good else n_min
        self.K = mxIter
        self.T = Thres
        self.bestModel = self.model(**{'data': data}) #Fit function all data points

    def loop(self):
        i = 0
        if self.num_pts > self.n_min:
            while i <= self.K:
                # Pick n_min points at random from dataset
                inlr = np.random.choice(self.num_pts, self.n_min, replace=False)
                loc_inlr = np.in1d(np.arange(0, self.num_pts), inlr)
                outlr = np.where(~loc_inlr)[0]
                potModel = self.model(**{'data': self.data[loc_inlr, :]})
                listErr = potModel.fit_error(self.data[~loc_inlr, :])
                inlr_num = np.size(inlr) + np.sum(listErr < self.T)
                if inlr_num > self.D:
                    pot_inlr = np.concatenate([inlr, outlr[listErr < self.T]], axis=0)
                    loc_pot_inlr = np.in1d(np.arange(0, self.num_pts), pot_inlr)
                    betterModel = self.model(**{'data': self.data[loc_pot_inlr, :]})
                    if betterModel.error < self.bestModel.error:
                        self.bestModel = betterModel
                i += 1
        else:
            # If the num_pts <= n_min, directly return the model
            self.bestModel = self.model(**{'data': self.data})
        return self.bestModel

# Helper functions
def rotation_2d(theta):
    # Return a 2D rotation matrix in the anticlockwise direction
    c, s = np.cos(theta), np.sin(theta)
    H_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])
    return H_rot

def trans_2d(cx, cy):
    H_trans = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1]])
    return H_trans

def scale_2d(sx, sy):
    H_scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1]])
    return H_scale

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def transformPoints(x, y, H):
    N = np.size(x)
    pts = np.stack([x, y, np.ones(N, )], axis=1) \
        if (N > 1) else \
        np.array([x, y, 1])
    pts = H.dot(pts.T)
    ox = pts[0, :] if N > 1 else pts[0]
    oy = pts[1, :] if N > 1 else pts[1]
    return (ox, oy)


def fillHoles(mask):
    # Fill holes in mask
    x_hole, y_hole = np.where(mask == 0)
    for x, y in zip(x_hole, y_hole):
        # Fill hole with the mean value
        opts = mask[x-2:x+2, y-2:y+2].reshape(-1)
        if (not isinstance(opts, list)) & (opts.size != 0) & (sum(opts) != 0):
            mask[x, y] = np.round(np.mean(opts[opts != 0]))
    return mask


def dummy_data(shape):

    num_of_frames = shape[0]

    true_list = [True] * num_of_frames
    false_list = [False] * num_of_frames

    data_dict = {}
    data_dict['is_bad'] = np.stack(true_list, axis=0) # You so naughty boi

    data_dict['mask'] = -1*np.ones(shape)
    data_dict['image'] = np.zeros(shape, dtype=np.uint8)
    data_dict['ds_num'] = -1*np.ones(num_of_frames) # Garbage number
    data_dict['pupil_center'] = -1*np.ones((num_of_frames,2, ))
    data_dict['iris_ellipse'] = -1*np.ones((num_of_frames,5, ))
    data_dict['pupil_ellipse'] = -1*np.ones((num_of_frames,5, ))

    # Ability to traceback
    data_dict['im_num'] = -1*np.ones(num_of_frames)  # Garbage number
    data_dict['archName'] = 'do_the_boogie!'

    # Keep flags as separate entries
    data_dict['mask_available'] = np.stack(false_list, axis=0)
    data_dict['pupil_center_available'] = np.stack(false_list, axis=0)
    data_dict['iris_ellipse_available'] = np.stack(false_list, axis=0)
    data_dict['pupil_ellipse_available'] = np.stack(false_list, axis=0)
    return data_dict


def fix_batch(data_dict):
    loc_bad = np.where(np.array(data_dict['is_bad']))[0]
    loc_good = np.where(np.array(~data_dict['is_bad']))[0]

    for bad_idx in loc_bad.tolist():
        random_good_idx = int(np.random.choice(loc_good, 1).item())
        print('replacing {} with {}'.format(bad_idx, random_good_idx))

        for key in data_dict.keys():
            data_dict[key][bad_idx] = data_dict[key][random_good_idx]
    return data_dict


def one_hot2dist(posmask):
    # Input: Mask. Will be converted to Bool.
    h, w = posmask.shape
    mxDist = np.sqrt((h-1)**2 + (w-1)**2)
    if np.any(posmask):
        assert len(posmask.shape) == 2
        res = np.zeros_like(posmask)
        posmask = posmask.astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        res = res/mxDist
    else:
        # No valid element exists for that category
        res = np.zeros_like(posmask)
    return res

def label2onehot(Label):
    Label = (np.arange(4) == Label[..., None]).astype(np.uint8)
    Label = np.rollaxis(Label, 2)
    return Label

def clean_mask(mask):
    '''
    Input: HXWXC mask
    Output: Cleaned mask
    cleans the mask by contraction and dilation of edges maps
    '''
    outmask = np.zeros_like(mask)
    classes_available = np.unique(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for cls_idx in np.nditer(classes_available):
        I = 255*np.uint8(mask == cls_idx)
        I = cv2.erode(I, kernel, iterations=1)
        I = cv2.dilate(I, kernel, iterations=1)
        outmask[I.astype(np.bool)] = cls_idx
    return outmask


def simple_string(ele):
    '''
    ele: String which needs to be stripped of all special characters and moved
    to lower subscript
    Returns a simplified string
    '''
    if type(ele) is list:
        ele = [''.join(e.lower() for e in str(string_ele) if e.isalnum())
               for string_ele in ele]
    else:
        ele = ''.join(e.lower() for e in str(ele) if e.isalnum())
    return ele


def scale_by_ratio(data_dict, scale_ratio, train_with_mask):

    more_features = False

    if 'Dikablis' in data_dict['archName']:
        more_features = True

    num_of_frames = data_dict['image'].shape[0]

    dsize = (round(scale_ratio*data_dict['image'].shape[2]),
             round(scale_ratio*data_dict['image'].shape[1]))

    H = np.array([[scale_ratio, 0, 0],
                  [0, scale_ratio, 0],
                  [0, 0, 1]])

    image_list = []
    if train_with_mask:
        mask_list = []

    for i in range(num_of_frames):
        image_list.append(cv2.resize(data_dict['image'][i], dsize,
                                        interpolation=cv2.INTER_LANCZOS4))
        if train_with_mask:
            mask_list.append(cv2.resize(data_dict['mask'][i],  dsize,
                                       interpolation=cv2.INTER_NEAREST))

        if data_dict['pupil_ellipse_available'][i]:
            data_dict['pupil_ellipse'][i] =\
                my_ellipse(data_dict['pupil_ellipse'][i]).transform(H)[0][:-1]

        if data_dict['iris_ellipse_available'][i]:
            data_dict['iris_ellipse'][i] = \
                my_ellipse(data_dict['iris_ellipse'][i]).transform(H)[0][:-1]

        if data_dict['pupil_center_available'][i]:
            data_dict['pupil_center'][i] = H[:2, :2].dot(data_dict['pupil_center'][i])

        if more_features:
            data_dict['eyeball'][i] *= scale_ratio
            data_dict['iris_lm_2D'][i] *= scale_ratio 
            data_dict['pupil_lm_2D'][i] *= scale_ratio


    data_dict['image'] = np.stack(image_list, axis=0)
    if train_with_mask:
        data_dict['mask'] = np.stack(mask_list, axis=0)

    return data_dict


def pad_to_shape(data_dict, to_size, mode='edge'):
    '''
    Pad an image and transform ellipses to a desired shape. The default mode
    replicates the border pixels to ensure there are no 0 pixels -> this is
    based on Shalini's recommendation.
    '''

    assert len(data_dict['image'].shape) == 3, 'Image required to be grayscale and 1 more dimension to create the volume'
    num_of_frames , r_in, c_in = data_dict['image'].shape
    r_out, c_out = to_size

    inc_r = 0.5*(r_out - r_in)
    inc_c = 0.5*(c_out - c_in)

    for i in range(num_of_frames):
        data_dict['mask'][i] = np.pad(data_dict['mask'][i],
                                   ((math.floor(inc_r), math.ceil(inc_r)),
                                    (math.floor(inc_c), math.ceil(inc_c))),
                                   mode='constant')
        data_dict['image'][i] = np.pad(data_dict['image'][i],
                                    ((math.floor(inc_r), math.ceil(inc_r)),
                                     (math.floor(inc_c), math.ceil(inc_c))),
                                    mode='edge')

        if data_dict['pupil_center_available'][i]:
            data_dict['pupil_center'][i,:2] += np.array([inc_c, inc_r])

        if data_dict['pupil_ellipse_available'][i]:
            data_dict['pupil_ellipse'][i,:2] += np.array([inc_c, inc_r])

        if data_dict['iris_ellipse_available'][i]:
            data_dict['iris_ellipse'][i,:2] += np.array([inc_c, inc_r])

    #specify the dimensions of the image and num of frames
    output_size = (num_of_frames, r_out, c_out)

    assert data_dict['image'].shape == output_size, 'Padded image must match shape'

    return data_dict


class mod_scalar():

    def __init__(self, xlims, ylims):
        self.slope = np.diff(ylims)/np.diff(xlims)
        self.intercept = ylims[1] - self.slope*xlims[1]
        self.xlims = xlims
        self.ylims = ylims

    def get_scalar(self, x_input):

        if x_input > self.xlims[1]:
            return self.ylims[1]

        if x_input < self.xlims[0]:
            return self.ylims[0]

        return self.slope*x_input + self.intercept


# def linVal(x, xlims, ylims, offset):
#     '''
#     Given xlims (x_min, x_max) and ylims (y_min, y_max), i.e, start and end,
#     compute the value of y=f(x). Offset contains the x0 such that for all x<x0,
#     y is clipped to y_min.
#     '''
#     if x < offset:
#         return ylims[0]
#     elif x > xlims[1]:
#         return ylims[1]
#     else:
#         y = (np.diff(ylims)/np.diff(xlims))*(x - offset)
#         return y.item()

def getValidPoints(LabelMat, isPartSeg=True, legacy=True):
    '''
    PartSeg annotations:
        pupil: label == 3, iris: label == 2
    EllSeg annotations:
        pupil: label == 2, iris: label == 1
    '''
    if legacy:
        # Convert mask to 0 -> 255
        im = np.uint8(255*LabelMat.astype(np.float32)/LabelMat.max())
    
        # Find edges for mask and inverted mask to ensure edge points from both
        # inside, and outside an edge, are addressed
        edges = cv2.Canny(im, 50, 100) + cv2.Canny(255-im, 50, 100)
        
    else:
        # skimage has a convenient function for us to use
        edges = find_boundaries(LabelMat)
        
    r, c = np.where(edges)

    # Initialize list of valid points
    pupilPts, irisPts = [], []
    for loc in zip(c, r):
        temp = LabelMat[loc[1]-1:loc[1]+2, loc[0]-1:loc[0]+2]

        # Filter out invalid points
        if isPartSeg:
            # Pupil points cannot have sclera or skin
            condPupil = np.any(temp == 0) or np.any(temp == 1) or (temp.size == 0)
            
            # Iris points cannot have skin or pupil
            condIris = np.any(temp == 0) or np.any(temp == 3) or (temp.size == 0)
        else:
            
            # Pupil points cannot have skin
            condPupil = np.any(temp == 0) or (temp.size == 0)
            
            # Iris points cannot have pupil
            condIris = np.any(temp == 2) or (temp.size == 0)

        # Keep valid points
        pupilPts.append(np.array(loc)) if not condPupil else None
        irisPts.append(np.array(loc)) if not condIris else None

    pupilPts = np.stack(pupilPts, axis=0) if len(pupilPts) > 0 else []
    irisPts = np.stack(irisPts, axis=0) if len(irisPts) > 0 else []
    return pupilPts, irisPts
    

def stackall_Dict(D):
    for key, value in D.items():
        if value:
            # Ensure it is not empty
            if type(D[key]) is list:
                print('Stacking: {}'.format(key))
                D[key] = np.stack(value, axis=0)
            elif type(D[key]) is dict:
                stackall_Dict(D[key])
    return D

def extract_datasets(subsets):
    '''
    subsets: contains an array of strings
    '''
    ds_idx = [str(ele).split('_')[0] for ele in np.nditer(subsets)]
    ds_present, ds_id = np.unique(ds_idx, return_inverse=True)
    return ds_present, ds_id

def convert_to_list_entries(data_dict):
    # Move everything back to numpy if it exists in torch

    for key, item in data_dict.items():
        if 'torch' in str(type(item)):
            data_dict[key] = item.detach().cpu().squeeze().numpy()

    # Generate empty template based on dictionary
    num_entries = data_dict[key].shape[0]
    out = []
    for ii in range(num_entries):
        out.append({key:item[ii] for key, item in data_dict.items() if 'numpy' in str(type(item)) and 'latent' not in key})
    return out

def plot_images_with_annotations(data_dict,
                                 args,
                                 show=True,
                                 write=None,
                                 rendering=False,
                                 mask = False,
                                 subplots=None,
                                 is_predict=True,
                                 plot_annots=True,
                                 remove_saturated=True,
                                 is_list_of_entries=True,
                                 mode=None,
                                 epoch=0,
                                 batch=0
                                 ):

    if not is_list_of_entries:
        list_data_dict = convert_to_list_entries(copy.deepcopy(data_dict))
    else:
        list_data_dict = data_dict

    #num_entries = len(list_data_dict)
    if args['frames'] > 9:
        num_entries = 9
    else:
        num_entries = args['frames'] 

    if subplots:
        rows, cols = subplots
    else:
        rows = round(min(np.floor(10**0.5), 4))
        cols = round(min(np.floor(10**0.5), 4))

    #problem here is the value of rows and cols when run the code to desktop
    #the batch size is 1 so the rows and cols arent good numbers
    fig, axs = plt.subplots(rows, cols, squeeze=True)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if (idx < num_entries):
                # Only plot entries within the range of list
                if plot_annots:
                    out_image = draw_annots_on_image(list_data_dict[idx],
                                                     is_predict=is_predict,
                                                     mask = mask,
                                                     rendering=rendering,
                                                     intensity_maps=100,
                                                     remove_saturated=remove_saturated)
                    axs[i, j].imshow(out_image)
                    #img = Image.fromarray(out_image * 100)
                    #img = img.save("/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/image_{}.jpg".format(idx))
                else:
                    print('plot_annots=false')
                    axs[i, j].imshow(list_data_dict[0]['image'][idx])
            idx += 1
    if show:
        plt.show(block=False)

    if write:
        os.makedirs(os.path.dirname(write), exist_ok=True)
        fig.savefig(write, dpi=150, bbox_inches='tight')
    plt.close('all')
    return

def draw_annots_on_image(data_dict,
                         remove_saturated=True,
                         intensity_maps=100,
                         pupil_index=2,
                         iris_index=1,
                         is_predict=True,
                         rendering=False,
                         mask = False):

    image = data_dict['image']
    image = image - image.min()
    image = (255*(image/image.max())).astype(np.uint8)
    out_image = np.stack([image]*3, axis=2)

    assert len(image.shape) == 2, 'Image must be grayscale'
    height, width = image.shape

    if remove_saturated:
        loc_image_non_sat = image <= (255-intensity_maps)
    else:
        loc_image_non_sat = image <= 255

    if rendering == False:
        if is_predict or data_dict['pupil_center_available']:
            pupil_center = data_dict['pupil_center']
            [rr, cc] = draw.disk((pupil_center[1].clip(6, height-6),
                                pupil_center[0].clip(6, width-6)),
                                radius=5)
            out_image[rr, cc, :] = 255

        if is_predict or data_dict['pupil_ellipse_available']:
            pupil_ellipse = data_dict['pupil_ellipse']
            # print("Pupil: {}".format(type(pupil_ellipse)))
            loc_pupil = data_dict['mask'] == pupil_index
            out_image[..., 0] +=  (intensity_maps*loc_pupil*loc_image_non_sat).astype(np.uint8)
            out_image[..., 1] +=  (intensity_maps*loc_pupil*loc_image_non_sat).astype(np.uint8)

            if np.all(np.abs(pupil_ellipse[0:4]) > 5):
                [rr_p, cc_p] = draw.ellipse_perimeter(round(pupil_ellipse[1]),
                                                    round(pupil_ellipse[0]),
                                                    round(pupil_ellipse[3]),
                                                    round(pupil_ellipse[2]),
                                                    orientation=pupil_ellipse[4],
                                                    shape=image.shape)
                rr_p = rr_p.clip(6, image.shape[0]-6)
                cc_p = cc_p.clip(6, image.shape[1]-6)

                out_image[rr_p, cc_p, 0] = 255

        if is_predict or data_dict['iris_ellipse_available']:
            iris_ellipse = data_dict['iris_ellipse']
            # print("Iris: {}".format(type(iris_ellipse)))
            loc_iris = data_dict['mask'] == iris_index
            out_image[..., 1] +=  (intensity_maps*loc_iris*loc_image_non_sat).astype(np.uint8)

            if np.all(np.abs(iris_ellipse[0:4]) > 5):
                [rr_i, cc_i] = draw.ellipse_perimeter(round(iris_ellipse[1]),
                                                    round(iris_ellipse[0]),
                                                    round(iris_ellipse[3]),
                                                    round(iris_ellipse[2]),
                                                    orientation=iris_ellipse[4],
                                                    shape=image.shape)
                rr_i = rr_i.clip(6, image.shape[0]-6)
                cc_i = cc_i.clip(6, image.shape[1]-6)

                out_image[rr_i, cc_i, 2] = 255
    else: 
        if mask:
            loc_pupil = data_dict['mask'] == pupil_index
            out_image[..., 0] +=  (intensity_maps*loc_pupil*loc_image_non_sat).astype(np.uint8)
            out_image[..., 1] +=  (intensity_maps*loc_pupil*loc_image_non_sat).astype(np.uint8)

            loc_iris = data_dict['mask'] == iris_index
            out_image[..., 1] +=  (intensity_maps*loc_iris*loc_image_non_sat).astype(np.uint8)
        else:
            loc_gaze = data_dict['gaze_img'] == 1
            out_image[..., 1] =  (loc_image_non_sat*loc_gaze*loc_image_non_sat).astype(np.uint8)
            out_image[..., 2] =  (loc_image_non_sat*loc_gaze*loc_image_non_sat).astype(np.uint8)


    return out_image.astype(np.uint8)


def merge_two_dicts(dict_A, dict_B):
    '''

    Parameters
    ----------
    dict_A : DICT
        Regalar dictionary.
    dict_B : DICT
        Regular dictionary.

    Returns
    -------
    dict_C : DICT.

    '''
    dict_C = dict_A.copy()
    dict_C.update(dict_B)
    return dict_C


def get_ellipse_info(param, H, cond):
    '''
    Parameters
    ----------
    param : np.array
        Given ellipse parameters, return the following:
            a) Points along periphery
            b) Normalized ellipse parameters
    H: np.array 3x3
        Normalizing matrix which converts ellipse to normalized coordinates
    Returns
    -------
    normParam: Normalized Ellipse parameters
    elPts: Points along ellipse periphery
    '''

    if cond:
        norm_param = my_ellipse(param).transform(H)[0][:-1]  # We don't want the area
        elPts = my_ellipse(norm_param).generatePoints(50, 'equiAngle') # Regular points
        elPts = np.stack(elPts, axis=1)

        norm_param = fix_ellipse_axis_angle(norm_param)

    else:
        # Ellipse does not exist
        norm_param = -np.ones((5, ))
        elPts = -np.ones((8, 2))
    return elPts, norm_param


def fix_ellipse_axis_angle(ellipse):
    ellipse = copy.deepcopy(ellipse)
    if ellipse[3] > ellipse[2]:
        ellipse[[2, 3]] = ellipse[[3, 2]]
        ellipse[4] += np.pi/2

    if ellipse[4] > np.pi:
        ellipse[4] += -np.pi
    elif ellipse[4] < 0:
        ellipse[4] += np.pi

    return ellipse


# Plot segmentation output, pupil and iris ellipses
def plot_segmap_ellpreds(image,
                         seg_map,
                         pupil_ellipse,
                         iris_ellipse,
                         thres=50, plot_ellipses=True):
    loc_iris = seg_map == 1
    loc_pupil = seg_map == 2

    out_image = np.stack([image]*3, axis=2)

    loc_image_non_sat = image < (255-thres)

    # Add green to iris
    out_image[..., 1] = out_image[..., 1] + thres*loc_iris*loc_image_non_sat
    rr, cc = np.where(loc_iris & ~loc_image_non_sat)

    # For iris locations which are staurated, fix to green
    out_image[rr, cc, 0] = 0
    out_image[rr, cc, 1] = 255
    out_image[rr, cc, 2] = 0

    # Add yellow to pupil
    out_image[..., 0] = out_image[..., 0] + thres*loc_pupil*loc_image_non_sat
    out_image[..., 1] = out_image[..., 1] + thres*loc_pupil*loc_image_non_sat
    rr, cc = np.where(loc_pupil & ~loc_image_non_sat)

    # For pupil locations which are staurated, fix to yellow
    out_image[rr, cc, 0] = 255
    out_image[rr, cc, 1] = 255
    out_image[rr, cc, 2] = 0

    # Sketch iris ellipse
    if plot_ellipses:
        [rr_i, cc_i] = draw.ellipse_perimeter(round(iris_ellipse[1]),
                                              round(iris_ellipse[0]),
                                              round(iris_ellipse[3]),
                                              round(iris_ellipse[2]),
                                              orientation=iris_ellipse[4])

        # Sketch pupil ellipse
        [rr_p, cc_p] = draw.ellipse_perimeter(round(pupil_ellipse[1]),
                                              round(pupil_ellipse[0]),
                                              round(pupil_ellipse[3]),
                                              round(pupil_ellipse[2]),
                                              orientation=pupil_ellipse[4])

        # Clip the perimeter display incase it goes outside bounds
        rr_i = rr_i.clip(6, image.shape[0]-6)
        rr_p = rr_p.clip(6, image.shape[0]-6)
        cc_i = cc_i.clip(6, image.shape[1]-6)
        cc_p = cc_p.clip(6, image.shape[1]-6)

        out_image[rr_i, cc_i, ...] = np.array([0, 0, 255])
        out_image[rr_p, cc_p, ...] = np.array([255, 0, 0])

    return out_image.astype(np.uint8)


# Data extraction helpers
def generateEmptyStorage(name, subset):
    '''
    This file generates an empty dictionary with
    all relevant fields. This helps in maintaining
    consistency across all datasets.
    '''
    Data = {k: [] for k in ['Images',  # Gray image
                            'dataset',  # Dataset
                            'subset',  # Subset
                            'resolution',  # Image resolution
                            'archive',  # H5 file name
                            'Info',  # Path to original image
                            'Masks',  # Mask
                            'Masks_pupil_in_iris',
                            'Masks_noSkin',  # Mask with only iris and pupil
                            'subject_id',  # Subject ID if available
                            'Fits',  # Pupil and Iris fits
                            'pupil_loc',
                            'pupil_in_iris_loc',
                            #new fields for TEyeD dataset
                            'Eyeball',
                            'Gaze_vector',
                            'pupil_lm_2D',
                            'pupil_lm_3D',
                            'iris_lm_2D',
                            'iris_lm_3D',
                            'timestamp'
                            ]}
    Data['Fits'] = {k: [] for k in ['pupil', 'iris']}

    Key = {k: [] for k in ['dataset',  # Dataset
                           'subset',  # Subset
                           'resolution',  # Image resolution
                           'subject_id',  # Subject ID if available
                           'archive',  # H5 file name
                           'Info',  # Path to original image
                           'Fits',  # Pupil and Iris fits
                           'pupil_loc',
                           'pupil_in_iris_loc']}

    Key['Fits'] = {k: [] for k in ['pupil', 'iris']}

    Data['dataset'] = name
    Data['subset'] = subset

    Key['dataset'] = name
    Key['subset'] = subset

    return Data, Key


def plot_2D_hist(x, y, x_lims, y_lims, str_save='temp.jpg', axs=None):
    H, xedges, yedges = np.histogram2d(x, y,
                                       bins=64,
                                       range=[x_lims, y_lims],
                                       density=False)

    # Log scale for better visuals
    H = np.log(H + 1)

    # Normalize for display purposes
    H = H - H.min()
    H = H/H.max()

    if axs is None:
        fig, axs = plt.subplots()
        axs.imshow(H,
                   interpolation='lanczos',
                   origin='upper', extent=tuple(x_lims+y_lims))
        fig.savefig(str_save, dpi=600, transparent=True, bbox_inches='tight')
    else:
        axs.imshow(H,
                   interpolation='lanczos',
                   origin='upper', extent=tuple(x_lims+y_lims))


def measure_contrast(image, mask=None):

    size = (3, 3)  # window size i.e. here is 3x3 window

    num_patches = (image.shape[0] - size[0] + 1)*(image.shape[1] - size[1] + 1)
    patches = np.lib.stride_tricks.sliding_window_view(image, size)
    patches = patches.reshape(num_patches, 9)

    try:
        # Contrast range from (0 -> 128)
        contrast = patches.std(axis=1)
    except:
        import pdb; pdb.set_trace()

    hist = scipy.ndimage.histogram(contrast.flatten(), 0, 32, 64)/num_patches

    hist_lb = []
    if mask is not None:
        label_patches = mask[size[0]-2:-size[0]+2,
                             size[1]-2:-size[1]+2]
        label_patches = label_patches.reshape(num_patches, )

        temp = []
        for label in range(0, 3):
            loc = label_patches == label
            if np.sum(loc) >= 1:
                temp.append(scipy.ndimage.histogram(contrast[loc],
                                                    0, 64, 64)/np.sum(loc))
            else:
                temp.append(np.zeros(64, ))
        temp = np.stack(temp, axis=0)
        hist_lb.append(temp)

    return hist, np.stack(hist_lb, axis=0)


def image_contrast(image, scales=[1, ], by_category=None):

    contrast = []
    contrast_by_class = []

    for scale in scales:
        dsize = tuple(int(ele*scale) for ele in image.shape)
        data = cv2.resize(image, dsize, interpolation=cv2.INTER_LANCZOS4)

        if by_category is not None:
            mask = cv2.resize(by_category, dsize,
                              interpolation=cv2.INTER_NEAREST)
        else:
            None
        out = measure_contrast(data, mask)

        contrast.append(out[0])

        if by_category is not None:
            contrast_by_class.append(out[1])

    contrast = np.stack(contrast, axis=0)
    if by_category is not None:
        contrast_by_class = np.stack(contrast_by_class, axis=0)

    return contrast, contrast_by_class


def construct_mask_from_ellipse(ellipses, res):
    if len(ellipses.shape) == 1:
        ellipses = ellipses[np.newaxis, np.newaxis, :]
        B = 1
        F = 1
    else:
        B = ellipses.shape[0]
        F = ellipses.shape[1]

    mask = np.zeros((B, F) + res)
    for b in range(B):
        for frame in range(F):
            ellipse = ellipses[b, frame, ...].tolist()
            [rr, cc] = draw.ellipse(round(ellipse[1]),
                                    round(ellipse[0]),
                                    round(ellipse[3]),
                                    round(ellipse[2]),
                                    shape=res,
                                    rotation=-ellipse[4])
            rr = np.clip(rr, 0, res[0]-1)
            cc = np.clip(cc, 0, res[1]-1)
            mask[b, frame, rr, cc] = 1
    return mask.astype(bool)

def contruct_ellipse_from_mask(mask, pupil_c, iris_c):
    assert mask.max() <= 3, 'Highest  class label cannot exceed 3'
    assert mask.min() >= 0, 'Lowest class label cannot be below 0'
    pts_pupil, pts_iris = getValidPoints(mask, isPartSeg=False, legacy=False)
    
    irisFit = ElliFit(**{'data': pts_iris})
    pupilFit = ElliFit(**{'data': pts_pupil})
    
    iris_ellipse = irisFit.model[:5]
    # iris_ellipse[:2] = iris_c
    
    pupil_ellipse = pupilFit.model[:5]
    # pupil_ellipse[:2] = pupil_c
    
    return pupil_ellipse, iris_ellipse

def  generate_rend_masks(rend_dict, H, W, iterations):
    channel = 3
    rendering_mask = torch.zeros((iterations, channel, H, W))
    mask_gaze = torch.zeros((iterations, H, W))

    for frame in range(iterations):
        mask_pupil = np.zeros((H, W))
        mask_iris = np.zeros((H, W))
        gaze_line = np.zeros((H,W))

        #generate the segmenatation mask contains the pupil and iris
        pupil_perimeter_point = rend_dict['pupil_UV'][frame][rend_dict['edge_idx_pupil']['outline']].to(int).detach().cpu().numpy()
        iris_perimeter_point = rend_dict['iris_UV'][frame][rend_dict['edge_idx_iris']['outline']].to(int).detach().cpu().numpy()
            
        cv2.drawContours(mask_pupil, [pupil_perimeter_point], -1, 2, -1)
        cv2.drawContours(mask_iris, [iris_perimeter_point], -1, 1, -1)

        rendering_mask[frame,1] = torch.from_numpy(mask_iris)
        rendering_mask[frame,2] = torch.from_numpy(mask_pupil)

        #generate the gaze vector and the eyeball
        eyeball_circle_points = rend_dict['eyeball_circle'][frame].to(int).detach().cpu().numpy()
        center_pupil_gaze = rend_dict['pupil_UV'][frame][0].to(int).detach().cpu().numpy()
        center_eyeball_gaze = rend_dict['eyeball_c_UV'][frame].to(int).detach().cpu().numpy()
        
        cv2.drawContours(gaze_line, [eyeball_circle_points], -1, 1, 3)
        cv2.line(gaze_line, (center_eyeball_gaze[0], center_eyeball_gaze[1]), 
                (center_pupil_gaze[0], center_pupil_gaze[1]),
                    1 , 3)
        
        mask_gaze[frame] = torch.from_numpy(gaze_line)

    rend_dict['predict'] = rendering_mask
    rend_dict['mask_gaze'] = mask_gaze

    return rend_dict

def assert_torch_invalid(X, string):
    assert not torch.isnan(X).any(), print('NaN problem ['+string+']')
    assert not torch.isinf(X).any(), print('inf problem ['+string+']')
    assert torch.all(torch.isfinite(X)), print('Some elements not finite problem ['+string+']')
    assert X.numel() > 0, print('Empty tensor problem ['+string+']')



