#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

import re
import os
import cv2
import h5py
import copy
import torch
import pickle
import random

import numpy as np
import scipy.io as scio

import matplotlib.pyplot as plt

from helperfunctions.data_augment import augment, flip
from torch.utils.data import Dataset

from helperfunctions.helperfunctions import simple_string, one_hot2dist
from helperfunctions.helperfunctions import pad_to_shape, get_ellipse_info
from helperfunctions.helperfunctions import extract_datasets, scale_by_ratio
from helperfunctions.helperfunctions import fix_ellipse_axis_angle, dummy_data

from Visualitation_TEyeD.gaze_estimation import generate_gaze_gt 

from helperfunctions.utils import normPts

from sklearn.model_selection import StratifiedKFold, train_test_split

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Deactive file locking


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class DataLoader_riteyes(Dataset):
    def __init__(self,
                 dataDiv_Obj,
                 path2data,
                 cond,
                 augFlag=False,
                 size=(480, 640),
                 fold_num=0,
                 num_frames=4,
                 sort='random',
                 args=None,
                 scale=False):

        self.mode = cond
        self.init_frames = num_frames

        self.ellseg = args['net_ellseg_head']
        self.train_with_mask = args['loss_w_rend_pred_2_gt_edge'] \
                                or args['loss_w_rend_gt_2_pred'] \
                                or args['loss_w_rend_pred_2_gt'] \
                                or args['net_ellseg_head']

        cond = 'train_idx' if 'train' in cond else cond
        cond = 'valid_idx' if 'valid' in cond else cond
        cond = 'test_idx' if 'test' in cond else cond
        
        # Operational variables
        self.arch = dataDiv_Obj.arch  # Available archives
        self.size = size  # Expected size of images
        self.scale = scale
        self.imList = dataDiv_Obj.folds[fold_num][cond]  # Image list
        self.augFlag = augFlag  # Augmentation flag
        self.equi_var = True  # Default is always True
        self.path2data = path2data  # Path to expected H5 files

        #  You can specify which augs you want as input to augment
        # TODO [0, 1, 8, 9, 11]
        self.augger = augment(choice_list=[0, 1, 11], mask_available=self.train_with_mask) if augFlag else []
        self.flipper = flip()

        # Get dataset index by archive ID
        ds_present, ds_index = extract_datasets(self.arch[self.imList[:, 1]])
        self.imList = np.hstack([self.imList, ds_index[:, np.newaxis]])
        self.fileObjs = {}

        avail_ds, counts = np.unique(ds_index, return_counts=True)

        # Repeat poorly represented datasets such that equal number of images
        # exist per dataset
        # TODO MODIFY TO SUPPORT MORE THAN ONE DATASET
        # TODO In case of more than one dataset we should have
        # the same number of images per dataset
        """if len(counts) > 1:
            extra_samples = []
            for ii, ds_itr in enumerate(avail_ds.tolist()):
                num_more_images_needed = max(counts) - counts[ii]
                if num_more_images_needed > 0:
                    loc = np.where(self.imList[:, -1] == ds_itr)[0]
                    extra_loc = np.random.choice(loc,
                                                 size=num_more_images_needed)
                    extra_samples.append(self.imList[extra_loc, :])

            extra_samples = np.concatenate(extra_samples, axis=0)
            self.imList = np.concatenate([self.imList, extra_samples])
            len_cond = self.imList.shape[0] == len(avail_ds)*max(counts)
            assert len_cond, 'Samples must equal N X the max samples present' """

        #delete rows in order to create batches of ten images
        #so the number of images in the dataset should be /10=0
        del_rows = self.imList.shape[0]%self.init_frames
        if del_rows!=0:
            self.imList = self.imList[:-del_rows]
        self.imList = np.reshape(self.imList, (-1,self.init_frames,3))

        for i in range(self.imList.shape[0]):
            if i >= self.imList.shape[0]:
                continue
            if not all(element == self.imList[i, 0, 1] for element in self.imList[i, ..., 1]):
                self.imList = np.delete(self.imList, i, axis=0)

        self.sort(sort) # Sort order of images

        if (cond == 'train_idx'):
            num_of_entries  = int(self.imList.shape[0]*args['train_data_percentage'])
            self.imList = self.imList[:num_of_entries]

        # print(f'Split: {self.mode}')
        # print(f'Num: {self.imList.shape[0]*self.imList.shape[1]}')
        # print(f'Train perc.: {args["train_data_percentage"]*100.0}%')

    def sort(self, sort, batch_size=None):

        if sort=='ordered':
            # Completely ordered
            loc = np.unique(self.imList,
                            return_counts=True,
                            axis=0)
            #print('Warning. Non-unique file list.') if np.any(loc[1]!=1) else print('Sorted list')
            self.imList = loc[0]
        elif sort == 'nothing':
            pass

        elif sort=='semiordered':
            # Randomize first, then sort by archNum
            self.sort(sort='random')
            loc = np.argsort(self.imList[:, 1, 1])
            self.imList = self.imList[loc, :]

        elif sort=='random':
            # Completely random selection. DEFAULT.
            loc = np.random.permutation(self.imList.shape[0])
            self.imList = self.imList[loc, :]

        elif sort=='mutliset_random':
            # Randomize first, then rearrange by BS / num_sets images per set.
            # This ensures that equal number of images from each dataset are
            # read in per batch read.
            self.sort('random')
            avail_ds, counts = np.unique(self.imList[:, 1, 2],
                                         return_counts=True)
            temp_imList = []
            for ds_itr in np.nditer(avail_ds):
                loc = self.imList[:, 1, 2] == ds_itr
                temp_imList.append(self.imList[loc, :])
            temp_imList = np.stack(temp_imList, axis=1).reshape(-1, 10, 3)
            assert temp_imList.shape == self.imList.shape, 'Incorrect reshaping'
            self.imList = temp_imList

        elif sort=='one_by_one_ds':
            # Randomize first, then rearrange such that each BS contains image
            # from a single dataset
            self.sort('random')
            avail_ds, counts = np.unique(self.imList[:, 1, 2],
                                         return_counts=True)

            # Create a list of information for each individual dataset
            # present within the selection
            temp_imList = []
            for ds_itr in np.nditer(avail_ds):
                loc = self.imList[:, 1, 2] == ds_itr
                temp_imList.append(self.imList[loc, :])

            cond = True
            counter = 0

            imList = [] # Blank initialization
            while cond:
                counter+=1
                # Keep extracting batch_size elements from each entry
                ds_order = random.sample(range(avail_ds.max()),
                                         avail_ds.max())

                for i in range(avail_ds.max()):
                    idx = ds_order[i] if ds_order else 0
                    start = (counter-1)*batch_size
                    stop = counter*batch_size

                    if stop < temp_imList[idx].shape[0]:
                        imList.append(temp_imList[idx][start:stop, ...])
                    else:
                        # A particular dataset has been completely sampled
                        counter = 0
                        cond = False # Break out of main loop
                        break # Break out of inner loop
            self.imList = np.concatenate(imList, axis=0)

        else:
            import sys
            sys.exit('Incorrect sorting options')

    def __len__(self):
        return self.imList.shape[0]

    def __del__(self, ):
        for entry, h5_file_obj in self.fileObjs.items():
            h5_file_obj.close()

    def __getitem__(self, idx):
        '''
        Reads images and all the required sources of information.
        Return a dictionary contains all the information.
        The image and the mask is a volume containing N gray scale images
        '''
        try:
            numClasses = 3
            data_dict = self.readEntry_new(idx)
            #data_dict = pad_to_shape(data_dict, to_size=(240, 320))            

            if self.scale:
                data_dict = scale_by_ratio(data_dict, self.scale, self.train_with_mask)

            data_dict = self.augger(data_dict) if self.augFlag else data_dict

            assert data_dict['image'].max() <= 255, 'Max luma should be <=255'
            assert data_dict['image'].min() >= 0, 'Min luma should be >=0'
            assert data_dict['image'].shape[2] == 320 and data_dict['image'].shape[1] == 240, 'previous functionally is not implemented, different dimensions than 320*240'

            if (np.random.rand(1) > 0.8) and self.augFlag:
                # Always keep flipping the image with 0.5 prob when augmenting
                # the datasets.
                data_dict = self.flipper(data_dict, self.train_with_mask)
            
        except Exception:
            print('Error reading and processing data!')
            data_dict = self.readEntry_new(idx)
            if (data_dict['image'].shape[2] != 320 and data_dict['image'].shape[1] != 240):
                print('previous functionally is not implemented, different dimensions than 320*240')
            im_num = self.imList[idx, :, 0]
            arch_num = self.imList[idx, :, 1]
            archStr = self.arch[arch_num[0]]
            print('Bad sampled number: {}'.format(im_num))
            print('Bad archive number: {}'.format(arch_num))
            print('Bad archive name: {}'.format(archStr))
            data_dict = dummy_data(shape=(self.init_frames, 480//2, 640//2))

        num_of_frames = self.init_frames
        height = data_dict['image'].shape[1]
        width = data_dict['image'].shape[2]

        for i in range(num_of_frames):
            data_dict['pupil_ellipse'][i] = fix_ellipse_axis_angle(data_dict['pupil_ellipse'][i])
            data_dict['iris_ellipse'][i] = fix_ellipse_axis_angle(data_dict['iris_ellipse'][i])

        if self.train_with_mask:
            spatial_weights_list = []
            distance_map_list = []

             # Modify labels by removing Sclera class
            if np.all(data_dict['mask_available']):
                data_dict['mask'][data_dict['mask'] == 1] = 0  # Move Sclera to 0
                data_dict['mask'][data_dict['mask'] == 2] = 1  # Move Iris to 1
                data_dict['mask'][data_dict['mask'] == 3] = 2  # Move Pupil to 2


            if self.ellseg:
                for i in range(num_of_frames):
                    if data_dict['mask_available'][i]:
                        # Find distance map for each class for surface loss
                        # Compute edge weight maps
                        spatial_weights = cv2.Canny(data_dict['mask'][i].astype(np.uint8), 0, 1)/255
                        spatial_weights_list.append(1 + cv2.dilate(spatial_weights, (3, 3),
                                                            iterations=1)*20)

                        # Calculate distance_maps for only Iris and Pupil.
                        # Pupil: 2. Iris: 1. Rest: 0.
                        distance_map = np.zeros(((3, ) + data_dict['image'][i].shape))
                        for k in range(0, numClasses):
                            distance_map[k, ...] = one_hot2dist(data_dict['mask'][i].astype(np.uint8)==k)
                        distance_map_list.append(distance_map)
                    else:
                        distance_map_list.append(np.zeros(((3, ) + data_dict['image'][i].shape)))
                        spatial_weights_list.append(np.zeros_like(data_dict['mask'][i]))

                data_dict['distance_map'] = np.stack(distance_map_list, axis=0)
                data_dict['spatial_weights'] = np.stack(spatial_weights_list, axis=0)

        #if the data_dict was created using the dummy_data generator skip this step since the std is zero
        #use the image number(id) to check when the dummy_data generator used
        if (np.all(data_dict['im_num']>=0)):
            pic = data_dict['image']
            data_to_torch = (pic - pic.mean())/pic.std()

        if np.any(np.isinf(data_dict['image'])) or np.any(np.isnan(data_dict['image'])):
            data_to_torch = np.zeros_like(data_dict['image']).astype(np.uint8)
            data_dict['is_bad'][i] = np.stack([True] * num_of_frames, axis=0)

        # Groundtruth annotation mask to torch long
        if self.train_with_mask:
            data_dict['mask'] = MaskToTensor()(data_dict['mask']).to(torch.long)

        data_dict['image'] = data_to_torch
        # Generate normalized pupil and iris information
        if self.equi_var:
            sc = max([width, height])
            H = np.array([[2/sc, 0, -1], [0, 2/sc, -1], [0, 0, 1]])
        else:
            H = np.array([[2/width, 0, -1], [0, 2/height, -1], [0, 0, 1]])


        iris_ellipse_norm_list = []
        pupil_ellipse_norm_list = []
        pupil_center_norm_list = []
        for i in range(num_of_frames):
            if not data_dict['is_bad'][i]:
                iris_ellipse_norm_list.append(get_ellipse_info(data_dict['iris_ellipse'][i], H,
                                                                data_dict['iris_ellipse_available'][i])[1])
                pupil_ellipse_norm_list.append(get_ellipse_info(data_dict['pupil_ellipse'][i], H,
                                                                data_dict['pupil_ellipse_available'][i])[1])

                # Generate normalized pupil center location
                pupil_center_norm_list.append(normPts(data_dict['pupil_center'][i],
                                                        np.array([width, height]),
                                                        by_max=self.equi_var))
            else:
                iris_ellipse_norm_list.append(-1*np.ones((5, )))
                pupil_center_norm_list.append(-1*np.ones((2, )))
                pupil_ellipse_norm_list.append(-1*np.ones((5, )))

        data_dict['iris_ellipse_norm'] = np.stack(iris_ellipse_norm_list, axis=0)
        data_dict['pupil_ellipse_norm'] = np.stack(pupil_ellipse_norm_list, axis=0)
        data_dict['pupil_center_norm'] = np.stack(pupil_center_norm_list, axis=0)

        return data_dict


    #read entry works. Create the 3D volume for input to the nn
    def readEntry(self, idx):
        '''
        Read a number of sequential images and all their groundtruths using partial loading
        Mask annotations. This is followed by OpenEDS definitions:
            0 -> Background
            1 -> Sclera (if available)
            2 -> Iris
            3 -> Pupil
        '''
        im_num = self.imList[idx, :, 0]
        set_num = self.imList[idx, :, 2]
        arch_num = self.imList[idx, :, 1]

        archStr = self.arch[arch_num[0]]
        archName = archStr.split(':')[0]

        # Use H5 files already open for data I/O. This enables catching.
        if archName not in self.fileObjs.keys():
            self.fileObjs[archName] = h5py.File(os.path.join(self.path2data,
                                                             str(archName)+'.h5'),
                                                'r', swmr=True)
        f = self.fileObjs[archName]

        num_of_frames = self.init_frames

        #define the variables
        is_bad_list = []
        image_list = []
        pupil_center_list = []
        pupil_center_available_list = []
        mask_noSkin_list = []
        mask_available_list = []
        pupil_param_list = []
        pupil_ellipse_available_list = []
        iris_param_list = []
        iris_ellipse_available_list = []
        eyeball_list = []
        eyeball_available_list = []
        gaze_vector_list = []
        gaze_vector_available_list = []
        pupil_lm_2D_list = []
        pupil_lm_2D_available_list = []
        pupil_lm_3D_list = []
        pupil_lm_3D_available_list = []
        iris_lm_2D_list = []
        iris_lm_2D_available_list = []
        iris_lm_3D_list = []
        iris_lm_3D_available_list = []

        for i, image_id in enumerate(im_num):

            more_features = False

            # Read information
            temp_image = f['Images'][image_id, ...]
            image_list.append(temp_image)

            # Get pupil center
            if f['pupil_loc'].__len__() != 0:
                temp_pupil_center = f['pupil_loc'][image_id, ...]
                temp_pupil_center_available = True
            else:
                temp_pupil_center_available = False
                temp_pupil_center = -np.ones(2, )
            pupil_center_list.append(temp_pupil_center)
            pupil_center_available_list.append(temp_pupil_center_available)

            # Get mask without skin
            if self.train_with_mask:
                if f['Masks_noSkin'].__len__() != 0:
                    temp_mask_noSkin = f['Masks_noSkin'][image_id, ...]
                    temp_mask_available = True
                    any_pupil = np.any(temp_mask_noSkin == 3)
                    any_iris = np.any(temp_mask_noSkin == 2)
                    if not (any_pupil and any_iris):
                        # atleast one pixel must belong to all classes
                        temp_mask_noSkin = -np.ones(temp_image.shape[:2])
                        temp_mask_available = False
                else:
                    temp_mask_noSkin = -np.ones(temp_image.shape[:2])
                    temp_mask_available = False
                mask_noSkin_list.append(temp_mask_noSkin)
                mask_available_list.append(temp_mask_available)

            # Pupil ellipse parameters
            if f['Fits']['pupil'].__len__() != 0:
                temp_pupil_ellipse_available = True
                temp_pupil_param = f['Fits']['pupil'][image_id, ...]
            else:
                temp_pupil_ellipse_available = False
                temp_pupil_param = -np.ones(5, )
            pupil_param_list.append(temp_pupil_param)
            pupil_ellipse_available_list.append(temp_pupil_ellipse_available)

            # Iris ellipse parameters
            if f['Fits']['iris'].__len__() != 0:
                temp_iris_ellipse_available = True
                temp_iris_param = f['Fits']['iris'][image_id, ...]
            else:
                temp_iris_ellipse_available = False
                temp_iris_param = -np.ones(5, )
            iris_param_list.append(temp_iris_param)
            iris_ellipse_available_list.append(temp_iris_ellipse_available)

            if 'Dikablis' in archName:
                more_features = True

            if more_features:
                if f['Eyeball'].__len__() != 0:
                    eyeball_available = True
                    temp_eyeball = f['Eyeball'][image_id, ...]
            else:
                eyeball_available = False
                temp_eyeball = -np.ones(4, )
            eyeball_list.append(temp_eyeball)
            eyeball_available_list.append(eyeball_available)

            if more_features:
                if f['Gaze_vector'].__len__() != 0:
                    gaze_vector_available = True
                    temp_gaze_vector = f['Gaze_vector'][image_id, ...]
            else:
                gaze_vector_available = False
                temp_gaze_vector = -np.ones(3, )
            gaze_vector_available_list.append(gaze_vector_available)
            gaze_vector_list.append(temp_gaze_vector)

            if more_features:
                if f['pupil_lm_2D'].__len__() != 0:
                    pupil_lm_2D_available = True
                    temp_pupil_lm_2D = f['pupil_lm_2D'][image_id, ...]
            else:
                pupil_lm_2D_available = False
                temp_pupil_lm_2D = -np.ones(17, )
            pupil_lm_2D_available_list.append(pupil_lm_2D_available)
            pupil_lm_2D_list.append(temp_pupil_lm_2D)

            if more_features:
                if f['pupil_lm_3D'].__len__() != 0:
                    pupil_lm_3D_available = True
                    temp_pupil_lm_3D = f['pupil_lm_3D'][image_id, ...]
            else:
                pupil_lm_3D_available = False
                temp_pupil_lm_3D = -np.ones(25, )
            pupil_lm_3D_available_list.append(pupil_lm_3D_available)
            pupil_lm_3D_list.append(temp_pupil_lm_3D)

            if more_features:
                if f['iris_lm_2D'].__len__() != 0:
                    iris_lm_2D_available = True
                    temp_iris_lm_2D = f['iris_lm_2D'][image_id, ...]
            else:
                iris_lm_2D_available = False
                temp_iris_lm_2D = -np.ones(17, )
            iris_lm_2D_available_list.append(iris_lm_2D_available)
            iris_lm_2D_list.append(temp_iris_lm_2D)

            if more_features:
                if f['iris_lm_3D'].__len__() != 0:
                    iris_lm_3D_available = True
                    temp_iris_lm_3D = f['iris_lm_3D'][image_id, ...]
            else:
                iris_lm_3D_available = False
                temp_iris_lm_3D = -np.ones(25, )
            iris_lm_3D_available_list.append(iris_lm_3D_available)
            iris_lm_3D_list.append(temp_iris_lm_3D)

        image = np.stack(image_list, axis=0)
        pupil_center = np.stack(pupil_center_list, axis=0)
        if self.train_with_mask:
            mask_noSkin = np.stack(mask_noSkin_list, axis=0)
        pupil_param = np.stack(pupil_param_list, axis=0)
        iris_param = np.stack(iris_param_list, axis=0)
        gaze_vector = np.stack(gaze_vector_list, axis=0)
        eyeball = np.stack(eyeball_list, axis=0)
        pupil_lm_2D = np.stack(pupil_lm_2D_list, axis=0)
        pupil_lm_3D = np.stack(pupil_lm_3D_list, axis=0)
        iris_lm_2D = np.stack(iris_lm_2D_list, axis=0)
        iris_lm_3D = np.stack(iris_lm_3D_list, axis=0)

        pupil_ellipse_available = pupil_ellipse_available_list
        pupil_center_available = pupil_center_available_list
        iris_ellipse_available = iris_ellipse_available_list
        gaze_vector_available = gaze_vector_available_list
        eyeball_available = eyeball_available_list
        pupil_lm_2D_available = pupil_lm_2D_available_list
        pupil_lm_2D_available = pupil_lm_3D_available_list
        iris_lm_2D_available = iris_lm_2D_available_list
        iris_lm_2D_available = iris_lm_3D_available_list

        data_dict = {}
        if self.train_with_mask:
            data_dict['mask'] = mask_noSkin
        data_dict['image'] = image
        data_dict['ds_num'] = set_num
        data_dict['pupil_center'] = pupil_center.astype(np.float32)
        data_dict['iris_ellipse'] = iris_param.astype(np.float32)
        data_dict['pupil_ellipse'] = pupil_param.astype(np.float32)
        data_dict['gaze_vector'] = gaze_vector.astype(np.float32)
        data_dict['eyeball'] = eyeball.astype(np.float32)
        data_dict['pupil_lm_2D'] = pupil_lm_2D.astype(np.float32)
        data_dict['pupil_lm_3D'] = pupil_lm_3D.astype(np.float32)
        data_dict['iris_lm_2D'] = iris_lm_2D.astype(np.float32)
        data_dict['iris_lm_3D'] = iris_lm_3D.astype(np.float32)

        is_bad_list = [False] * num_of_frames

        # Extra check to not return bad batches
        if self.train_with_mask:
            if (np.any(data_dict['mask']<0) or np.any(data_dict['mask']>3)) and mask_available_list:
                # This is a basic sanity check and should never be triggered
                # unless a freak accident caused something to change
                is_bad_list = [True] * num_of_frames

        # Ability to traceback
        data_dict['im_num'] = im_num
        data_dict['archName'] = archName

        # Keep flags as separate entries
        if self.train_with_mask: data_dict['mask_available'] = np.stack(mask_available_list, axis=0)
        pupil_center_available_list = pupil_center_available \
            if not np.all(pupil_center == -1) else False
        iris_ellipse_available_list = iris_ellipse_available\
            if not np.all(iris_param == -1) else False
        pupil_ellipse_available_list = pupil_ellipse_available\
            if not np.all(pupil_param == -1) else False

        data_dict['is_bad'] = np.stack(is_bad_list, axis=0)
        data_dict['pupil_center_available'] = np.stack(pupil_center_available_list, axis=0)
        data_dict['iris_ellipse_available'] = np.stack(iris_ellipse_available_list, axis=0)
        data_dict['pupil_ellipse_available'] = np.stack(pupil_ellipse_available_list, axis=0)
        data_dict['gaze_vector_available'] = np.stack(gaze_vector_available_list, axis=0)
        data_dict['eyeball_available'] = np.stack(eyeball_available_list, axis=0)
        data_dict['pupil_lm_2D_available'] = np.stack(pupil_lm_2D_available_list, axis=0)
        data_dict['pupil_lm_3D_available'] = np.stack(pupil_lm_3D_available_list, axis=0)
        data_dict['iris_lm_2D_available'] = np.stack(iris_lm_2D_available_list, axis=0)
        data_dict['iris_lm_3D_available'] = np.stack(iris_lm_3D_available_list, axis=0)
        
        return data_dict

#read entry works. Create the 3D volume for input to the nn
    def readEntry_new(self, idx):
        '''
        Read a number of sequential images and all their groundtruths using partial loading
        Mask annotations. This is followed by OpenEDS definitions:
            0 -> Background
            1 -> Sclera (if available)
            2 -> Iris
            3 -> Pupil
        '''
        im_num = self.imList[idx, ..., 0]
        set_num = self.imList[idx, ..., 2]
        arch_num = self.imList[idx, ..., 1]

        archStr = self.arch[arch_num[0]]
        archName = archStr.split(':')[0]

        # Use H5 files already open for data I/O. This enables catching.
        if archName not in self.fileObjs.keys():
            self.fileObjs[archName] = h5py.File(os.path.join(self.path2data,
                                                             str(archName)+'.h5'),
                                                'r', swmr=True)
        f = self.fileObjs[archName]

        more_features = False

        num_of_frames = self.init_frames

        # Read information
        image = f['Images'][im_num, ...]

        # Get pupil center
        if f['pupil_loc'].__len__() != 0:
            pupil_center = f['pupil_loc'][im_num, ...]
            pupil_center_available = [True] * num_of_frames
        else:
            pupil_center_available = [False] * num_of_frames
            pupil_center = -np.ones(num_of_frames, 2, )

        # Get mask without skin
        if self.train_with_mask:
            if f['Masks_noSkin'].__len__() != 0:
                mask_noSkin = f['Masks_noSkin'][im_num, ...]
                mask_available = [True] * num_of_frames
                any_pupil = np.any(mask_noSkin == 3)
                any_iris = np.any(mask_noSkin == 2)
                if not (any_pupil and any_iris):
                    # atleast one pixel must belong to all classes
                    mask_noSkin = -np.ones(image.shape[:2])
                    mask_available = [False] * num_of_frames
            else:
                mask_noSkin = -np.ones(image.shape[:2])
                mask_available = [False] * num_of_frames

        # Pupil ellipse parameters
        if f['Fits']['pupil'].__len__() != 0:
            pupil_ellipse_available = [True] * num_of_frames
            pupil_param = f['Fits']['pupil'][im_num, ...]
        else:
            pupil_ellipse_available = [False] * num_of_frames
            pupil_param = -np.ones(num_of_frames, 5, )

        # Iris ellipse parameters
        if f['Fits']['iris'].__len__() != 0:
            iris_ellipse_available = [True] * num_of_frames
            iris_param = f['Fits']['iris'][im_num, ...]
        else:
            iris_ellipse_available = [False] * num_of_frames
            iris_param = -np.ones(num_of_frames, 5, )

        if 'Dikablis' in archName:
            more_features = True

        if more_features:
            if f['Eyeball'].__len__() != 0:
                eyeball_available = [True] * num_of_frames
                eyeball = f['Eyeball'][im_num, ...]
        else:
            eyeball_available = [False] * num_of_frames
            eyeball = -np.ones(num_of_frames, 4, )

        if more_features:
            if f['Gaze_vector'].__len__() != 0:
                gaze_vector_available = [True] * num_of_frames
                gaze_vector = f['Gaze_vector'][im_num, ...]
        else:
            gaze_vector_available = [False] * num_of_frames
            gaze_vector = -np.ones(num_of_frames, 3, )

        if more_features:
            if f['pupil_lm_2D'].__len__() != 0:
                pupil_lm_2D_available = [True] * num_of_frames
                pupil_lm_2D = f['pupil_lm_2D'][im_num, ...]
        else:
            pupil_lm_2D_available = [False] * num_of_frames
            pupil_lm_2D = -np.ones(num_of_frames, 17, )

        if more_features:
            if f['pupil_lm_3D'].__len__() != 0:
                pupil_lm_3D_available = [True] * num_of_frames
                pupil_lm_3D = f['pupil_lm_3D'][im_num, ...]
        else:
            pupil_lm_3D_available = [False] * num_of_frames
            pupil_lm_3D = -np.ones(num_of_frames, 25, )

        if more_features:
            if f['iris_lm_2D'].__len__() != 0:
                iris_lm_2D_available = [True] * num_of_frames
                iris_lm_2D = f['iris_lm_2D'][im_num, ...]
        else:
            iris_lm_2D_available = [False] * num_of_frames
            iris_lm_2D = -np.ones(num_of_frames, 17, )

        if more_features:
            if f['iris_lm_3D'].__len__() != 0:
                iris_lm_3D_available = [True] * num_of_frames
                iris_lm_3D = f['iris_lm_3D'][im_num, ...]
        else:
            iris_lm_3D_available = [False] * num_of_frames
            iris_lm_3D = -np.ones(num_of_frames, 25, )

        data_dict = {}
        if self.train_with_mask:
            data_dict['mask'] = mask_noSkin
        data_dict['image'] = image
        data_dict['ds_num'] = set_num
        data_dict['pupil_center'] = pupil_center.astype(np.float32)
        data_dict['iris_ellipse'] = iris_param.astype(np.float32)
        data_dict['pupil_ellipse'] = pupil_param.astype(np.float32)
        data_dict['gaze_vector'] = gaze_vector.astype(np.float32)
        data_dict['eyeball'] = eyeball.astype(np.float32)
        data_dict['pupil_lm_2D'] = pupil_lm_2D.astype(np.float32)
        data_dict['pupil_lm_3D'] = pupil_lm_3D.astype(np.float32)
        data_dict['iris_lm_2D'] = iris_lm_2D.astype(np.float32)
        data_dict['iris_lm_3D'] = iris_lm_3D.astype(np.float32)

        is_bad_list = [False] * num_of_frames

        # Extra check to not return bad batches
        if self.train_with_mask:
            if (np.any(data_dict['mask']<0) or np.any(data_dict['mask']>3)) and not np.all(mask_available):
                # This is a basic sanity check and should never be triggered
                # unless a freak accident caused something to change
                is_bad_list = [True] * num_of_frames

        # Ability to traceback
        data_dict['im_num'] = im_num
        data_dict['archName'] = archName

        # Keep flags as separate entries
        if self.train_with_mask: data_dict['mask_available'] = np.stack(mask_available, axis=0)
        data_dict['pupil_center_available'] = np.stack(pupil_center_available, axis=0) \
            if not np.all(pupil_center == -1) else np.stack([False] * num_of_frames, axis=0)
        data_dict['iris_ellipse_available'] = np.stack(iris_ellipse_available, axis=0)\
            if not np.all(iris_param == -1) else np.stack([False] * num_of_frames, axis=0)
        data_dict['pupil_ellipse_available'] = np.stack(pupil_ellipse_available, axis=0)\
            if not np.all(pupil_param == -1) else np.stack([False] * num_of_frames, axis=0)

        data_dict['is_bad'] = np.stack(is_bad_list, axis=0)
        data_dict['gaze_vector_available'] = np.stack(gaze_vector_available, axis=0)
        data_dict['eyeball_available'] = np.stack(eyeball_available, axis=0)
        data_dict['pupil_lm_2D_available'] = np.stack(pupil_lm_2D_available, axis=0)
        data_dict['pupil_lm_3D_available'] = np.stack(pupil_lm_3D_available, axis=0)
        data_dict['iris_lm_2D_available'] = np.stack(iris_lm_2D_available, axis=0)
        data_dict['iris_lm_3D_available'] = np.stack(iris_lm_3D_available, axis=0)
        
        return data_dict

def listDatasets(AllDS):
    dataset_list = np.unique(AllDS['dataset'])
    subset_list = np.unique(AllDS['subset'])
    return (dataset_list, subset_list)


def readArchives(path2arc_keys):
    D = os.listdir(path2arc_keys)
    AllDS = {'archive': [], 'dataset': [], 'subset': [], 'subject_id': [],
             'im_num': [], 'pupil_loc': [], 'iris_loc': []}

    for chunk in D:
        # Load archive key
        chunkData = scio.loadmat(os.path.join(path2arc_keys, chunk))
        N = np.size(chunkData['archive'])
        pupil_loc = chunkData['pupil_loc']
        subject_id = chunkData['subject_id']

        if not chunkData['subset']:
            print('{} does not have subsets.'.format(chunkData['dataset']))
            chunkData['subset'] = 'none'

        if type(pupil_loc) is list:
            # Replace pupil locations with -1
            print('{} does not have pupil center locations'.format(chunkData['dataset']))
            pupil_loc = -1*np.ones((N, 2))

        if chunkData['Fits']['iris'][0, 0].size == 0:
            # Replace iris locations with -1
            print('{} does not have iris center locations'.format(chunkData['dataset']))
            iris_loc = -1*np.ones((N, 2))
        else:
            iris_loc = chunkData['Fits']['iris'][0, 0][:, :2]

        loc = np.arange(0, N)
        res = np.flip(chunkData['resolution'], axis=1)  # Flip the resolution to [W, H]

        AllDS['im_num'].append(loc)
        AllDS['subset'].append(np.repeat(chunkData['subset'], N))
        AllDS['dataset'].append(np.repeat(chunkData['dataset'], N))
        AllDS['archive'].append(chunkData['archive'].reshape(-1)[loc])
        AllDS['iris_loc'].append(iris_loc[loc, :]/res[loc, :])
        AllDS['pupil_loc'].append(pupil_loc[loc, :]/res[loc, :])
        AllDS['subject_id'].append(subject_id)

    # Concat all entries into one giant list
    for key, val in AllDS.items():
        AllDS[key] = np.concatenate(val, axis=0)
    return AllDS


def rmDataset(AllDS, rmSet):
    '''
    Remove datasets.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [True if simple_string(ele) in simple_string(rmSet)
           else False for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['dataset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData


def rmSubset(AllDS, rmSet):
    '''
    Remove subsets.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [True if simple_string(ele) in simple_string(rmSet)
           else False for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['subset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData


def selDataset(AllDS, selSet):
    '''
    Select datasets of interest.
    '''
    dsData = copy.deepcopy(AllDS)
    dataset_list = listDatasets(dsData)[0]
    loc = [False if simple_string(ele) in simple_string(selSet)
           else True for ele in dataset_list]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['dataset'] == dataset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))
    return dsData


def selSubset(AllDS, selSubset):
    '''
    Select subsets of interest.
    '''
    dsData = copy.deepcopy(AllDS)
    subset_list = listDatasets(dsData)[1]
    subset_list_temp = copy.deepcopy(subset_list)
    
    for idx, element in enumerate(subset_list):
        if 'Dikablis' in element:
            split = element.split('_')
            subset_list_temp[idx] = split[0] + '_' + split[1]

    loc = [False if simple_string(ele) in simple_string(selSubset)
           else True for ele in subset_list_temp]
    rmIdx = np.where(loc)[0]
    for i in rmIdx:
        loc = dsData['subset'] == subset_list[i]
        dsData = copy.deepcopy(rmEntries(dsData, loc))

    return dsData


def rmEntries(AllDS, ent):
    dsData = copy.deepcopy(AllDS)
    dsData['subject_id'] = AllDS['subject_id'][~ent, ]
    dsData['pupil_loc'] = AllDS['pupil_loc'][~ent, :]
    dsData['iris_loc'] = AllDS['iris_loc'][~ent, :]
    dsData['archive'] = AllDS['archive'][~ent, ]
    dsData['dataset'] = AllDS['dataset'][~ent, ]
    dsData['im_num'] = AllDS['im_num'][~ent, ]
    dsData['subset'] = AllDS['subset'][~ent, ]
    return dsData


def generate_strat_indices(AllDS):
    '''
    Removing images with pupil center values which are 10% near borders.
    Does not remove images with a negative pupil center.
    Returns the indices and a pruned data record.
    '''
    loc_oBounds = (AllDS['pupil_loc'] < 0.10) | (AllDS['pupil_loc'] > 0.90)
    loc_oBounds = np.any(loc_oBounds, axis=1)
    loc_nExist = np.any(AllDS['pupil_loc'] < 0, axis=1)
    loc = loc_oBounds & ~loc_nExist  # Location of images to remove
    AllDS = rmEntries(AllDS, loc)

    # Get ellipse centers, in case pupil is missing use the iris centers
    loc_nExist = np.any(AllDS['pupil_loc'] < 0, axis=1)
    ellipse_centers = AllDS['pupil_loc']
    ellipse_centers[loc_nExist, :] = AllDS['iris_loc'][loc_nExist, :]

    # Generate 2D histogram of pupil centers
    numBins = 5
    _, edgeList = np.histogramdd(ellipse_centers, bins=numBins)
    xEdges, yEdges = edgeList

    archNum = np.unique(AllDS['archive'],
                        return_index=True,
                        return_inverse=True)[2]

    # Bin the pupil center location and return that bin ID
    binx = np.digitize(ellipse_centers[:, 0], xEdges, right=True)
    biny = np.digitize(ellipse_centers[:, 1], yEdges, right=True)

    # Convert 2D bin locations into indices
    indx = np.ravel_multi_index((binx, biny, archNum),
                                (numBins+1, numBins+1, np.max(archNum)+1))
    indx = indx - np.min(indx)

    # Remove entries which occupy a single element in the grid
    print('Original # of entries: {}'.format(np.size(binx)))
    countInfo = np.unique(indx, return_counts=True)

    for rmInd in np.nditer(countInfo[0][countInfo[1] <= 2]):
        ent = indx == rmInd
        indx = indx[~ent]
        AllDS = copy.deepcopy(rmEntries(AllDS, ent))
    print('# of entries after stratification: {}'.format(np.size(indx)))
    return indx, AllDS


def generate_fileList(AllDS, mode='vanilla', notest=True):
    indx, AllDS = generate_strat_indices(AllDS) # This function removes samples with pupil center close to edges

    subject_identifier = list(map(lambda x,y:x+':'+y, AllDS['archive'], AllDS['subject_id']))

    archNum = np.unique(subject_identifier,
                        return_index=True,
                        return_inverse=True)[2]

    feats = np.stack([AllDS['im_num'], archNum, indx], axis=1)

    validPerc = .10

    if 'vanilla' in mode:
        # vanilla splits from the selected datasets.
        # Stratification by pupil center and dataset.
        params = re.findall('\d+', mode)
        if len(params) == 1:
            trainPerc = float(params[0])/100
            print('Training data set to {}%. Validation data set to {}%.'.format(
                        int(100*trainPerc), int(100*validPerc)))
        else:
            trainPerc = 1 - validPerc
            print('Training data set to {}%. Validation data set to {}%.'.format(
                        int(100*trainPerc), int(100*validPerc)))

        data_div = Datasplit(1, subject_identifier)

        if not notest:
            # Split into train and test
            train_feats, test_feats = train_test_split(feats,
                                                       train_size = trainPerc,
                                                       stratify = None,
                                                       shuffle=False)
        else:
            # Do not split into train and test
            train_feats = feats
            test_feats = []

        # Split training further into validation - Using this shuffle the dataset
        """ train_feats, valid_feats = train_test_split(train_feats,
                                                    test_size = 0.2,
                                                    random_state = None,
                                                    stratify = train_feats[:,-1]) """


        train_feats, valid_feats = train_test_split(train_feats,
                                                    test_size = 0.1,
                                                    shuffle=False)
        data_div.assignIdx(0, train_feats, valid_feats, test_feats)

    if 'fold' in mode:
        # K fold validation.
        K = int(re.findall('\d+', mode)[0])

        data_div = Datasplit(K, subject_identifier)
        skf = StratifiedKFold(n_splits=K, shuffle=True)
        train_feats, test_feats = train_test_split(feats,
                                                   train_size = 1 - validPerc,
                                                   stratify = indx)
        i=0
        for train_loc, valid_loc in skf.split(train_feats, train_feats[:, -1]):
            data_div.assignIdx(i, train_feats[train_loc, :],
                               train_feats[valid_loc, :],
                               test_feats)
            i+=1

    if 'none' in mode:
        # No splits. All images are placed in train, valid and test.
        # This process ensure's there no confusion.
        data_div = Datasplit(1, subject_identifier)
        data_div.assignIdx(0, feats, feats, feats)

    return data_div


def generateIdx(samplesList, batch_size):
    '''
    Takes in 2D array <samplesList>
    samplesList: 1'st dimension image number
    samplesList: 2'nd dimension hf5 file number
    batch_size: Number of images to be present in a batch
    If no entries are found, generateIdx will return an empty list of batches
    '''
    if np.size(samplesList) > 0:
        num_samples = samplesList.shape[0]
        num_batches = np.ceil(num_samples/batch_size).astype(np.int)
        np.random.shuffle(samplesList) # random.shuffle works on the first axis
        batchIdx_list = []
        for i in range(0, num_batches):
            y = (i+1)*batch_size if (i+1)*batch_size<num_samples else num_samples
            batchIdx_list.append(samplesList[i*batch_size:y, :])
    else:
        batchIdx_list = []
    return batchIdx_list

def foldInfo():
    D = {'train_idx': [], 'valid_idx': [], 'test_idx': []}
    return D

class Datasplit():
    def __init__(self, K, archs):
        self.splits = K
        self.folds = [foldInfo() for i in range(0, self.splits)]
        self.arch = np.unique(archs)

    def assignIdx(self, foldNum, train_idx, valid_idx, test_idx):
        # train, valid and test idx contains image number, h5 file and stratify index
        self.checkUnique(train_idx)
        self.checkUnique(valid_idx)
        self.checkUnique(test_idx)

        self.folds[foldNum]['train_idx'] = train_idx[:, :2] if type(train_idx) is not list else []
        self.folds[foldNum]['valid_idx'] = valid_idx[:, :2] if type(valid_idx) is not list else []
        self.folds[foldNum]['test_idx'] = test_idx[:, :2] if type(test_idx) is not list else []

    def checkUnique(self, ID):
        if type(ID) is not list:
            imNums = ID[:, 0]
            chunks = ID[:, 1]
            chunks_present = np.unique(chunks)
            for chunk in chunks_present:
                loc = chunks == chunk
                unq_flg = np.size(np.unique(imNums[loc])) != np.size(imNums[loc])
                if unq_flg:
                    print('Not unique! WARNING')

if __name__=="__main__":
    # This scripts verifies all datasets and returns the total number of images
    # Run sandbox.py to verify dataloader.
    path2data = '/media/rakshit/Monster/Datasets'
    path2arc_keys = os.path.join(path2data, 'MasterKey')

    AllDS = readArchives(path2arc_keys)
    datasets_present, subsets_present = listDatasets(AllDS)

    print('Datasets selected ---------')
    print(datasets_present)
    print('Subsets selected ---------')
    print(subsets_present)

    dataDiv_Obj = generate_fileList(AllDS, mode='vanilla')
    N = [value.shape[0] for key, value in dataDiv_Obj.folds[0].items() if len(value) > 0]
    print('Total number of images: {}'.format(np.sum(N)))

    with open('CurCheck.pkl', 'wb') as fid:
        pickle.dump(dataDiv_Obj, fid)
