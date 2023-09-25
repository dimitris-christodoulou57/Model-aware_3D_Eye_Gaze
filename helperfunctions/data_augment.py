#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:05:17 2019

@author: Rakshit
"""
import cv2
import copy
import random
import numpy as np
import imgaug.augmenters as iaa

from scipy.ndimage import gaussian_filter, rotate, shift

from helperfunctions.helperfunctions import plot_images_with_annotations
from helperfunctions.helperfunctions import pad_to_shape, scale_by_ratio

# Only for debugging purposes
# from helperfunctions import plot_images_with_annotations
# from helperfunctions import pad_to_shape, scale_by_ratio


def apply_motion_blur(image, size, angle):

    # Code taken from: https://stackoverflow.com/a/57629531/2127561
    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    return cv2.filter2D(image, -1, k)


class flip():

    def __init__(self, ):
        pass

    def __call__(self, data_dict, mask_available):
        num_of_frames = data_dict['image'].shape[0]

        for i in range(num_of_frames):
            data_dict['image'][i] = np.fliplr(data_dict['image'][i])
            if mask_available:
                data_dict['mask'][i] = np.fliplr(data_dict['mask'][i])
            
            _, height, width = data_dict['image'].shape

            # Shift the center
            data_dict['iris_ellipse'][i][0] = width - data_dict['iris_ellipse'][i][0]
            data_dict['pupil_center'][i][0] = width - data_dict['pupil_center'][i][0]
            data_dict['pupil_ellipse'][i][0] = width - data_dict['pupil_ellipse'][i][0]

            # Invert the ellipse
            if data_dict['iris_ellipse_available'][i]:
                data_dict['iris_ellipse'][i][-1] = -data_dict['iris_ellipse'][i][-1]
            if data_dict['pupil_ellipse_available'][i]:
                data_dict['pupil_ellipse'][i][-1] = -data_dict['pupil_ellipse'][i][-1]

        #shift the center of eyeball anf pupil center.
        data_dict['eyeball'][..., 1] = width - data_dict['eyeball'][..., 1]
        data_dict['gaze_vector'][..., 0] =  - data_dict['gaze_vector'][..., 0]

        data_dict['iris_lm_2D'][...,1] = width - data_dict['iris_lm_2D'][..., 1]
        data_dict['iris_lm_2D'][...,3] = width - data_dict['iris_lm_2D'][..., 3]
        data_dict['iris_lm_2D'][...,5] = width - data_dict['iris_lm_2D'][..., 5]
        data_dict['iris_lm_2D'][...,7] = width - data_dict['iris_lm_2D'][..., 7]
        data_dict['iris_lm_2D'][...,9] = width - data_dict['iris_lm_2D'][..., 9]
        data_dict['iris_lm_2D'][...,11] = width - data_dict['iris_lm_2D'][..., 11]
        data_dict['iris_lm_2D'][...,13] = width - data_dict['iris_lm_2D'][..., 13]
        data_dict['iris_lm_2D'][...,15] = width - data_dict['iris_lm_2D'][..., 15]

        data_dict['iris_lm_3D'][...,1] = width - data_dict['iris_lm_3D'][..., 1]
        data_dict['iris_lm_3D'][...,4] = width - data_dict['iris_lm_3D'][..., 4]
        data_dict['iris_lm_3D'][...,7] = width - data_dict['iris_lm_3D'][..., 7]
        data_dict['iris_lm_3D'][...,10] = width - data_dict['iris_lm_3D'][..., 10]
        data_dict['iris_lm_3D'][...,13] = width - data_dict['iris_lm_3D'][..., 13]
        data_dict['iris_lm_3D'][...,16] = width - data_dict['iris_lm_3D'][..., 16]
        data_dict['iris_lm_3D'][...,19] = width - data_dict['iris_lm_3D'][..., 19]
        data_dict['iris_lm_3D'][...,22] = width - data_dict['iris_lm_3D'][..., 22]

        data_dict['pupil_lm_2D'][...,1] = width - data_dict['pupil_lm_2D'][..., 1]
        data_dict['pupil_lm_2D'][...,3] = width - data_dict['pupil_lm_2D'][..., 3]
        data_dict['pupil_lm_2D'][...,5] = width - data_dict['pupil_lm_2D'][..., 5]
        data_dict['pupil_lm_2D'][...,7] = width - data_dict['pupil_lm_2D'][..., 7]
        data_dict['pupil_lm_2D'][...,9] = width - data_dict['pupil_lm_2D'][..., 9]
        data_dict['pupil_lm_2D'][...,11] = width - data_dict['pupil_lm_2D'][..., 11]
        data_dict['pupil_lm_2D'][...,13] = width - data_dict['pupil_lm_2D'][..., 13]
        data_dict['pupil_lm_2D'][...,15] = width - data_dict['pupil_lm_2D'][..., 15]

        data_dict['pupil_lm_3D'][...,1] = width - data_dict['pupil_lm_3D'][..., 1]
        data_dict['pupil_lm_3D'][...,4] = width - data_dict['pupil_lm_3D'][..., 4]
        data_dict['pupil_lm_3D'][...,7] = width - data_dict['pupil_lm_3D'][..., 7]
        data_dict['pupil_lm_3D'][...,10] = width - data_dict['pupil_lm_3D'][..., 10]
        data_dict['pupil_lm_3D'][...,13] = width - data_dict['pupil_lm_3D'][..., 13]
        data_dict['pupil_lm_3D'][...,16] = width - data_dict['pupil_lm_3D'][..., 16]
        data_dict['pupil_lm_3D'][...,19] = width - data_dict['pupil_lm_3D'][..., 19]
        data_dict['pupil_lm_3D'][...,22] = width - data_dict['pupil_lm_3D'][..., 22]

        return data_dict

# choice_list=[0, 1, 8, 9, 11]
class augment():
    def __init__(self, choice_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mask_available=False):

        self.choice_mux = {0: self.do_nothing,
                           1: self.gauss_blur,
                           2: self.motion_blur,
                           3: self.modify_gamma,
                           4: self.modify_exposure,
                           5: self.gauss_noise,
                           6: self.circular_lines,
                           7: self.scale_image,
                           8: self.rotate_image,
                           9: self.translate_image,
                           10: self.add_fog,
                           11: self.random_noise}
        self.choice_list = choice_list

        self.fog_augger = iaa.Fog()
        self.mask_available = mask_available

    def __call__(self, data_dict, choice=None):
        return self.forward(data_dict, choice)

    def forward(self, data_dict, choice=None):
        choice = random.choice(self.choice_list) if choice is None else choice
        data_dict = self.choice_mux[choice](copy.deepcopy(data_dict))
        return data_dict

    def test(self, data_dict):

        out_data_dict_list = []
        for aug_index in range(11):
            out_data_dict_list.append(self.forward(data_dict,
                                                   choice=aug_index))
        plot_images_with_annotations(out_data_dict_list,
                                     plot_annots=True,
                                     subplots=(3,4),
                                     write='./aug_types.png')
        print('Augmented images plotted')

    def gauss_blur(self, data_dict):
        # Gaussian blur
        sigma_value = np.random.randint(1, 2)
        data_dict['image'] = gaussian_filter(data_dict['image'], sigma_value)
        return data_dict

    def motion_blur(self, data_dict):
        angle = int(180*np.random.rand(1, ))
        data_dict['image'] = apply_motion_blur(data_dict['image'],
                                               size=7, angle=angle)
        return data_dict

    def modify_gamma(self, data_dict):
        gamma = [0.6, 0.8, 1.2, 1.4][np.random.randint(0, 4)]
        table = 255.0*(np.linspace(0, 1, 256)**gamma)
        data_dict['image'] = cv2.LUT(data_dict['image'], table).astype(np.uint8)
        return data_dict

    def modify_exposure(self, data_dict):
        # Exposure +/- amt to washout Iris
        if data_dict['iris_ellipse_available']:
            loc = data_dict['mask'] == 2  # Iris class index
            iris_intensity = data_dict['image'][loc]
            max_inc = int(0.8*(255 - np.median(iris_intensity))) if np.any(loc) else 50
            max_red = int(0.8*(np.median(iris_intensity))) if np.any(loc) else 50
        else:
            max_inc = 50
            max_red = 50
        dL = (max_inc + max_red)*np.random.rand(1) - max_red
        data_dict['image'] = data_dict['image'].astype(np.float32) + dL
        data_dict['image'] = np.clip(data_dict['image'], 0, 255).astype(np.uint8) # Clip to limits
        return data_dict

    def gauss_noise(self, data_dict):
        # Gaussian noise taken from https://stackoverflow.com/questions/43699326/
        mean = 0.0   # some constant
        std = 14*np.random.rand() + 2   # some constant (standard deviation)
        height, width = data_dict['image'].shape
        data_dict['image'] = data_dict['image'] + \
            np.random.normal(mean, std, (height, width))
        data_dict['image'] = np.clip(data_dict['image'], 0, 255)
        data_dict['image'] = data_dict['image'].astype(np.uint8)
        return data_dict
    

    def random_noise(self, data_dict):
        # Gaussian noise taken from https://stackoverflow.com/questions/43699326/
        frame, height, width = data_dict['image'].shape
        random_noise_perc = 0.3
        data_dict['image'] = data_dict['image'] + \
            np.random.rand(frame,height, width) * random_noise_perc
        data_dict['image'] = np.clip(data_dict['image'], 0, 255)
        data_dict['image'] = data_dict['image'].astype(np.uint8)
        return data_dict

    def circular_lines(self, data_dict):
        # Circular white lines anywhere in the image
        height, width = data_dict['image'].shape
        shift_x = width*np.random.rand(1, ) - width//2
        shift_y = height*np.random.rand(1, ) - height//2
        xc, yc = width//2 - shift_x, height//2 - shift_y

        num_lines = np.random.randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*np.random.rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta)
            data_dict['image'] = cv2.line(data_dict['image'],
                                          (x1, y1), (x2, y2),
                                          (255, 255, 255), 4)
        data_dict['image'] = data_dict['image'].astype(np.uint8)
        return data_dict

    def translate_image(self, data_dict):
        frames, height, width = data_dict['image'].shape

        shift_height = int(height * 0.2)
        shift_width = int(width * 0.2)
        shift_width = np.random.randint(-shift_width, shift_width + 1)
        shift_height = np.random.randint(-shift_height, shift_height + 1)
        shift_tup = (shift_width, shift_height)

        if np.all(data_dict['pupil_center_available']):
            temp = data_dict['pupil_center'] + np.array(shift_tup)

        elif np.all(data_dict['iris_ellipse_available']):
            temp = data_dict['iris_ellipse'][...,:2] + np.array(shift_tup)

        if np.any(temp <= 30) or \
            (np.all(temp[0] > (width - 30))) or \
                (np.all(temp[1] > (height - 30))):
            # Ensure that the centers stay within the image spatial extent
            # Failure to do so could result in NaNs and unwanted consequences
            # In such conditions, reroute the shift to 0s.
            shift_tup = (0, 0)
            return data_dict
        
        shifted_image = shift(data_dict['image'], (0, shift_height, shift_width), mode='constant', cval=0.0)
        data_dict['image'] = shifted_image

        if self.mask_available:
            mask_trans = shift(data_dict['mask'], (0, shift_height, shift_width), mode='constant', cval=0.0)
            data_dict['mask'] = mask_trans


        if np.all(data_dict['pupil_center_available']):
            data_dict['pupil_center'] += np.array(shift_tup)

        if np.all(data_dict['pupil_ellipse_available']):
            data_dict['pupil_ellipse'][...,:2] += np.array(shift_tup)

        if np.all(data_dict['iris_ellipse_available']):
            data_dict['iris_ellipse'][...,:2] += np.array(shift_tup)

        data_dict['eyeball'][...,1:3] += np.array(shift_tup)

        #TODO translate the pupil and iris landmark
        return data_dict

    def scale_image(self, data_dict):
        shape = data_dict['image'].shape
        data_dict = scale_by_ratio(data_dict, 0.4*random.random() + 0.5)
        data_dict = pad_to_shape(data_dict, shape)
        return data_dict

    def rotate_image(self, data_dict):
        # Rotate image upto +/- 45 degrees
        ang = 45*2*(np.random.rand(1) - 0.5)
        ang_rad = np.deg2rad(ang)
        _, height, width = data_dict['image'].shape

        center = (int(0.5*width), int(0.5*height))

        data_dict['image'] = rotate(data_dict['image'], ang.item(), axes=(1,2), reshape=False)

        if self.mask_available:
            data_dict['mask'] = rotate(data_dict['mask'], ang.item(), axes=(1,2), reshape=False)

        # Rotation matrix - note that it is transposed!
        rot_mat = np.array([[np.cos(ang_rad), np.sin(ang_rad)],
                            [-np.sin(ang_rad), np.cos(ang_rad)]]).squeeze()

        # Pupil center normalized to image center
        if np.all(data_dict['pupil_center_available']):
            shifted_pc = data_dict['pupil_center'] - np.array(center)
            data_dict['pupil_center'] = np.dot(shifted_pc, rot_mat) +\
                np.array(center)

        if np.all(data_dict['pupil_ellipse_available']):
            shifted_pc = data_dict['pupil_ellipse'][...,:2] - np.array(center)
            data_dict['pupil_ellipse'][...,:2] =\
                np.dot(shifted_pc, rot_mat) + np.array(center)
            data_dict['pupil_ellipse'][...,-1] += -ang_rad  # Note the -ve sign

        if np.all(data_dict['iris_ellipse_available']):
            shifted_ic = data_dict['iris_ellipse'][...,:2] - np.array(center)
            data_dict['iris_ellipse'][...,:2] =\
                np.dot(shifted_ic, rot_mat) + np.array(center)
            data_dict['iris_ellipse'][...,-1] += -ang_rad  # Note the -ve sign

        shifted_pc = data_dict['eyeball'][...,1:3] - np.array(center)
        data_dict['pupil_ellipse'][...,1:3] =\
                np.dot(shifted_pc, rot_mat) + np.array(center)
        
        #rotate the gaze vector
        rot_mat = np.array([[np.cos(ang_rad).item(), -np.sin(ang_rad).item(), 0],
                            [np.sin(ang_rad).item(), np.cos(ang_rad).item(), 0],
                            [0, 0, 1]]).squeeze()
        data_dict['gaze_vector'] = np.dot(data_dict['gaze_vector'], rot_mat)

        #TODO rotate the pupil and iris landmark

        return data_dict
    
    def add_fog(self, data_dict):
        data_dict['image'] = self.fog_augger(images=data_dict['image'])
        return data_dict

    def do_nothing(self, data_dict):
        return data_dict


def getRandomLine(xc, yc, theta):
    x1 = xc - 50*np.random.rand(1) * (1 if np.random.rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150*np.random.rand(1) + 50) * \
        (1 if np.random.rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    return [int(ele.item()) for ele in [x1, y1, x2, y2]]


def normalizer(image):
    # No matter what type the image is in, it will scale and shift to 0, 255
    return np.uint8((image-image.min())*255/(image.max()-image.min()))


if __name__ == '__main__':
    path_H5 = '/data/datasets/All'
    name_H5 = 'OpenEDS_train_1.h5'

    import os
    import h5py

    f = h5py.File(os.path.join(path_H5, name_H5), mode='r')

    im_num = 500

    # Read information
    image = f['Images'][im_num, ...]

    # Get pupil center
    if f['pupil_loc'].__len__() != 0:
        pupil_center = f['pupil_loc'][im_num, ...]
        pupil_center_available = True
    else:
        pupil_center_available = False
        pupil_center = -np.ones(2, )

    # Get mask without skin
    if f['Masks_noSkin'].__len__() != 0:
        mask_noSkin = f['Masks_noSkin'][im_num, ...]
        mask_available = True
    else:
        mask_noSkin = -np.ones(image.shape[:2])
        mask_available = False

    # Pupil ellipse parameters
    if f['Fits']['pupil'].__len__() != 0:
        pupil_ellipse_available = True
        pupil_param = f['Fits']['pupil'][im_num, ...]
    else:
        pupil_ellipse_available = False
        pupil_param = -np.ones(5, )

    # Iris ellipse parameters
    if f['Fits']['iris'].__len__() != 0:
        iris_ellipse_available = True
        iris_param = f['Fits']['iris'][im_num, ...]
    else:
        iris_ellipse_available = False
        iris_param = -np.ones(5, )

    data_dict = {}
    data_dict['mask'] = mask_noSkin
    data_dict['image'] = image
    data_dict['ds_num'] = 0
    data_dict['pupil_center'] = pupil_center
    data_dict['iris_ellipse'] = iris_param
    data_dict['pupil_ellipse'] = pupil_param

    # Keep flags as separate entries
    data_dict['mask_available'] = mask_available
    data_dict['pupil_center_available'] = pupil_center_available if not np.all(pupil_center == -1) else False
    data_dict['iris_ellipse_available'] = iris_ellipse_available if not np.all(iris_param == -1) else False
    data_dict['pupil_ellipse_available'] = pupil_ellipse_available if not np.all(pupil_param == -1) else False

    augger = augment()
    augger.test(data_dict)
