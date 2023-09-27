#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""

import os
import gc
import sys
import cv2
import copy
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pprint

from models_mux import model_dict
from args_maker import make_args as default_args
from helperfunctions.utils import get_nparams
from helperfunctions.helperfunctions import plot_segmap_ellpreds
from helperfunctions.helperfunctions import contruct_ellipse_from_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', type=str, default='../examples/',
                        help='path to eye videos')
    parser.add_argument('--save_maps', type=int, default=0,
                        help='save segmentation maps')
    parser.add_argument('--save_overlay', type=int, default=1,
                        help='save output overlay')
    parser.add_argument('--plot_ellipses', type=int, default=0,
                        help='plots regressed or ellipse fits')
    parser.add_argument('--ext', type=str, default='git_ok',
                        help='process videos with given extension')
    parser.add_argument('--loadfile', type=str, default='../pretrained/public.git_ok',
                        help='choose the weights you want to evalute the videos with. Recommended: all')
    parser.add_argument('--align_width', type=int, default=1,
                        help='reshape videos by matching width, default: True')
    parser.add_argument('--eval_on_cpu', type=int, default=0,
                        help='evaluate using CPU instead of GPU')
    parser.add_argument('--check_for_string_in_fname', type=str, default='',
                        help='process video with a certain string in filename')
    parser.add_argument('--skip_ransac', type=int, default=1,
                        help='if using ElliFit, it skips outlier removal')
    parser.add_argument('--display_regressed_ellipses', type=int, default=1,
                        help='use regressed ellipse parameters')

    args = parser.parse_args()
    opt = vars(args)
    print('------')
    print('parsed arguments:')
    pprint(opt)

    # %% For compatibility
    args.use_ada_instance_norm_mixup = 0

    return args

# %% Preprocessing functions and module
# Input frames must be resized to 320X240


def preprocess_frame(img, op_shape, align_width=True):

    if align_width:
        # If aliging the width to maintain a certain aspect ratio. Most use
        # cases fall within this flag. Notable exception is the OpenEDS dataset

        if op_shape[1] != img.shape[1]:
            # If the horizontal widths do not match ..

            sc = op_shape[1]/img.shape[1]  # Find the scaling required
            width = int(img.shape[1] * sc)
            height = int(img.shape[0] * sc)
            img = cv2.resize(img,
                             (width, height),
                             interpolation=cv2.INTER_LANCZOS4)

            if op_shape[0] > img.shape[0]:
                # Vertically pad array if the vertical dimensions do not match
                # and the number of required pixels are greater
                pad_width = op_shape[0] - img.shape[0]
                if pad_width % 2 == 0:
                    # If the number of required pixels to be padded are
                    # divisible by 2
                    img = np.pad(img,
                                 ((pad_width//2, pad_width//2),
                                  (0, 0)))
                else:
                    # Else, keep one side short
                    img = np.pad(img,
                                 ((pad_width//2, 1+pad_width//2),
                                 (0, 0)))

                # Scaling and shifting amounts
                scale_shift = (sc, pad_width)

            elif op_shape[0] < img.shape[0]:
                # Vertically chop array off if the number of required pixels
                # are lesser
                pad_width = op_shape[0] - img.shape[0]
                if pad_width % 2 == 0:
                    img = img[-pad_width//2:+pad_width//2]
                else:
                    img = img[-pad_width//2:1+pad_width//2]

                # Scaling and shifting amounts
                scale_shift = (sc, pad_width)

            else:
                scale_shift = (sc, 0)
        else:
            scale_shift = (1, 0)
    else:
        sys.exit('Height alignment not implemented! Exiting ...')

    # Normalize the image to Z scores
    img = (img - img.mean())/img.std()

    # Add a dummy color channel
    img = torch.from_numpy(img).to(torch.float32)

    # Return values
    return img, scale_shift


# %% Forward operation on network
def evaluate_ellseg_on_image(frame, model):

    assert len(frame.shape) == 3, 'Frame must be [1,H,W]'

    with torch.no_grad():
        data_dict = {'image': frame}
        out_dict = model(data_dict)

    if np.sum(out_dict['mask'] == 2) > 50:
        # TODO
        model_pupil = type('model', (object, ), {})
        model_pupil.model = np.array([-1, -1, -1, -1, -1])
        model_pupil.Phi = np.array([-1, -1, -1, -1, -1])
    else:
        # print('Not enough pupil points')
        model_pupil = type('model', (object, ), {})
        model_pupil.model = np.array([-1, -1, -1, -1, -1])
        model_pupil.Phi = np.array([-1, -1, -1, -1, -1])

    if np.sum(out_dict == 1) > 50:
        # TODO
        model_iris = type('model', (object, ), {})
        model_iris.model = np.array([-1, -1, -1, -1, -1])
        model_iris.Phi = np.array([-1, -1, -1, -1, -1])
    else:
        # print('Not enough iris points')
        model_iris = type('model', (object, ), {})
        model_iris.model = np.array([-1, -1, -1, -1, -1])
        model_iris.Phi = np.array([-1, -1, -1, -1, -1])

    out_dict['fit_pupil_ellipse'] = model_pupil.model
    out_dict['fit_iris_ellipse'] = model_iris.model

    return out_dict

# %% Rescale operation to bring mask & ellipses back to original res


def rescale_to_original(out_dict, scale_shift, orig_shape):

    pupil_ellipse = out_dict['pupil_ellipse'].squeeze()
    iris_ellipse = out_dict['iris_ellipse'].squeeze()
    seg_map = out_dict['mask'].squeeze()

    # Fix pupil ellipse
    pupil_ellipse[1] = pupil_ellipse[1] - np.floor(scale_shift[1]//2)
    pupil_ellipse[:-1] = pupil_ellipse[:-1]*(1/scale_shift[0])

    # Fix iris ellipse
    iris_ellipse[1] = iris_ellipse[1] - np.floor(scale_shift[1]//2)
    iris_ellipse[:-1] = iris_ellipse[:-1]*(1/scale_shift[0])

    if scale_shift[1] < 0:
        # Pad background
        seg_map = np.pad(seg_map,
                         ((-scale_shift[1]//2, -scale_shift[1]//2),
                          (0, 0)))
    elif scale_shift[1] > 0:
        # Remove extra pixels
        seg_map = seg_map[scale_shift[1]//2:-scale_shift[1]//2, ...]

    seg_map = cv2.resize(seg_map,
                         (orig_shape[1], orig_shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    out_dict['mask'] = seg_map
    out_dict['iris_ellipse'] = iris_ellipse
    out_dict['pupil_ellipse'] = pupil_ellipse
    return out_dict


# %% Definition for processing per video


def evaluate_ellseg_per_video(path_vid, args, model):
    path_dir, full_file_name = os.path.split(path_vid)
    file_name = os.path.splitext(full_file_name)[0]
    
    if os.path.exists(os.path.join(path_dir, 'meta', file_name+'_meta.pkl')):
        print('{} already processed'.format(file_name))
        return

    if args['eval_on_cpu']:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if args['check_for_string_in_fname'] in file_name:
        print('Processing file: {}'.format(path_vid))
    else:
        print('Skipping video {}'.format(path_vid))
        return False

    # Open a video reader object
    vid_obj = cv2.VideoCapture(str(path_vid))

    # Return parameters of the video object
    FR_COUNT = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
    FR = vid_obj.get(cv2.CAP_PROP_FPS)
    H = vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W = vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH)

    if args['save_overlay']:
        
        # Ensure a folder exists
        os.makedirs(os.path.join(path_dir, 'meta'), exist_ok=True)
        
        # Output video with predicted mask
        path_vid_out = os.path.join(path_dir, 'meta', file_name+'_ellseg.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_out = cv2.VideoWriter(path_vid_out, fourcc,
                                  int(FR), (int(W), int(H)))

    # Dictionary to save output ellipses
    ellipse_out_dict = {}

    ret = True
    pbar = tqdm(total=FR_COUNT)

    counter = 0  # Frame counter
    while ret:
        ret, frame = vid_obj.read()

        if not ret:
            print('Video read end reached.')
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.max() < 5:
            # Frame is quite dark, skip processing this frame.
            print('Poor luminance in frame: {}'.format(counter))
            continue

        frame_scaled_shifted, scale_shift = preprocess_frame(frame,
                                                             (240, 320),
                                                             args['align_width'])

        # Add a dummy batch channel
        input_tensor = frame_scaled_shifted.unsqueeze(0).to(device)

        # Run the prediction network
        out_dict = evaluate_ellseg_on_image(input_tensor, model)

        # Return ellipse predictions back to original dimensions
        out_dict = rescale_to_original(out_dict,
                                       scale_shift,
                                       frame.shape)

        # Generate visuals
        if args['display_regressed_ellipses']:
            frame_overlayed_with_op = plot_segmap_ellpreds(frame,
                                                           out_dict['mask'],
                                                           out_dict['pupil_ellipse'],
                                                           out_dict['iris_ellipse'],
                                                           plot_ellipses=args['plot_ellipses'])
        else:
            # Fit ellipse to the output segmentation mask and display that
            pupil_fit, iris_fit = contruct_ellipse_from_mask(out_dict['mask'],
                                                             out_dict['pupil_ellipse'][:2],
                                                             out_dict['iris_ellipse'][:2])
            frame_overlayed_with_op = plot_segmap_ellpreds(frame,
                                                           out_dict['mask'],
                                                           pupil_fit,
                                                           iris_fit,
                                                           plot_ellipses=args['plot_ellipses'])

        if args['save_overlay']:
            vid_out.write(frame_overlayed_with_op[..., ::-1])

        # Append to dictionary and move everything to CPU if not already on it
        for key, value in out_dict.items():
            if 'torch' in str(type(value)):
                out_dict[key] = value.cpu().numpy()
                
        out_dict.pop('predict')

        ellipse_out_dict[counter] = out_dict

        pbar.update(1)
        counter+=1

    if args['save_overlay']:
        vid_out.release()
        
    vid_obj.release()
    pbar.close()

    if args['save_maps']:
        # Save out ellipses and masks dictionary
        os.makedirs(os.path.join(path_dir, 'meta'), exist_ok=True)
        with open(os.path.join(path_dir, 'meta', file_name+'_meta.pkl'), 'wb') as fid:
            pickle.dump(ellipse_out_dict, fid)
            
    # dT_list= [value['dT'] for key, value in ellipse_out_dict.items()]
    # print('Frame rate: {}'.format(1/np.mean(dT_list)))
    
    del ellipse_out_dict
    del out_dict
    gc.collect()

    return True


if __name__=='__main__':
    args = vars(parse_args())

    #%% Load network, weights and get ready to evalute
    netDict = torch.load(args['loadfile'], map_location='cpu')
    print("load weights")

    # Update values from the arguments stored in weights, this ensures that no
    # parameter gets incorrectly set
    
    for key, value in netDict['args'].items():
        args[key] = value
        
    # Feed in args missing in dict due to repo updates
    args['groups'] = 1

    model = model_dict['DenseElNet'](args,
                                     norm=torch.nn.InstanceNorm2d,
                                     act_func=torch.nn.functional.leaky_relu)

    print("model loaded")

    # Strictly load and assign weights. They must match perfectly.
    model.load_state_dict(netDict['state_dict'], strict=True)

    if not args['eval_on_cpu']:
        print("cuda run")
        model.cuda()
        
    num_params = get_nparams(model)
    print('# of paramters: %d' %num_params)

    #%% Read in each video
    path_obj = Path(args['path2data']).rglob('*.'+args['ext'])

    for path in path_obj:
        print(path)
        if '_ellseg' not in str(path):
            try:
                evaluate_ellseg_per_video(path, args, model)
            except:
                print('Failed to evaluate {}. Skipping.'.format(path))


