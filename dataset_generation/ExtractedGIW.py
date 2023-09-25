#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:53:54 2021

@author: rakshit

This scrip extracts two types of video sequences from the Gaze-in-Wild project
A) Calibration eye and scene videos including the VOR sequence
B) Eye videos with manual annotations
"""
import os
import sys
import pickle
import msgpack
import argparse
import itertools
import matlab.engine
import scipy.io as scio

import numpy as np

sys.path.append('/home/rakshit/Documents/MATLAB/gaze-in-wild/SupportFunctions/PythonSupport')

from pprint import pprint
from HelperFunctions import convert_ProcessData_to_pycompat

eng = matlab.engine.start_matlab()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='extract calibration sequence or labels',
                    type=str, default='calib')
parser.add_argument('--path2ds', help='Path to datasets',
                    type=str, default='/media/rakshit/tank/Dataset')
parser.add_argument('--path_save', help='directory to save outputs',
                    type=str, default='/media/rakshit/tank/GIW_op')
parser.add_argument('--path2extracted', default='/media/rakshit/tank/GIW_rearranged')
args = parser.parse_args()

print('Argument list: ')
pprint(vars(args))

def _load_object_legacy(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'rb') as fh:
        data = pickle.load(fh, encoding='bytes')
    return data

def load_object(file_path, allow_legacy=True):
    import gc
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'rb') as fh:
        try:
            gc.disable()  # speeds deserialization up.
            data = msgpack.unpack(fh, raw=False, strict_map_key=False)
        except Exception as e:
            if not allow_legacy:
                raise e
            else:
                print('{} has a deprecated format: Will be updated on save'.format(file_path))
                data = _load_object_legacy(file_path)
        finally:
            gc.enable()
    return data

def match_stamps_and_get_data(timestamps, pupil_data, cond):
    data = {}
    temp = [[ele['timestamp'], loc_entry] for loc_entry, ele in enumerate(pupil_data['pupil_positions']) if ele['id']==cond]
    global_stamps, loc_entry = zip(*temp)
    global_stamps = np.array(global_stamps)

    # For a given stamp for each individual eye video, find consecutive stamp
    # in pupil_data stamps
    for frame, stamp in np.ndenumerate(timestamps):
        # frame returned as a tuple so just pick the actual entry value
        loc = np.argmin(np.abs(stamp - global_stamps))
        data[frame[0]] = {'pupil_positions':[], 'gaze_positions':[]}
        data[frame[0]]['pupil_positions'] = pupil_data['pupil_positions'][loc_entry[loc]]

    temp = [[ele['timestamp'], loc_entry] for loc_entry, ele in enumerate(pupil_data['gaze_positions'])]
    global_stamps, loc_entry = zip(*temp)
    global_stamps = np.array(global_stamps)

    for frame, stamp in np.ndenumerate(timestamps):
        # frame returned as a tuple so just pick the actual entry value
        loc = np.argmin(np.abs(stamp - global_stamps))
        data[frame[0]]['gaze_positions'] = pupil_data['gaze_positions'][loc_entry[loc]]

    return data

def extract_calib_pupil_data(path_data, PrIdx, TrIdx):
    # Load calibration points
    path_calib_data = os.path.join(path_data, str(PrIdx), str(TrIdx), 'Gaze', 'offline_data')
    calib_data = load_object(path_calib_data + '/offline_calibration_gaze')

    N = len(calib_data['manual_ref_positions'])

    frames = []
    calib_dict = {'calib': {}}
    for i in range(N):
        frames.append(calib_data['manual_ref_positions'][i]['index_range'])
        calib_dict['calib'][i] = calib_data['manual_ref_positions'][i]

    frame_range = list(itertools.chain(*frames))
    calib_dict['calib_start'] = min(frame_range)
    calib_dict['calib_stop'] = max(frame_range)

    # Load timestamps
    timestamps_eye0 = np.load(os.path.join(path_data, str(PrIdx), str(TrIdx), 'Gaze', 'eye0_timestamps.npy'))
    timestamps_eye1 = np.load(os.path.join(path_data, str(PrIdx), str(TrIdx), 'Gaze', 'eye1_timestamps.npy'))

    # Load pupil data
    pupil_data = load_object(os.path.join(path_data, str(PrIdx), str(TrIdx), 'Gaze', 'pupil_data'))

    # match timestamps to stamps from pupil_data and save out pupil data
    calib_dict['pupil_0'] = match_stamps_and_get_data(np.array(timestamps_eye0).squeeze(),
                                                      pupil_data, cond=0)
    calib_dict['pupil_1'] = match_stamps_and_get_data(np.array(timestamps_eye1).squeeze(),
                                                      pupil_data, cond=1)

    return calib_dict


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

if __name__=='__main__':
    list_name = np.arange(1, 24)
    loc_prune = np.in1d(list_name, [4, 5, 7, 21])
    list_name = list_name[~loc_prune]

    path_data = os.path.join(args.path2ds, 'Gaze-in-Wild')

    list_task = ['Indoor_Walk', 'Ball_Catch', 'Visual_Search', 'Tea_Making']

    # all_calib_pupil_data = {}
    for task in list_task:

        path_Labels = os.path.join(path_data, 'extracted_data', task, 'Labels')
        path_ProcessData = os.path.join(path_data, 'extracted_data', task, 'ProcessData_cleaned')
        path_OperationalData = os.path.join(path_data, 'extracted_data', task, 'TempData')

        list_ProcessData = os.listdir(path_ProcessData)

        for fid_pd in list_ProcessData:
            ProcessData = loadmat(os.path.join(path_ProcessData, fid_pd))['ProcessData']
            # ProcessData = convert_ProcessData_to_pycompat(ProcessData)
            # ProcessData = eng.load(os.path.join(path_ProcessData, fid_pd))

            PrIdx = ProcessData['PrIdx']
            TrIdx = ProcessData['TrIdx']

            OperationalData = eng.load(os.path.join(path_OperationalData,
                                                    'Params_PrIdx_{}_TrIdx_{}.mat'.format(PrIdx, TrIdx)))
            RotMat_etg = OperationalData['RotMat_etg']
            R0 = ProcessData['R0']
            R1 = ProcessData['R1']

            if args.mode == 'calib':
                calib_pupil_data = extract_calib_pupil_data(path_data, PrIdx, TrIdx)
                # all_calib_pupil_data[(PrIdx.item(), TrIdx.item())] = calib_pupil_data
                calib_pupil_data['eye0_GIW_to_PL'] = np.linalg.inv(R0).dot(np.linalg.inv(RotMat_etg))
                calib_pupil_data['eye1_GIW_to_PL'] = np.linalg.inv(R1).dot(np.linalg.inv(RotMat_etg))
                calib_pupil_data['eyec_GIW_to_PL'] = np.linalg.inv(RotMat_etg)

            elif args.mode == 'classification':
                sys.exit('Not imlemented')

            with open(args.path_save+'/calib_pupil_data_PrIdx_{}_TrIdx_{}.pkl'.format(PrIdx, TrIdx), 'wb') as fid:
                pickle.dump(calib_pupil_data, fid)
