import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt


def draw_gaze(img, eyeball, gaze_vector):

    img = img.numpy()
    eyeball = eyeball.numpy()
    gaze_vector = gaze_vector.numpy()

    eyeball_radius = eyeball[0].astype(int)
    eyebll_center_x = eyeball[1].astype(int)
    eyebll_center_y = eyeball[2].astype(int)

    img = img - img.min()
    img = (255*(img/img.max())).astype(np.uint8)
    out_img = np.stack([img]*3, axis=2)

    #draw the eyeball circle
    cv2.circle(out_img, (eyebll_center_x,eyebll_center_y), 
               eyeball_radius, (0,255,255), 3)
    
    #draw the eyeball center in image
    cv2.circle(out_img, (eyebll_center_x,eyebll_center_y), 
               0, (0,255,255), 3)
    
    #gaze vector line
    end_point_x = (eyeball_radius * gaze_vector[0]).astype(int) + eyebll_center_x
    end_point_y = (eyeball_radius * gaze_vector[1]).astype(int) + eyebll_center_y

    cv2.line(out_img, (eyebll_center_x, eyebll_center_y), 
             (end_point_x, end_point_y),
             (0,255,255), 3)

    plt.imshow(out_img)
    plt.savefig('frame_circle.jpg')

def draw_landmark(img, landmark_pupil, landmark_iris):

    img = img.numpy()
    landmark_iris = landmark_iris.numpy()
    landmark_pupil = landmark_pupil.numpy()

    img = img - img.min()
    img = (255*(img/img.max())).astype(np.uint8)
    out_img = np.stack([img]*3, axis=2)

    landmark_iris = np.reshape(landmark_iris[1:], (-1,2)).astype(int)
    landmark_pupil = np.reshape(landmark_pupil[1:], (-1,2)).astype(int)
    for i in range(landmark_pupil.shape[0]):

        cv2.circle(out_img, (landmark_iris[i][0], landmark_iris[i][1]), 
               0, (0,255,255), 3)
        cv2.circle(out_img, (landmark_pupil[i][0], landmark_pupil[i][1]), 
               0, (0,255,255), 3)

    plt.imshow(out_img)
    plt.savefig('landmark.jpg')

    


if __name__ == "__main__":
    import pickle
    import os
    from args_maker import *
    import helperfunctions.CurriculumLib as CurLib
    from helperfunctions.CurriculumLib import DataLoader_riteyes
    from torch.utils.data import DataLoader

    N_frames = 3

    path_cur_obj = os.path.join(default_repo,'cur_objs', 'dataset_selections.pkl')  # 'one_vs_one' 'cond_'+'TEyeD'+'.pkl'
    DS_sel = pickle.load(open(path_cur_obj, 'rb'))
    AllDS = CurLib.readArchives(masterkey_root)
    #Train and Validation object
    AllDS_cond = CurLib.selSubset(AllDS, DS_sel['train']['TEyeD'])
    dataDiv_obj = CurLib.generate_fileList(AllDS_cond, mode='vanilla', notest=False)
    train_obj = DataLoader_riteyes(dataDiv_obj, dataset_root, 'train', True, (480, 640), 
                                    scale=0.5, num_frames=N_frames)

    # %% Specify flags of importance
    train_obj.augFlag = False

    train_obj.equi_var = 0

    # %% Modify path information
    train_obj.path2data = dataset_root

    train_obj.scale = 1
    
    
    train_loader = DataLoader(train_obj,
                                shuffle=False,
                                num_workers=0,
                                drop_last=True,
                                pin_memory=True,
                                batch_size=2,
                                sampler=None,
                                )
    
    train_loader.dataset.sort('ordered')

    train_batches_per_ep = len(train_loader)


    train_loader = iter(train_loader)
    data_dict = next(train_loader)

    draw_gaze(data_dict['image'][0][0], data_dict['eyeball'][0][0], data_dict['gaze_vector'][0][0])
