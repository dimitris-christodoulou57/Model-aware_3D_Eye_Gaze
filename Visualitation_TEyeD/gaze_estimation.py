import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt


def draw_gaze(img, eyeball, gaze_vector):

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

def generate_gaze_gt(eyeball, gaze_vector, H, W):

    gaze_line = np.zeros((H,W))
    
    eyeball_radius = eyeball[0].astype(int)
    eyebll_center_x = eyeball[1].astype(int)
    eyebll_center_y = eyeball[2].astype(int)

    cv2.circle(gaze_line, (eyebll_center_x,eyebll_center_y), 
               eyeball_radius, 1, 3)
    
    #draw the eyeball center in image
    cv2.circle(gaze_line, (eyebll_center_x,eyebll_center_y), 
               0, 1, 3)
    
    #gaze vector line
    end_point_x = (eyeball_radius * gaze_vector[0]).astype(int) + eyebll_center_x
    end_point_y = (eyeball_radius * gaze_vector[1]).astype(int) + eyebll_center_y

    cv2.line(gaze_line, (eyebll_center_x, eyebll_center_y), 
             (end_point_x, end_point_y),
             1, 3)
    
    return gaze_line

def draw_landmark(img, landmark_pupil, landmark_iris):

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

    


