import time
from collections import OrderedDict
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
import cv2
from pytorch3d import transforms
from einops import rearrange, reduce, repeat
try:
    from helperfunctions.helperfunctions import assert_torch_invalid
except:
    def assert_torch_invalid(X, string):
        assert not torch.isnan(X).any(), print('NaN problem ['+string+']')
        assert not torch.isinf(X).any(), print('inf problem ['+string+']')
        assert torch.all(torch.isfinite(X)), print('Some elements not finite problem ['+string+']')
        assert X.numel() > 0, print('Empty tensor problem ['+string+']')


def render_semantics(out_dict, H, W, args, data_dict=None):
    '''
    T is the 3D eyeball center position
    R_pupil: is the rotation matrix from 3D to camera frame (expressed in roll, yaw, pitch)
    R_iris: is the rotation matrix from 3D to camera frame (expressed in roll, yaw, pitch)
    r_pupil: is the radius of the pupil
    r_iris: is the iris radius
    L_pupil: distance between eyeball and pupil center
    L_iris: distance between eyeball and iris center
    focal: [f_x, f_y]
    '''

    T=out_dict['T'] 
    R=out_dict['R']
    r_pupil=out_dict['r_pupil'] 
    r_iris=out_dict['r_iris']
    L=out_dict['L'] 
    focal=out_dict['focal']

    # Scale and bound predictions
    T, R, r_pupil, r_iris, L, L_p, focal = scale_and_bound(T, R, r_pupil, r_iris, L, focal, args)

    # Tensor shape compatibility
    T, R, r_pupil, r_iris, L, L_p, focal, iterations = tensor_shape_compatibility(T, R, r_pupil, r_iris, L, L_p, focal, args)

    # Euler angles [deg] to rotation matrix
    Rotation = euler_to_rotation(R)

    # Generate 3D pupil and iris template pointclouds
    N_angles = args['temp_n_angles']
    N_radius = args['temp_n_radius']

    pupil_XYZ, iris_XYZ, edge_idx_iris, edge_idx_pupil = template_generation(N_angles, 
                                                                             N_radius, 
                                                                             r_pupil, 
                                                                             r_iris, L_p, 
                                                                             args=args)
    

    pupil_3D, pupil_UV, iris_3D, iris_UV = project_templates_to_2D(pupil_XYZ, iris_XYZ, Rotation, T, 
                                                      focal[:, 0], focal[:, 0], W/2, H/2)
    
    pupil_c_3D = pupil_3D[:,0,:]; pupil_c_UV = pupil_UV[:,0,:]

    eyeball_c_3D, eyeball_c_UV = extrinsics_project(torch.zeros(T.shape[0], 1, 3).to(T.device), 
                                                    Rotation, T, focal[:, 0], 
                                                    focal[:, 0], W/2, H/2)   
    eyeball_c_3D = eyeball_c_3D.squeeze(1); eyeball_c_UV = eyeball_c_UV.squeeze(1)

    # # Gaze vector is a unit vector going from the eyeball center to the pupil circle center
    # temp_gaze_vector_3D = (pupil_c_3D - eyeball_c_3D)
    # gaze_vector_3D = temp_gaze_vector_3D / \
    #             (torch.norm(temp_gaze_vector_3D, dim=-1, keepdim=True) + 1e-9)
    
    # The gaze vector in our eye model coordinates is: [0, 0, -1]
    # The gaze vector in our camera coordinates is: R @ [0, 0, -1] = -Rotation[:, 2]
    # (R|t convert from our eye coordinates to our camera coordinates)
    gaze_vector_3D = -Rotation[:, :, 2]
    # We aditionally need to flip the z-axis to convert 
    # from our camera coordiantes to the coordinate system of Wolfgang dataset
    # (Wolfgang coordinates are same as our camera coordiantes, just z is flipped to look towards the camera)
    gaze_vector_3D[:, 2] = -gaze_vector_3D[:, 2] 


    temp_gaze_vector_UV = (pupil_c_UV - eyeball_c_UV)
    gaze_vector_UV = temp_gaze_vector_UV / \
                (torch.norm(temp_gaze_vector_UV, dim=-1, keepdim=True) + 1e-9)

    #define the yaw and pitch rotation vector in radian. Mainly for NVGaze gaze vector
    rotation = 80 * R * torch.tensor([[0,1,1]]).to(R.device)
    rotation_rad = rotation * math.pi / 180
    rotation_rad = torch.stack((rotation_rad[...,2],
                                rotation_rad[...,1]),
                                axis=1)

    out_dict['T'] = T
    out_dict['R'] = R
    out_dict['r_pupil'] = r_pupil
    out_dict['r_iris'] = r_iris
    out_dict['L'] = L
    out_dict['L_p'] = L_p
    out_dict['focal'] = focal

    rend_dict = {
        #'pupil_3D': pupil_3D,
        'pupil_UV': pupil_UV,
        #'iris_3D'
        'iris_UV': iris_UV,
        'edge_idx_iris': edge_idx_iris,
        'edge_idx_pupil': edge_idx_pupil,
        #'pupil_c_3D': pupil_c_3D,
        'pupil_c_UV': pupil_c_UV,
        #'eyeball_c_3D': eyeball_c_3D,
        'eyeball_c_UV': eyeball_c_UV,
        'gaze_vector_3D': gaze_vector_3D,
        'gaze_vector_UV': gaze_vector_UV,
        'rotation_rad': rotation_rad
    }

    # # TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO
    # fig = plt.figure(figsize=plt.figaspect(1.))
    # ax_1 = fig.add_subplot(2, 2, 1)

    # i = 3

    # ax_1.imshow((data_dict['image'][i]+2)/4, alpha=0.7)

    # ax_1.scatter(pupil_UV[i, :,0].detach(), pupil_UV[i, :,1].detach(), marker='x', color='black', alpha=0.8, s=1**2)
    # ax_1.scatter(iris_UV[i, :,0].detach(), iris_UV[i, :,1].detach(), marker='o', color='green', alpha=0.5, s=1**2)
    # ax_1.scatter(pupil_UV[i, edge_idx_pupil['left'],0].detach(), pupil_UV[i, edge_idx_pupil['left'],1].detach(), marker='x', color='red', alpha=0.8, s=3**2)
    # ax_1.scatter(pupil_UV[i, edge_idx_pupil['right'],0].detach(), pupil_UV[i, edge_idx_pupil['right'],1].detach(), marker='x', color='red', alpha=0.8, s=3**2)
    # ax_1.scatter(iris_UV[i, edge_idx_iris['left'],0].detach(), iris_UV[i, edge_idx_iris['left'],1].detach(), marker='o', color='red', alpha=0.5, s=3**2)
    # ax_1.scatter(iris_UV[i, edge_idx_iris['right'],0].detach(), iris_UV[i, edge_idx_iris['right'],1].detach(), marker='o', color='red', alpha=0.5, s=3**2)
    # ax_1.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color='black')
    # ax_1.plot([0, W], [H/2, H/2], color='gray')
    # ax_1.plot([W/2, W/2], [0, H], color='black')
    # ax_1.set_xlabel('X Label')
    # ax_1.set_ylabel('Y Label')

    # ax_2 = fig.add_subplot(2, 2, 2)
    # ax_2.imshow((data_dict['image'][i]+2)/4)

    
    # eyeball_radius = data_dict['eyeball'][i][0]
    # eyebll_center_x = data_dict['eyeball'][i][1]
    # eyebll_center_y = data_dict['eyeball'][i][2]

    # circle1 = plt.Circle((eyebll_center_x, eyebll_center_y), eyeball_radius, color='r', fill=False)
    # ax_2.add_patch(circle1)
    # ax_2.scatter([eyebll_center_x], [eyebll_center_y], color='r')
    # ax_2.plot([eyebll_center_x, eyebll_center_x+eyeball_radius * data_dict['gaze_vector'][i][0]],
    #           [eyebll_center_y, eyebll_center_y+eyeball_radius * data_dict['gaze_vector'][i][1]], color='r', linewidth='2')
    # ax_2.text(0, 25, f'Gaze:{data_dict["gaze_vector"][i]}')    
    # ax_2.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color='black')
    # ax_2.plot([0, W], [H/2, H/2], color='gray')
    # ax_2.plot([W/2, W/2], [0, H], color='black')

    # ax_3 = fig.add_subplot(2, 1, 2, projection='3d')
    # ax_3 = pr.plot_basis(ax_3, R=Rotation[i].detach(), p=T[i].detach(), s = 5)
    # ax_3 = pr.plot_basis(ax_3, s = 5)

    # u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j]
    # x_sphere = L[i].detach()*np.cos(u)*np.sin(v)
    # y_sphere = L[i].detach()*np.sin(u)*np.sin(v)
    # z_sphere = L[i].detach()*np.cos(v)
    # x_sphere += T[i,0].detach().item()
    # y_sphere += T[i,1].detach().item()
    # z_sphere += T[i,2].detach().item()
    # ax_3.plot_wireframe(x_sphere, y_sphere, z_sphere, color="grey", alpha=0.1)

    # ax_3.scatter(pupil_3D[i, :, 0].detach(), pupil_3D[i, :, 1].detach(), pupil_3D[i, :, 2].detach(), marker='x', color='black', alpha=0.8, s=1**2)
    # ax_3.scatter(iris_3D[i, :, 0].detach(), iris_3D[i, :, 1].detach(), iris_3D[i, :, 2].detach(), marker='o', color='green', alpha=0.5, s=1**2)
    # ax_3.scatter(pupil_3D[i, edge_idx_pupil['left'],0].detach(), pupil_3D[i, edge_idx_pupil['left'],1].detach(), pupil_3D[i, edge_idx_pupil['left'],2].detach(), marker='x', color='red', s=3**2)
    # ax_3.scatter(pupil_3D[i, edge_idx_pupil['right'],0].detach(), pupil_3D[i, edge_idx_pupil['right'],1].detach(), pupil_3D[i, edge_idx_pupil['right'],2].detach(), marker='x', color='red', s=3**2)
    # ax_3.scatter(iris_3D[i, edge_idx_iris['left'],0].detach(), iris_3D[i, edge_idx_iris['left'],1].detach(), iris_3D[i, edge_idx_iris['left'],2].detach(), marker='o', color='red', s=3**2)
    # ax_3.scatter(iris_3D[i, edge_idx_iris['right'],0].detach(), iris_3D[i, edge_idx_iris['right'],1].detach(), iris_3D[i, edge_idx_iris['right'],2].detach(), marker='o', color='red', s=3**2)

    # # ax_2.plot([T[i,0].detach(), pupil_c_3D[i,0].detach()], [T[i,1].detach(), pupil_c_3D[i,1].detach()], [T[i,2].detach(), pupil_c_3D[i,2].detach()], color='k', linewidth=3)
    # # ax_2.plot([0.0, temp_gaze_vector_3D[i,0].detach()], [0.0, temp_gaze_vector_3D[i,1].detach()], [0.0, temp_gaze_vector_3D[i,2].detach()], color='k', linewidth=3)
    
    # R1 = np.eye(3)[None,...]
    # R2 = np.eye(3)[None,...]
    # R3 = np.eye(3)[None,...]
    # R1[:, 0, 0] = -1.0; R1[:, 1, 1] = -1.0
    # R2[:, 0, 0] = -1.0; R2[:, 2, 2] = -1.0
    # R3[:, 1, 1] = -1.0; R3[:, 2, 2] = -1.0
    # T_t = T.detach()
    # gaze_gt = data_dict['gaze_vector'].clone().detach().numpy()
    # gaze_R1 = (R1 @ Rotation[:, :, 2][..., None].clone().detach().numpy())[..., 0]
    # gaze_Rt0 = torch.transpose(Rotation, 1, 2)[:, :, 2].clone().detach().numpy()
    # ln = L[i].detach().item() * 2.0
    # ax_3.plot( [T_t[i,0] + 0.0, T_t[i,0] + ln*gaze_gt[i,0]], 
    #            [T_t[i,1] + 0.0, T_t[i,1] + ln*gaze_gt[i,1]], 
    #            [T_t[i,2] + 0.0, T_t[i,2] - ln*gaze_gt[i,2]], 
    #           color='k', linewidth=3)
    # # ax_3.plot( [T_t[i,0] + 0.0, T_t[i,0] + ln*gaze_vector_3D[i,0].detach()], 
    # #            [T_t[i,1] + 0.0, T_t[i,1] + ln*gaze_vector_3D[i,1].detach()], 
    # #            [T_t[i,2] + 0.0, T_t[i,2] + ln*gaze_vector_3D[i,2].detach()], 
    # #           color='b', linewidth=3)
    # # ax_3.plot( [T_t[i,0] + 0.0, T_t[i,0] + ln*gaze_Rt0[i,0]], 
    # #            [T_t[i,1] + 0.0, T_t[i,1] + ln*gaze_Rt0[i,1]], 
    # #            [T_t[i,2] + 0.0, T_t[i,2] + ln*gaze_Rt0[i,2]], 
    # #           color='b', linewidth=3)
    # # ax_3.plot( [T_t[i,0] + 0.0, T_t[i,0] + ln*gaze_R1[i,0]], 
    # #            [T_t[i,1] + 0.0, T_t[i,1] + ln*gaze_R1[i,1]], 
    # #            [T_t[i,2] + 0.0, T_t[i,2] + ln*gaze_R1[i,2]], 
    # #           color='r', linewidth=3)
              
    # ax_3.set_xlabel('X Label')
    # ax_3.set_ylabel('Y Label')
    # ax_3.set_zlabel('Z Label')
    # ax_3.view_init(-45, 90) # Initial viewing angle

    
    # plt.savefig('tmp.jpg', bbox_inches='tight')  

    # # TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO  TODO

    return out_dict, rend_dict


def scale_and_bound(T, R, r_pupil, r_iris, L, focal, args):
    '''
    Scale and bound predictions
    '''

    # found on google
    # iris radius 4.5-7.5
    # pupil radius 1-4.5
    # eyeball radius 9-15
    # for an average eye risi/eyeball diameter ratio ~0.4
    if args ['scale_bound_eye'] == 'version_0':
        #eyeball radius
        L = 5.0 * L + 12.0

        #pupil raidus
        r_pupil = 4.0 * r_pupil + 4.0
        #iris radius
        r_iris = 4.0 * r_iris + 5.5

        L_p = torch.sqrt(torch.abs(L**2 - r_iris**2))

        # Eyeball center
        scale = torch.tensor([[[17.0, 17.0, 60.0]]]).to(T.device)
        offset = torch.tensor([[[0.0, 0.0, 55.0]]]).to(T.device)
        T = scale * T + offset
        #rotation angle in range of [-80.0,80.0] degrees
        # TODO since we use the ZYX change the location of roll
        R = 80.0 * R * (torch.tensor([[0, 1, 1]])).to(R.device)

        #focal length
        focal = 750.0 * focal + 750.0

    elif args ['scale_bound_eye'] == 'version_0_1':
        #eyeball radius
        L = 3.5 * L + 12.5

        #pupil raidus
        r_pupil = 2.0 * r_pupil + 3.0
        #iris radius
        r_iris = 2.0 * r_iris + 6.0

        L_p = torch.sqrt(torch.abs(L**2 - r_iris**2))

        # Eyeball center
        scale = torch.tensor([[[20.0, 20.0, 60.0]]]).to(T.device)
        offset = torch.tensor([[[0.0, 0.0, 60.0]]]).to(T.device)
        T = scale * T + offset
        #rotation angle in range of [-80.0,80.0] degrees
        # TODO since we use the ZYX change the location of roll
        R = 80.0 * R * (torch.tensor([[0, 1, 1]])).to(R.device)

        #focal length
        focal = 800.0 * focal + 800.0    
        
    elif args ['scale_bound_eye'] == 'version_1':
        #eyeball radius
        L = 3.5 * L + 12.5

        #iris radius
        r_i_min = L*0.3
        r_i_max = L*0.55
        r_iris = r_i_min + (r_i_max - r_i_min) * ((r_iris+1.0)/2.0)

        #pupil raidus
        r_p_min = 1.0
        r_p_max = r_iris
        r_pupil = r_p_min + (r_p_max - r_p_min) * ((r_pupil+1.0)/2.0)

        L_p = torch.sqrt(torch.abs(L**2 - r_iris**2))

        # Eyeball center
        scale = torch.tensor([[[20.0, 20.0, 60.0]]]).to(T.device)
        offset = torch.tensor([[[0.0, 0.0, 60.0]]]).to(T.device)
        T = scale * T + offset
        #rotation angle in range of [-80.0,80.0] degrees
        # TODO since we use the ZYX change the location of roll
        R = 80.0 * R * (torch.tensor([[0, 1, 1]])).to(R.device)

        #focal length
        focal = 600.0 * focal + 800.0

    else:
        raise NotImplementedError


    return T, R, r_pupil, r_iris, L, L_p, focal


def tensor_shape_compatibility(T, R, r_pupil, r_iris, L, L_p, focal, args):  
    '''Tensor shape compatibility'''
    NUM_OF_BATCHES = args['batch_size']
    NUM_OF_FRAME = args['frames']
    iterations = NUM_OF_BATCHES * NUM_OF_FRAME

    # TODO: Remove later
    T = T.expand(-1, NUM_OF_FRAME, -1)
    L = L.expand(-1, NUM_OF_FRAME)
    L_p = L_p.expand(-1, NUM_OF_FRAME)
    focal = focal.expand(-1, NUM_OF_FRAME, -1)
    r_iris = r_iris.expand(-1, NUM_OF_FRAME)

    # TODO: Remove later (after removing for loop)
    T = torch.flatten(T, end_dim=1)
    focal = torch.flatten(focal, end_dim=1)
    L = torch.flatten(L, end_dim=1)
    L_p = torch.flatten(L_p, end_dim=1)
    r_iris = torch.flatten(r_iris, end_dim=1)
    r_pupil = torch.flatten(r_pupil, end_dim=1)
    R = torch.flatten(R, end_dim=1)

    return T, R, r_pupil, r_iris, L, L_p, focal, iterations


def euler_to_rotation(R_deg):
    #Rotation Matrix of pupil using the angle 
    #Convert degrees to radian
    R_radian = R_deg * math.pi / 180

    #Fix the coordinate system to bring the iris and pupil in front of the camera
    Rotation = transforms.euler_angles_to_matrix(R_radian, "ZYX") 

    return Rotation


# code from this github repo : https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
def vectorized_linspace(start, end, steps):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(end.device)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(end.device)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size).to(end.device)
    end = end.contiguous().view(view_size).expand(out_size).to(end.device)

    out = start_w * start + end_w * end

    return out


def template_edge_indexes(angles, N_radius):
    assert angles.ndim == 2
    assert isinstance(N_radius, int)

    B, N_angles = angles.shape

    edge_idx_iris = {}
    edge_idx_pupil = {}

    # Extract iris edge angle positions
    edge_iris_left = (angles[0]>=math.pi*0.9) * (angles[0]<=math.pi*1.1) # Left
    edge_iris_right = (angles[0]<=math.pi*0.1) + (angles[0]>=math.pi*1.9) # Right
    edge_idx_iris['left'] = (N_radius-1)*N_angles + torch.where(edge_iris_left)[0]
    edge_idx_iris['right'] = (N_radius-1)*N_angles + torch.where(edge_iris_right)[0]
    edge_idx_iris['sides'] = torch.cat((edge_idx_iris['left'], edge_idx_iris['right']))
    edge_idx_iris['outline'] = (N_radius-1)*N_angles + torch.arange(N_angles)

    #Extract pupil edge angle postion (Take all circle points)
    edge_pupil_left = (angles[0]>math.pi*0.50) * (angles[0]<=math.pi*1.50)
    edge_pupil_right = (angles[0]<=math.pi*0.50) + (angles[0]>math.pi*1.50)
    edge_idx_pupil['left'] = (N_radius-1)*N_angles + torch.where(edge_pupil_left)[0]
    edge_idx_pupil['right'] = (N_radius-1)*N_angles + torch.where(edge_pupil_right)[0]
    edge_idx_pupil['sides'] = torch.cat((edge_idx_pupil['left'], edge_idx_pupil['right']))
    edge_idx_pupil['outline'] = (N_radius-1)*N_angles + torch.arange(N_angles)

    return edge_idx_iris, edge_idx_pupil


def template_generation(N_angles, N_radius, r_pupil, r_iris, L_p, args):
    assert isinstance(N_angles, int)
    assert isinstance(N_radius, int)
    assert r_pupil.ndim == r_iris.ndim == L_p.ndim == 1
    assert r_pupil.shape[0] == r_iris.shape[0] == L_p.shape[0]

    B = args['batch_size'] * args['frames']
    device = r_pupil.device

    # Angles [0, 2*pi) [B, N_ang]
    angles = np.linspace([0]*B, [1.9999*math.pi]*B, N_angles, 
                        axis=-1, dtype=np.float32)
    angles = torch.from_numpy(angles).to(device)

    # Get edge point indices
    edge_idx_iris, edge_idx_pupil = template_edge_indexes(angles, N_radius) 
    
    # Pupil radius (circle)
    radius_pupil = vectorized_linspace(torch.zeros(B), r_pupil, N_radius)

    # Iris radius (ring)
    radius_iris = vectorized_linspace(r_pupil, r_iris, N_radius)

    # Adjust shape 
    angles = rearrange(angles, 'b n -> b 1 n')

    radius_pupil = rearrange(radius_pupil, 'b n -> b n 1')
    radius_iris = rearrange(radius_iris, 'b n -> b n 1')

    # Pupil template pointcloud
    pupil_X = rearrange((radius_pupil * torch.cos(angles)), 'b n1 n2 -> b (n1 n2)')
    pupil_Y = rearrange((radius_pupil * torch.sin(angles)), 'b n1 n2 -> b (n1 n2)')
    pupil_Z = rearrange(L_p, 'b -> b 1') * torch.ones_like(pupil_Y).to(device)
    pupil_Z *= -1.0 # So that it ends up in front of camera after [R|T]

    pupil_XYZ = torch.stack((pupil_X, pupil_Y, pupil_Z), dim=-1)
    
    # Iris template pointcloud
    iris_X = rearrange((radius_iris * torch.cos(angles)), 'b n1 n2 -> b (n1 n2)')
    iris_Y = rearrange((radius_iris * torch.sin(angles)), 'b n1 n2 -> b (n1 n2)')
    iris_Z = rearrange(L_p, 'b -> b 1') * torch.ones_like(pupil_Y).to(device)
    iris_Z *= -1.0 # So that it ends up in front of camera after [R|T]

    iris_XYZ = torch.stack((iris_X, iris_Y, iris_Z), dim=-1)

    return pupil_XYZ, iris_XYZ, edge_idx_iris, edge_idx_pupil


def extrinsics_project(P, R, T, fx, fy, cx, cy):
    assert 1 == fx.ndim == fy.ndim
    assert T.ndim == 2
    assert R.ndim == 3
    assert P.ndim == 3
    assert isinstance(cx, float)
    assert isinstance(cy, float)

    # Extrinsic transformation
    P_3D = (R @ rearrange(P, 'b n d -> b d n')) + rearrange(T, 'b d -> b d 1')
    P_3D = rearrange(P_3D, 'b d n -> b n d')
    # Pinhole projection
    # * 1.0 so that P_3D elements are not changed from operations on UV
    # (otherwise UV would be a pointer to P_3D)
    UV = P_3D * 1.0 

    UV[..., 0] *= fx.unsqueeze(-1)
    UV[..., 1] *= fy.unsqueeze(-1)
    UV = UV[..., :2] / (UV[..., 2:] + 1e-9)
    UV[..., 0] += cx
    UV[..., 1] += cy

    return P_3D, UV


def project_templates_to_2D(P_pupil, P_iris, R, T, fx, fy, cx, cy):
    # K = torch.tensor([[fx,  0.0, cx], 
    #                   [0.0, fy,  cy], 
    #                   [0.0, 0.0, 1.0]], requires_grad=True).to(fx.device)


    Pupil_3D, UV_pupil = extrinsics_project(P_pupil, R, T, fx, fy, cx, cy)
    
    Iris_3D, UV_iris = extrinsics_project(P_iris, R, T, fx, fy, cx, cy)

    return Pupil_3D, UV_pupil, Iris_3D, UV_iris


def eyeball_center(in_dict, H, W, args):

    T=in_dict['T'] 
    R=in_dict['R']
    r_pupil=in_dict['r_pupil'] 
    r_iris=in_dict['r_iris']
    L=in_dict['L'] 
    focal=in_dict['focal']

    cx = W/2
    cy = H/2
    fx = focal[:, 0]
    fy = focal[:, 0]

    B = args['batch_size'] * args['frames']
    device = r_pupil.device

    # Euler angles [deg] to rotation matrix
    Rotation = euler_to_rotation(R)

    #generate template for eyeball_circle
    # Angles [0, 2*pi) [B, N_ang]
    N_angles = 100
    angles = np.linspace([0]*B, [1.9999*math.pi]*B, N_angles, 
                        axis=-1, dtype=np.float32)
    angles = torch.from_numpy(angles).to(device)

    L = rearrange(L, 'b -> b 1')
    T_circle = rearrange(T, 'b n -> b 1 n')

    eyeball_X = L * torch.cos(angles)
    eyeball_Y = L * torch.sin(angles)
    eyeball_Z = torch.zeros_like(eyeball_X)

    eyeball_XYZ = torch.stack((eyeball_X, eyeball_Y, eyeball_Z), dim=-1)

    eyeball_XYZ += T_circle

    temp_UV = eyeball_XYZ

    temp_UV[...,0] *= fx.unsqueeze(-1)
    temp_UV[...,1] *= fy.unsqueeze(-1)

    eyeball_UV = temp_UV[...,:2]/temp_UV[...,2:]

    eyeball_UV[...,0] += cx
    eyeball_UV[...,1] += cy

    return eyeball_UV[:,1:,:]

def eye_model_visualize(T1, T2, T3, R1, R2, R3, r_pupil, r_iris, L, fx, fy):
    # Simulate NN output
    W = 640
    H = 480
    T = torch.tensor([[[T1, T2, T3]]])
    R = torch.tensor([[[R1, R2, R3]]])
    r_pupil = torch.tensor([[r_pupil]])
    r_iris = torch.tensor([[r_iris]])
    L = torch.tensor([[L]])
    focal = torch.tensor([[[fx, fy]]])

    args = {} 
    args['frames'] = 1
    args['batch_size'] = 1

    # Scale and bound predictions
    T, R, r_pupil, r_iris, L, L_p, focal = scale_and_bound(T, R, r_pupil, r_iris, L, focal)

    # Tensor shape compatibility
    T, R, r_pupil, r_iris, L, L_p, focal, iterations = tensor_shape_compatibility(T, R, r_pupil, r_iris, L, L_p, focal, args)

    # Pupil and iris template
    N_angles = 1
    N_radius = 1
    pupil_XYZ, iris_XYZ, edge_idx_iris, edge_idx_pupil = template_generation(N_angles, 
                                                                             N_radius, 
                                                                             r_pupil, 
                                                                             r_iris, 
                                                                             L_p,
                                                                             args=args)
    pupil_XYZ_i = pupil_XYZ[0]
    iris_XYZ_i = iris_XYZ[0] 

    # 3D sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j]
    x_sphere = L[0]*np.cos(u)*np.sin(v)
    y_sphere = L[0]*np.sin(u)*np.sin(v)
    z_sphere = L[0]*np.cos(v)

    if False:
        #Plot everything before changing to camera coordinates
        fig = plt.figure(figsize=plt.figaspect(2.))

        ax_0 = fig.add_subplot(2, 1, 1)
        ax_0.scatter(pupil_XYZ_i[:, 0].detach(), pupil_XYZ_i[:, 1].detach(), marker='x', color='black')
        ax_0.scatter(iris_XYZ_i[:, 0].detach(), iris_XYZ_i[:, 1].detach(), marker='o', color='green')
        ax_0.scatter(pupil_XYZ_i[edge_idx_pupil, 0].detach(), pupil_XYZ_i[edge_idx_pupil, 1].detach(), marker='x', color='red')
        ax_0.scatter(iris_XYZ_i[edge_idx_iris, 0].detach(), iris_XYZ_i[edge_idx_iris, 1].detach(), marker='o', color='red')
        ax_0.set_xlabel('X Label')
        ax_0.set_ylabel('Y Label')

        ax_1 = fig.add_subplot(2, 1, 2, projection='3d')
        ax_1 = pr.plot_basis(ax_1, s = 5)
        ax_1.plot_wireframe(x_sphere, y_sphere, z_sphere, color="grey", alpha=0.1)
        ax_1.scatter(pupil_XYZ_i[:,0].detach(), pupil_XYZ_i[:,1].detach(), pupil_XYZ_i[:,2].detach(), marker='x', color='black', alpha=0.6)
        ax_1.scatter(iris_XYZ_i[:,0].detach(), iris_XYZ_i[:,1].detach(), iris_XYZ_i[:,2].detach(), marker='o', color='green', alpha=0.1)
        ax_1.scatter(pupil_XYZ_i[edge_idx_pupil,0].detach(), pupil_XYZ_i[edge_idx_pupil,1].detach(), pupil_XYZ_i[edge_idx_pupil,2].detach(), marker='x', color='red')
        ax_1.scatter(iris_XYZ_i[edge_idx_iris,0].detach(), iris_XYZ_i[edge_id_iris,1].detach(), iris_XYZ_i[edge_idx_iris,2].detach(), marker='o', color='red')
        ax_1.set_xlabel('X Label')
        ax_1.set_ylabel('Y Label')
        ax_1.set_zlabel('Z Label')
        plt.show()

    # Euler angles [pip install ipympldeg] to rotation matrix
    Rotation = euler_to_rotation(R)

    # Project 3D template to 2D image frame
    pupil_point3D, pupil_UV, iris_point3D, iris_UV = project_templates_to_2D(
                        pupil_XYZ, iris_XYZ, Rotation, T, 
                        focal[:, 0], focal[:, 0], W/2, H/2)
    pupil_UV_i = pupil_UV[0]
    iris_UV_i = iris_UV[0]
    pupil_point3D = pupil_point3D[0]
    iris_point3D = iris_point3D[0]

    #estimate the gaze point
    #initial_gaze = torch.tensor([0, 0, -1], dtype=torch.float)
    #print(initial_gaze)
    #pd_gaze_vec = Rotation @ initial_gaze
    #pd_gaze_vec = pd_gaze_vec * 100
    #print('after rotation')
    #print(pd_gaze_vec)
                    
    #pd_gaze_norm = torch.linalg.norm(pd_gaze_vec, dim=1, keepdims=True)
    #pd_gaze_vec_normalized = pd_gaze_vec / pd_gaze_norm
    
    #Plot everything before changing in camera coordinates
    #fig = plt.figure()
    fig = plt.figure(figsize=plt.figaspect(2.))
    ax_0 = fig.add_subplot(2, 1, 1)

    ax_0.scatter(pupil_UV_i[:,0].detach(), pupil_UV_i[:,1].detach(), marker='x', color='black', alpha=0.8, s=1**2)
    ax_0.scatter(iris_UV_i[:,0].detach(), iris_UV_i[:,1].detach(), marker='o', color='green', alpha=0.5, s=1**2)
    ax_0.scatter(pupil_UV_i[edge_idx_pupil['left'],0].detach(), pupil_UV_i[edge_idx_pupil['left'],1].detach(), marker='x', color='red', alpha=0.8, s=3**2)
    ax_0.scatter(pupil_UV_i[edge_idx_pupil['right'],0].detach(), pupil_UV_i[edge_idx_pupil['right'],1].detach(), marker='x', color='red', alpha=0.8, s=3**2)
    ax_0.scatter(iris_UV_i[edge_idx_iris['left'],0].detach(), iris_UV_i[edge_idx_iris['left'],1].detach(), marker='o', color='red', alpha=0.5, s=3**2)
    ax_0.scatter(iris_UV_i[edge_idx_iris['right'],0].detach(), iris_UV_i[edge_idx_iris['right'],1].detach(), marker='o', color='red', alpha=0.5, s=3**2)
    ax_0.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color='black')
    ax_0.plot([0, W], [H/2, H/2], color='gray')
    ax_0.plot([W/2, W/2], [0, H], color='black')
    ax_0.set_xlabel('X Label')
    ax_0.set_ylabel('Y Label')

    ax_1 = fig.add_subplot(2, 1, 2, projection='3d')
    ax_1 = pr.plot_basis(ax_1, s = 5)
    x_sphere += T[0,0].item()
    y_sphere += T[0,1].item()
    z_sphere += T[0,2].item()
    ax_1.plot_wireframe(x_sphere, y_sphere, z_sphere, color="grey", alpha=0.1)

    ax_1.scatter(pupil_point3D[:, 0].detach(), pupil_point3D[:, 1].detach(), pupil_point3D[:, 2].detach(), marker='x', color='black', alpha=0.8, s=1**2)
    ax_1.scatter(iris_point3D[:, 0].detach(), iris_point3D[:, 1].detach(), iris_point3D[:, 2].detach(), marker='o', color='green', alpha=0.5, s=1**2)
    ax_1.scatter(pupil_point3D[edge_idx_pupil['left'],0].detach(), pupil_point3D[edge_idx_pupil['left'],1].detach(), pupil_point3D[edge_idx_pupil['left'],2].detach(), marker='x', color='red', s=3**2)
    ax_1.scatter(pupil_point3D[edge_idx_pupil['right'],0].detach(), pupil_point3D[edge_idx_pupil['right'],1].detach(), pupil_point3D[edge_idx_pupil['right'],2].detach(), marker='x', color='red', s=3**2)
    ax_1.scatter(iris_point3D[edge_idx_iris['left'],0].detach(), iris_point3D[edge_idx_iris['left'],1].detach(), iris_point3D[edge_idx_iris['left'],2].detach(), marker='o', color='red', s=3**2)
    ax_1.scatter(iris_point3D[edge_idx_iris['right'],0].detach(), iris_point3D[edge_idx_iris['right'],1].detach(), iris_point3D[edge_idx_iris['right'],2].detach(), marker='o', color='red', s=3**2)
    ax_1.set_xlabel('X Label')
    ax_1.set_ylabel('Y Label')
    ax_1.set_zlabel('Z Label')
    ax_1.view_init(-45, -90) # Initial viewing angle

    # print(f'Rotation=[{R[0, 0]}, {R[0, 1]}, {R[0, 2]}]')
    # print(f'Translation=[{T[0, 0]}, {T[0, 1]}, {T[0, 2]}]')
    # print(f'r_pupil=[{r_pupil[0]}]')
    # print(f'r_iris=[{r_iris[0]}]')
    # print(f'L=[{L[0]}]')
    # print(f'focal=[{focal[0, 0]}, {focal[0, 1]}]')

    plt.savefig('eye_model_vis.jpg')
    plt.show()


if __name__ == "__main__":
    # #specify the param of the eye model for the training 
    # gt_T = torch.tensor([[0.0, 0.0, 58.0],]).to(device)
    # gt_R = torch.tensor([[-3.5, -13.0, 0.0],]).to(device)
    # gt_r_pupil = torch.tensor([[1.1],]).to(device)
    # gt_r_iris = torch.tensor([[6.5],]).to(device)
    # gt_L = torch.tensor([[10.1],]).to(device)
    # gt_focal = torch.tensor([[370.0, 600.0],]).to(device)
    #
    T1 = 0.; T2 = 0.; T3 = 0.
    R1 = 0.; R2 = 0.; R3 = 0.
    r_pupil = 0.
    r_iris = 0.
    L = 0.
    fx = 0.; fy=0.
    args = {}
    args['batch_size'] = 1
    args['frames'] = 1
    eye_model_visualize(T1, T2, T3, R1, R2, R3, r_pupil, r_iris, L, fx, fy)

    device = 'cpu'

    img = cv2.imread("/home/dchristodoul/deeplearning/3d_input/0.png", 0)
    label = np.load("/home/dchristodoul/deeplearning/3d_input/0.npy")

    r = np.where(label)[0]
    c = int(0.5*(np.max(r) + np.min(r)))
    top, bot = (0, c+150-(c-150)) if c-150<0 else (c-150, c+150)

    img = img[top:bot, :]
    label = label[top:bot, :]

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LANCZOS4)
    label = cv2.resize(label, (640, 480), interpolation=cv2.INTER_LANCZOS4)

    gt_dict = {}
    label_tensor = torch.tensor(label).unsqueeze(0).unsqueeze(0)
    gt_dict['mask'] = label_tensor.to(device)

    T = torch.tensor([[[0.0, 0.0, -0.5]]]).cpu()
    R = torch.tensor([[[0.0, -0.15, -0.08]]]).cpu()
    r_pupil = torch.tensor([[-0.8]]).cpu()
    r_iris = torch.tensor([[0.5]]).cpu()
    L = torch.tensor([[1.0]]).cpu()
    focal = torch.tensor([[[0.0, 0.2]]]).cpu()

    eyeball_param = {
        'T': T,
        'R': R,
        'r_pupil': r_pupil,
        'r_iris': r_iris,
        'L': L,
        'focal': focal
    }

    W = 640
    H = 480

    plt.clf()

    rend, loss = render_semantics(eyeball_param, H, W, args = args) 