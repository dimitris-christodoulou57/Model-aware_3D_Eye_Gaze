import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import copy
import math
import cv2
import os
#import wandb

from PIL import Image
from skimage import draw
from itertools import chain
from skimage.transform import rescale
from skimage.segmentation.boundaries import find_boundaries
from scipy.ndimage import distance_transform_edt as distance


EPS = 1e-40


def convert_to_list_entries(data_dict):
    # Move everything back to numpy if it exists in torch

    for key, item in data_dict.items():
        if 'torch' in str(type(item)):
            data_dict[key] = item.detach().cpu().squeeze().numpy()

    # Generate empty template based on dictionary
    num_entries = data_dict[key].shape[0]
    out = []
    for ii in range(num_entries):
        out.append({key:item[ii] for key, item in data_dict.items() if 'numpy' in str(type(item))})
    return out


def draw_annots_on_image(data_dict,
                         remove_saturated=True,
                         intensity_maps=100,
                         pupil_index=2,
                         iris_index=1,
                         is_predict=True,
                         rendering=False,
                         image_id = 0):

    image = data_dict['image'][image_id]
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
        if is_predict or data_dict['pupil_center_available'][image_id]:
            pupil_center = data_dict['pupil_center'][image_id]
            [rr, cc] = draw.disk((pupil_center[1].clip(6, height-6),
                                pupil_center[0].clip(6, width-6)),
                                radius=5)
            out_image[rr, cc, :] = 255

        if is_predict or data_dict['pupil_ellipse_available'][image_id]:
            pupil_ellipse = data_dict['pupil_ellipse'][image_id]
            # print("Pupil: {}".format(type(pupil_ellipse)))
            loc_pupil = data_dict['mask'][image_id] == pupil_index
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

        if is_predict or data_dict['iris_ellipse_available'][image_id]:
            iris_ellipse = data_dict['iris_ellipse'][image_id]
            # print("Iris: {}".format(type(iris_ellipse)))
            loc_iris = data_dict['mask'][image_id] == iris_index
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
        loc_pupil = data_dict['mask'][image_id] == pupil_index
        out_image[..., 0] +=  (intensity_maps*loc_pupil*loc_image_non_sat).astype(np.uint8)
        out_image[..., 1] +=  (intensity_maps*loc_pupil*loc_image_non_sat).astype(np.uint8)

        loc_iris = data_dict['mask'][image_id] == iris_index
        out_image[..., 1] +=  (intensity_maps*loc_iris*loc_image_non_sat).astype(np.uint8)

    return out_image.astype(np.uint8)


def plot_images_with_annotations(data_dict,
                                 args,
                                 show=True,
                                 write=None,
                                 rendering=False,
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

    idx = 0
    """plot some images to wandb
    class_labels = {1: 'iris',2: 'pupil',3: 'scelra',4: 'background'}
    if mode == 'train':
        wandb.log({"training/my_image_key": wandb.Image(list_data_dict[idx]['image'], masks = {"predictions":{"mask_data": list_data_dict[idx]['mask'], "class_labels": class_labels}}), 'epoch': epoch, 'batch': batch})
    elif mode == 'valid':
        wandb.log({"valid/my_image_key": wandb.Image(list_data_dict[idx]['image'], masks = {"predictions":{"mask_data": list_data_dict[idx]['mask'], "class_labels": class_labels}}), 'epoch': epoch, 'batch': batch})
    elif mode == 'test':
        wandb.log({"testing/my_image_key": wandb.Image(list_data_dict[idx]['image'], masks = {"predictions":{"mask_data": list_data_dict[idx]['mask'], "class_labels": class_labels}}), 'epoch': epoch, 'batch': batch}) """

    """ plt.axis('off')
    ### save images
    for i in range(10):
        print('here')
        img = list_data_dict[0]['image'][i]
        img = img - img.min()
        img = (255*(img/img.max())).astype(np.uint8)
        out_image = np.stack([img]*3, axis=2)
        name = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/image_{}.jpg'.format(i)
        plt.imshow(out_image)
        plt.savefig(name, bbox_inches='tight')
    plt.axis('on') """

    #problem here is the value of rows and cols when run the code to desktop
    #the batch size is 1 so the rows and cols arent good numbers
    fig, axs = plt.subplots(rows, cols, squeeze=True)

    for i in range(rows):
        for j in range(cols):
            if (idx < num_entries):
                # Only plot entries within the range of list
                if plot_annots:
                    out_image = draw_annots_on_image(list_data_dict[0],
                                                     is_predict=is_predict,
                                                     rendering=rendering,
                                                     intensity_maps=100,
                                                     remove_saturated=remove_saturated,
                                                     image_id = idx)
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