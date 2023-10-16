# Model-aware_3D_Eye_Gaze

**[ETH ZURICH, Computer Vision Lab (CVL)](https://vision.ee.ethz.ch)**

Nikola Popovic, Dimitrios Christodoulou, Danda Pani Paudel, Xi Wang, Luc Van Gool

### Abstract

The task of predicting 3D eye gaze from eye images can be performed either by (a) end-to-end learning for image-to-gaze mapping or by (b) fitting a 3D eye model onto images. The former case requires 3D gaze labels, while the latter requires eye semantics or landmarks to facilitate the model fitting. Although obtaining eye semantics and landmarks is relatively easy, fitting an accurate 3D eye model on them remains very challenging due to its ill-posed nature in general. On the other hand, obtaining large-scale 3D gaze data is cumbersome due to the required hardware setups and computational demands.

In this work, we propose to predict 3D eye gaze from weak supervision of eye semantic segmentation masks and direct supervision of a few 3D gaze vectors. The proposed method combines the best of both worlds by leveraging large amounts of weak annotations--which are easy to obtain, and only a few 3D gaze vectors--which alleviate the difficulty of fitting 3D eye models on the semantic segmentation of eye images. Thus, the eye gaze vectors, used in the model fitting, are directly supervised using the few-shot gaze labels. Additionally, we propose a transformer-based network architecture, that serves as a solid baseline for our improvements. Our experiments in diverse settings illustrate the significant benefits of the proposed method, achieving about 5 degrees lower angular gaze error over the baseline, when only 0.05% 3D annotations of the training images are used.

[['Paper']()][['Dataset'](https://www.hci.uni-tuebingen.de/publications/fuhl2021teyed/)][['BibTex'](#citing-this-work)]

![Desing](images/teaser.jpg)


## Installation

The code requires `python>=3.7`, as well as PyTorch and TorchVision. The instructions to install both, PyTorch and TorchVision, can be found [here](https://pytorch.org/get-started/locally). More information about the dependencies installation can be found in the previous link too. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

The following part gives more guidance for the installation and conda environment creation.

```
conda activate -n 3d_gaze python=3.7.1

pip install ipython ipykernel
pip install ipympl
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install h5py
pip install -U scikit-learn
pip install scikit-image
pip install matplotlib
pip install opencv-python
pip install tensorboardX
pip install imgaug
pip install faiss-gpu
pip install pytorch3d
pip install pytransform3d
pip install einops
pip install wandb
```

## <a name="GettingStarted"></a>Getting Started

First download the code, and the dataset. 

After downloading the dataset, the data generation process should be completed. To extract the necessary images for the training and testing for the video recording the Python file /data_generation/Extract_TEyeD.py should be run. After processing the recordings, the data split has to be performed. The specifications of the recording's names that are used for training and testing are mentioned in the /cur_obj/datasetSelection.py. 

There are two different ways to prepare the training, validation and testing dataloader. You can either generate a pkl file containing the necessary information, in case you want to have exactly the same split and avoid wasting time in preparation, or set the use_pkl_for_dataload = False to process the data each time. To generate that pkl file you need to run the following python file cur_obj/createDataloaders_baseline.py. You can find our data split as a pkl file as cur_objs/dataDiv_obj_test.pkl, and the pickle load commands should be uncommented in main.py, to load the paper results.

The models folder contains a variety of implementations for the backbone of our framework. Moreover, the implementation of our 3D eye model can be found in the rendering folder. Finally, you can customize and change arguments in args_maker.py file, like the optimizer, learning rates, weight decay, weight for losses, activate different heads (segmentation or rendering) etc. 

To be able to run and train/validating/testing our code the following command must to executed. You have to adjust the other arguments, to achieve the desirable results. The entry script is the run.py. 

```
python -u run.py \
--path_data='.../Datasets/All' \
--path2MasterKey='.../Datasets/MasterKey' \
--path_exp_tree='.../Results' \
--model='res_50_3' \
--cur_obj='TEyeD' \
--exp_name='3D_eye_framework'
```

## <a name="Models"></a>Model Checkpoints

Moreover, our code gives you the chance to save the model's weights to be able to continue training, validating only, or testing only. To activate one of these features, the following arguments should be chances respectively, continue training by specifying the path to weights using the weights_path argument, only_valid=1, only_test=1.

You can download the pre-trained networks from the following table. 

| Model | Download |
| ------------------- | :------------------: |
| Gaze (Ours)         | [download](https://drive.google.com/drive/folders/12LZP_o-huu8bBCK082RY7XhtxrtR12R0?usp=share_link) |
| Segmentation        | [download](https://drive.google.com/drive/folders/1oEk-ORefO3o8HStWjDL9Ttmll2sdjFRS?usp=share_link) |
| Segmentation + Gaze + Center | [download](https://drive.google.com/drive/folders/1THcA80u1llQvplL_V5Q510YrsCag7_BS?usp=share_link) |


```
python -u run.py \
--path_data='.../Datasets/All' \
--path2MasterKey='.../Datasets/MasterKey' \
--path_exp_tree='.../Results' \
--weiths_path='.../pretrained/all_losses/results/last.pt' \
--model='res_50_3' \
--exp_name='pretrained_sem'
```

## Dataset - TEyeD

See [here](https://www.hci.uni-tuebingen.de/assets/pdf/publications/fuhl2021teyed.pdf) for an overview of the dataset. The dataset can be downloaded from a FTP server. Just connect via FTP as user TEyeDUser and without password to nephrit.cs.uni-tuebingen.de (ftp://nephrit.cs.uni-tuebingen.de).

The TEyeD is the world's largest unified public dataset of eye images captured at close distances using seven head-mounted eye trackers. The TEyeD contains the 2D & 3D segmentation masks, the pupil center, the 2D & 3D landmarks, the position and the radius of the eyeball, the gaze vector, and the eye moment. The following image shows example images from the dataset with annotations. 

![Dataset](images/dataset.jpg)

TEyeD does not have a predefined data split, so we randomly select around 348K images for training and approximately 36K images for testing from the Dikablis folder (path /TEyeDSSingleFiles/Dikablis). The dataloader requires two different folders created on the data contained in the previous folder, namely 'All' and 'Masterkey'. To generate these folders the /data_generation/Extract_TEyeD.py should be executed. The /cur_obj/datasetSelection.py is used to specify the date split, training and testing. To be more precise, the different recordings for each split are specified. Temporal downsampling is applied to reduce the frame rate from 25 Hz to 6.25 Hz so that there are significant eye movements and to avoid identical eye images.


## License

The MIT License

Copyright (c) 2023 Dimitrios Christodoulou, Nikola Popovic, Danda Pani Paudel, Xi Wang, Luc Van Gool

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, and Gabriel Diaz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citing This Work

Please cite as:

```
@inproceedings{modelaware3deye,
  title = {Model-aware 3D Eye Gaze from Weak and Few-shot Supervisions},
  author = {Nikola Popovic and Dimitrios Christodoulou and Danda Pani Paudel and Xi Wang and Luc Van Gool},
  booktitle = {ISMAR '23},
  year = {2023},
}
```
