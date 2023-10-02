
# ARS-MAGSAC : Adaptive Reordering Sampler with Neurally Guided MAGSAC
>:newspaper: The code for ICCV 2023 paper, [Adaptive Reordering Sampler with Neurally Guided MAGSAC](https://arxiv.org/abs/2212.13185) (ARS-MAGSAC).
>
>Tong Wei, Jiri Matas, Daniel Barath.

>:loudspeaker: The AR-Sampler can be easily tried in C++ repo of [MAGSAC++](http://github.com/danini/magsac) by choose the 4th option of sampler.

>:file_folder: Download the data from [ars_data](https://cmp.felk.cvut.cz/~weitong/ars_data.zip).


- [Introduction](#introduction)
- [Environments](#environments)
- [Code Structure](#code-structure)
- [Data Structure](#data-structure)
- [Training](#training)
- [Testing](#testing)
- [Demo](#demo)
- [Citation](#citation)


## Introduction
ARS-MAGSAC is a new learning-based robust estimator proposed for two-view epipolar geometry estimation. 
We propose a new sampler guided by deep pirors, updates the inlier probabilties among RANASC iterations.
The deep pirors are learned from the coordinates, side information (SNN ratio), scale and orientation features.

## Environments
Install MAGSAC++ in Python,
```
git clone -b ngmagsac-test http://github.com/danini/magsac
mkdir build
cmake ..
make
cd ..
python setup.py install

git clone -b ngmagsac-base http://github.com/danini/graph-cut-ransac
```
Training requires the following python packages:
```
Python (3.7.11)
PyTorch (1.7.0)
OpenCV (3.4.2)
tensorboardX
```
(optional) Install the initial training module, which refers to [NG-RANSAC](https://github.com/vislearn/ngransac).
```
cd init
python setup.py install
```
## Code Structure

-------

[images](images/) -- folder contains the images used for demo  
[models](models/) -- folder includes the pretrained models, and initial trained model  
[networks](networks/) -- folder contains possible training networks   
[demo](demo.py) -- demo to try ARS-MAGSAC on examples   
[train](train_pymagsac.py) -- main training    
[init](init_pymagsac.py) -- initia trainiing  
[test](test_pymagsac.py) -- batch testing of ARS-MAGSAC  
[dataset](dataset.py) -- customized data loader  
[utils](util.py) -- utility functions and parameters.

-------

## Data Structure

You can download the [datasets](https://cmp.felk.cvut.cz/~weitong/ars_data.zip) and put them under the folder [traindata](traindata/).
Each data file structure is as follows,
```
traindata/<dataset_name>/<data_variant>/<image_pair>.npy
```
The datasets are dumped and saved in 'npy'. Each contains the coordinates, SNN ratios, image sizes, calibration matrices, ground-truth rotation matrices, 
the scales and orientations from SIFT features.
```python
[pts1, pts2, sideinfo, img1size, img2size, K1, K2, R, t, size1, size2, ang1, ang2]
```
PhotoTourism is dumped by [prepare_data](prepare_data/prepare_data_st.py), partially borrrowed 
from [NG-RANSAC](https://github.com/vislearn/ngransac) and [Ransac-Tourial-data](https://github.com/ducha-aiki/ransac-tutorial-2020-data).

## Train ARS-MAGSAC

```bash
python train_pymagsac.py -id 4 -m models/st_peters_square/weights_init_E_rs_r0.80_.net -rs -r 0.8 -t 0.75 -w1 4 -w3 6 -l all  -ds st_peters_square
```
The proposed sampler is used by set "-id 4", and the proposed affine loss is used 
when "-l all", and set w3 and w4 as non-zero values. Fundamental matrices are estimated when setting "-fmat" as True.

You can either train our method on [initial trained model](models/st_peters_square/weights_init_E_rs_r0.80_.net) or use the following steps to do the initial training on KL-divergence.
```bash
python init_pymagsac.py -rs -r 0.8 -t 0.001 -ds st_peters_square
```
Important parameters, check more details [here](util.py):

`--src_pth` -- the path of the input datasets.

`--datasets` -- specify the datasets (scenes).

`--rs` -- use ORB or RootSIFT correspondences, default as SIFT.

`--model` -- load the initial model for end-to-end training or a trained model for testing,

`--loss` -- select the loss function.

`--fmat` -- if provided, use the seven-point algorithm for fundamental matrix estimation.

`--network` --0 using residual blocks as NG-RANSAC, 1 using dense blocks, 2 use [CLNet](https://arxiv.org/abs/2101.00591), 3 use residual blocks concated with GNN layers.

`--sampler_id` --1 for PROSAC and 4 for AR-Sampler.

`--threshold` -- the threshold used in the minial sovler.


## Testing ARS-MAGSAC

The evaluation can be performed as,
```bash
python test_pymagsac.py -id 4 -m models/st_peters_square/weights_e2e_pymagsac_10_E_rs_r0.80_t0.75_w1_0.10_w3_0.60_w4_0.30_.net -rs -r 0.8 -t 2 -v test -w1 4 -w3 6 -bm -var 0.99
```
AUC scores for essential matrix estimation on PhotoTourism over 5, 10, 20 degrees of thresholds, and the pose errors and run time will ve returned. 
F1 scores and epipolar geometry errors are used for evaluating fundamental matrix estimation.

## Demo

Try the demo script first and compare the proposed ARS-MAGSAC with OpenCV-RANSAC as follows. We estimate the essential matrix from two examples [images](images/).

This will output the estimated models and inlier numbers from both methods, also, save the output images after each step, i.e., SIFT feature detection, SNN ratio test, and the final matches.
Note that one-fifth matches are drawn for clear visualization.
```bash
python demo.py -m models/st_peters_square/weights_e2e_pymagsac_10_E_rs_r0.80_t0.75_w1_0.10_w3_0.60_w4_0.30_.net -rs -r 0.8 -id 4 -t 0.75
```
![](images/comparison.jpg)


##  Citation
Check more details in the [paper](https://arxiv.org/abs/2212.13185), and cite it as:
```
@misc{wei2023adaptive,
      title={Adaptive Reordering Sampler with Neurally Guided MAGSAC}, 
      author={Tong Wei and Jiri Matas and Daniel Barath},
      year={2023},
      eprint={2111.14093},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
