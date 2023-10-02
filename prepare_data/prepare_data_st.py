"""
Extract SIFT features of the RANSAC-tutorial data
https://github.com/ducha-aiki/ransac-tutorial-2020-data

"""
import numpy as np
import cv2
import h5py
import math
import argparse
import os
from utils import *
from utils import loadh5 as load_h5
#import torch
import util

# parse command line arguments
parser = argparse.ArgumentParser(
	description='Dump the local features.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opt = parser.parse_args()
print('Using dataset: ', opt.dataset, opt.variant)
DIR = opt.img_pth
out_dir = opt.src + opt.dataset + '/'+ opt.variant + '_data'

seq = opt.dataset

if opt.orb:
	out_dir += '_orb'
if opt.rootsift:
	out_dir += '_rs'
out_dir += '/'
if not os.path.isdir(out_dir): os.makedirs(out_dir)

matches = load_h5(f'{DIR}/{seq}/matches.h5')
F_gt = load_h5(f'{DIR}/{seq}/Fgt.h5')
E_gt = load_h5(f'{DIR}/{seq}/Egt.h5')
matches_scores = load_h5(f'{DIR}/{seq}/match_conf.h5')
K1_K2 = load_h5(f'{DIR}/{seq}/K1_K2.h5')
R = load_h5(f'{DIR}/{seq}/R.h5')
T = load_h5(f'{DIR}/{seq}/T.h5')



detector = cv2.SIFT_create(opt.nfeatures, contrastThreshold = 1e-5)
count = 0
for k,F in F_gt.items():
	

	m = matches[k]
	ms = matches_scores[k]

	img_id1 = k.split('-')[0]
	img_id2 = k.split('-')[1]

	img1_fname = f'{DIR}/{seq}/images/' + img_id1 + '.jpg'
	img2_fname = f'{DIR}/{seq}/images/' + img_id2 + '.jpg'

	img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(cv2.imread(img2_fname), cv2.COLOR_BGR2RGB)

	#detect features
	kp1, desc1 = detector.detectAndCompute(img1, None)
	kp2, desc2 = detector.detectAndCompute(img2, None)
	print(kp1)
	# root sift normalization
	if opt.rootsift:
		desc1 = util.rootSift(desc1)
		desc2 = util.rootSift(desc2)
	# feature matching
	bf = cv2.BFMatcher()
	detected_matches = bf.knnMatch(desc1, desc2, k=2)
	#print(matches)
	pts1 = []
	pts2 = []
	ratios = []
	sizes1 = []
	sizes2 = []
	angles1 = []
	angles2 = []
	#pts2.append(kp2)
	#pts1.append(kp1)

	K1 = K1_K2[img_id1 + '-' + img_id2][0][0]
	K2 = K1_K2[img_id1 + '-' + img_id2][0][1]

	#ratios = ms.reshape(-1)
	for (m,n) in detected_matches:
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)

		ratios.append(m.distance / (n.distance))
		sizes1.append(kp1[m.queryIdx].size)
		angles1.append(kp1[m.queryIdx].angle)
		sizes2.append(kp2[m.trainIdx].size)
		angles2.append(kp2[m.trainIdx].angle)

	print("Matches:", len(detected_matches))
	pts1 = np.array([pts1])
	pts2 = np.array([pts2])
	ratios = np.array([ratios])
	ratios = np.expand_dims(ratios, 2)
	print(ratios)
	sizes1 = np.array([sizes1])
	sizes1 = np.expand_dims(sizes1, 2)

	sizes2 = np.array([sizes2])
	sizes2 = np.expand_dims(sizes2, 2)

	angles1 = np.array([angles1])
	angles1 = np.expand_dims(angles1, 2)

	angles2 = np.array([angles2])
	angles2 = np.expand_dims(angles2, 2)

	R1 = R[img_id1]
	R2 = R[img_id2]
	T1 = T[img_id1]
	T2 = T[img_id2]
	dR = np.dot(R2, R1.T)
	dT = T2 - np.dot(dR, T1)
	print(np.shape(img1)[:2])
	np.save(out_dir+'pair_%s_%s.npy'%(img_id1,img_id2),[
		pts1.astype(np.float32),
		pts2.astype(np.float32),
		ratios.astype(np.float32),
		np.shape(img1)[:2],
		np.shape(img2)[:2],
		K1.astype(np.float32),
		K2.astype(np.float32),
		dR.astype(np.float32),
		dT.astype(np.float32),
		sizes1.astype(np.float32),
		angles1.astype(np.float32),
		sizes2.astype(np.float32),
		angles2.astype(np.float32)
	])
	print("We have save %d pair images data" %count)
	count += 1
	if (count > opt.pairnum):
		break

