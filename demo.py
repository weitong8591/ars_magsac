import numpy as np
import cv2
import torch
import pymagsac
import time
from networks.resnet import ResNet
from draw_matches import draw_matches

import util

parser = util.create_parser('Fits an essential matrix (default) or fundamental matrix (-fmat) using OpenCV RANSAC vs. NG-RANSAC.')

parser.add_argument('--image1', '-img1', default='images/demo1.jpg',
	help='path to image 1')

parser.add_argument('--image2', '-img2', default='images/demo2.jpg',
	help='path to image 2')

parser.add_argument('--outimg', '-out', default='images/comparison.jpg',
	help='images will store a matching image under this file name')


parser.add_argument('--focallength1', '-fl1', type=float, default=900, 
	help='focal length of image 1 (only used when fitting the essential matrix)')

parser.add_argument('--focallength2', '-fl2', type=float, default=900, 
	help='focal length of image 2 (only used when fitting the essential matrix)')


opt = parser.parse_args()

if opt.fmat:
	print("\nFitting Fundamental Matrix...\n")
else:
	print("\nFitting Essential Matrix...\n")

# setup detector
if opt.orb:
	print("Using ORB.\n")
	if opt.nfeatures > 0:
		detector = cv2.ORB_create(nfeatures=opt.nfeatures)
	else:
		detector = cv2.ORB_create()
else:
	if opt.rootsift:
		print("Using RootSIFT.\n")
	else:
		print("Using SIFT.\n")
	if opt.nfeatures > 0:
		detector = cv2.xfeatures2d.SIFT_create(nfeatures=opt.nfeatures, contrastThreshold=1e-5)
	else:
		detector = cv2.xfeatures2d.SIFT_create()

# loading neural guidence network#
model_file = opt.model
if len(model_file) == 0:
	model_file = util.create_session_string('e2e', opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session)
	model_file = 'models/weights_' + model_file + '.net'
	print("No model file specified. Inferring pre-trained model from given parameters:")
	print(model_file)

model = ResNet(opt.resblocks)
model.load_state_dict(torch.load(model_file))
model = model.cuda()
model.eval()
print("Successfully loaded model.")

print("\nProcessing pair:")
print("Image 1: ", opt.image1)
print("Image 2: ", opt.image2)

# read images
img1_ori = cv2.imread(opt.image1)
img1 = cv2.cvtColor(img1_ori, cv2.COLOR_BGR2GRAY)

img2_ori = cv2.imread(opt.image2)
img2 = cv2.cvtColor(img2_ori, cv2.COLOR_BGR2GRAY)

# calibration matrices of image 1 and 2, principal point assumed to be at the center
K1 = np.eye(3)
K1[0,0] = K1[1,1] = opt.focallength1
K1[0,2] = img1.shape[1] * 0.5
K1[1,2] = img1.shape[0] * 0.5

K2 = np.eye(3)
K2[0,0] = K2[1,1] = opt.focallength2
K2[0,2] = img2.shape[1] * 0.5
K2[1,2] = img2.shape[0] * 0.5

# detect features
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

print("\nFeature found in image 1:", len(kp1))
print("Feature found in image 2:", len(kp2))

# root sift normalization
if opt.rootsift:
	print("Using RootSIFT normalization.")
	desc1 = util.rootSift(desc1)
	desc2 = util.rootSift(desc2)

# feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)


good_matches = []
pts1 = []
pts2 = []
m_draw = []
ratios = []
sizes1 = []
sizes2 = []
angles1 = []
angles2 = []
print("")
if opt.ratio < 1.0:
	print("Using Lowe's ratio filter with", opt.ratio)

for (m,n) in matches:
	m_draw.append(m)
	if m.distance < 0.8*n.distance: # apply Lowe's ratio filter
		
		good_matches.append(m)
		
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)

		ratios.append(m.distance / n.distance)

		sizes1.append(kp1[m.queryIdx].size)
		angles1.append(kp1[m.queryIdx].angle)

		sizes2.append(kp2[m.trainIdx].size)
		angles2.append(kp2[m.trainIdx].angle)

print("Number of valid matches:", len(good_matches))

pts1 = np.array([pts1])
pts2 = np.array([pts2])

ratios = np.array([ratios])
ratios = np.expand_dims(ratios, 2)


sizes1 = np.array([sizes1])
sizes1 = np.expand_dims(sizes1, 2)

sizes2 = np.array([sizes2])
sizes2 = np.expand_dims(sizes2, 2)

angles1 = np.array([angles1])
angles1 = np.expand_dims(angles1, 2)

angles2 = np.array([angles2])
angles2 = np.expand_dims(angles2, 2)

# ------------------------------------------------
# fit fundamental or essential matrix using OPENCV
# ------------------------------------------------
if opt.fmat:

	# === CASE FUNDAMENTAL MATRIX =========================================

	ransac_model, ransac_inliers = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=opt.threshold, confidence=0.999)
else:
	# === CASE ESSENTIAL MATRIX =========================================

	# normalize key point coordinates when fitting the essential matrix
	pts1 = cv2.undistortPoints(pts1, K1, None)
	pts2 = cv2.undistortPoints(pts2, K2, None)

	K = np.eye(3)

	ransac_model, ransac_inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=opt.threshold)

print("\n=== Model found by RANSAC: ==========\n")
print(ransac_model)

print("\nRANSAC Inliers:", ransac_inliers.sum())

# ---------------------------------------------------
# fit fundamental or essential matrix using ARS-MAGSAC
# ---------------------------------------------------

scale_ratio = sizes2/sizes1
ang = ((angles2 - angles1) % 180) * (3.141592653 / 180)

# create data tensor of feature coordinates and matching ratios
correspondences = np.concatenate((pts1, pts2, ratios, scale_ratio, ang), axis=2)
correspondences = np.transpose(correspondences)
correspondences = torch.from_numpy(correspondences).float()

# predict neural guidance, i.e. RANSAC sampling probabilities
log_probs = model(correspondences.unsqueeze(0).cuda())[0] #zero-indexing creates and removes a dummy batch dimension
probs = torch.exp(log_probs).view(log_probs.size(0), -1).cpu().detach().numpy()
out_model = torch.zeros((3, 3)).float() # estimated model
out_inliers = torch.zeros(log_probs.size()) # inlier mask of estimated model
out_gradients = torch.zeros(log_probs.size()) # gradient tensor (only used during training)


# ARS-MAGSAC
if opt.fmat:

	# === CASE FUNDAMENTAL MATRIX =========================================

	# undo normalization of x and y image coordinates
	util.denormalize_pts(correspondences[0:2], img1.shape)
	util.denormalize_pts(correspondences[2:4], img2.shape)
	weights = probs[0]
	sorted_indices = np.argsort(weights)[::-1]

	# fetch the coordinates and reorder them according tto the inlier probabilities
	pts1 = correspondences[0:2].squeeze().numpy().T
	pts2 = correspondences[2:4].squeeze().numpy().T

	sorted_pts1 = pts1[sorted_indices]
	sorted_pts2 = pts2[sorted_indices]
	ransac_time = 0
	# call pymagsac
	if opt.sampler_id == 1 or opt.sampler_id == 4:

		start_time = time.time()
		out_model, out_inliers, samples = pymagsac.findFundamentalMatrix(
			np.ascontiguousarray(sorted_pts1), np.ascontiguousarray(sorted_pts2),
			K1.numpy(), K2.numpy(),
			float(img1.shape[0]), float(img1.shape[1]), float(img2.shape[0]), float(img2.shape[1]),
			probabilities=weights,
			use_magsac_plus_plus=True,
			sigma_th=opt.threshold,
			sampler_id=opt.sampler_id,
			save_samples=False)
		ransac_time += time.time() - start_time
	else:
		start_time = time.time()
		out_model, out_inliers, samples = pymagsac.findFundamentalMatrix(
			np.ascontiguousarray(pts1), np.ascontiguousarray(pts2),
			K1[b].numpy(), K2[b].numpy(),
			float(img1.shape[0]), float(img1.shape[1]), float(img2.shape[0]), float(img2.shape[1]),
			probabilities=weights,
			variance=opt.variance,
			use_magsac_plus_plus=True,
			sigma_th=opt.threshold,
			sampler_id=opt.sampler_id,
			save_samples=False)
		ransac_time += time.time() - start_time

		incount = np.sum(out_inliers)

else:

	# === CASE ESSENTIAL MATRIX =========================================
	weights = probs[0]
	print(weights.shape)
	sorted_indices = np.argsort(weights)[::-1]
	# fetch the coordinates and reorder them according tto the inlier probabilities
	pts1 = correspondences[0:2].squeeze().numpy().T
	pts2 = correspondences[2:4].squeeze().numpy().T
	sorted_pts1 = pts1[sorted_indices]
	sorted_pts2 = pts2[sorted_indices]

	ransac_time = 0
	# call pymagsac
	if opt.sampler_id == 1 or opt.sampler_id == 4:

		start_time = time.time()
		out_model, out_inliers, samples = pymagsac.findEssentialMatrix(
			np.ascontiguousarray(sorted_pts1), np.ascontiguousarray(sorted_pts2),
			K1, K2,
			float(img1.shape[0]), float(img1.shape[1]), float(img2.shape[0]), float(img2.shape[1]),
			probabilities=weights,
			use_magsac_plus_plus=True,
			sigma_th=opt.threshold,
			sampler_id=opt.sampler_id,
			save_samples=False)
		ransac_time += time.time() - start_time
	else:
		start_time = time.time()
		out_model, out_inliers, samples = pymagsac.findEssentialMatrix(
			np.ascontiguousarray(pts1), np.ascontiguousarray(pts2),
			K1, K2,
			float(img1.shape[0]), float(img1.shape[1]), float(img2.shape[0]), float(img2.shape[1]),
			probabilities=weights,
			variance=opt.variance,
			use_magsac_plus_plus=True,
			sigma_th=opt.threshold,
			sampler_id=opt.sampler_id,
			save_samples=False)
		ransac_time += time.time() - start_time

	incount = np.sum(out_inliers)

print("\n=== Model found by ARS-MAGSAC: =======\n")
print(out_model)

print("\nARS-MAGSAC Inliers: ", int(incount))

# create a visualization of the matching, comparing results of RANSAC and NG-RANSAC
out_inliers = out_inliers.ravel().tolist()
ransac_inliers = ransac_inliers.ravel().tolist()

match_sift = cv2.drawMatches(img1_ori, kp1, img2_ori, kp2, m_draw, None, flags=2, matchColor=(-1))
cv2.imwrite('images/sift.jpg', match_sift)

match_snn = draw_matches(img1_ori, kp1, img2_ori, kp2, good_matches, [1]*2000, color = None)
cv2.imwrite('images/snn.jpg', match_snn)

match_img_ransac = draw_matches(img1_ori, kp1, img2_ori, kp2, good_matches, ransac_inliers, color = None)
cv2.imwrite('images/ransac.jpg', match_img_ransac)

match_img_ours = draw_matches(img1_ori, kp1, img2_ori, kp2, good_matches, out_inliers, color = None)
cv2.imwrite('images/ours.jpg', match_img_ours)

match_img = np.concatenate((match_img_ransac, match_img_ours), axis=0)
cv2.imwrite(opt.outimg, match_img)
print("\nDone. Visualization of the result stored as", opt.outimg)
