import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import pymagsac
import util

from tensorboardX import SummaryWriter
from networks.resnet import ResNet
from networks.dense import DenseNet
from networks.clnet import CLNet
from networks.gnnet import GNNet
from dataset import SparseDataset

# parse command line arguments
parser = util.create_parser(
	description="Train ARS-MAGSAC.")
opt = parser.parse_args()

# construct folder that should contain pre-calculated correspondences
data_folder = opt.variant + '_data'
if opt.orb:
	data_folder += '_orb'
if opt.rootsift:
	data_folder += '_rs'

# load the train dataset
train_data = opt.datasets.split(',') #support multiple training datasets used jointly
train_data = [opt.src_pth + ds + '/' + data_folder + '/' for ds in train_data]
trainset = SparseDataset(train_data, opt.ratio, opt.nfeatures, opt.fmat, opt.nosideinfo)
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6, batch_size=opt.batchsize)

print('Threshold:', opt.threshold)
print('Epoch number:', opt.epochs, '\n')
print('Using datasets:', train_data)

# create or load model
if opt.network == 0:
	model = ResNet(opt.resblocks)
elif opt.network == 1:
	model = DenseNet()
elif opt.network == 2:
	model = CLNet()
else:
	model = GNNet(opt.resblocks)

print("\nImage pairs: ", len(trainset), "\n")

if len(opt.model) > 0:
	model.load_state_dict(torch.load(opt.model))
model = model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=opt.learningrate)

# weights of different loss components
w1 = float(opt.w1/(opt.w1 + opt.w2 + opt.w3 + opt.w4))
w2 = float(opt.w2/(opt.w1 + opt.w2 + opt.w3 + opt.w4))
w3 = float(opt.w3/(opt.w1 + opt.w2 + opt.w3 + opt.w4))
w4 = float(opt.w4/(opt.w1 + opt.w2 + opt.w3 + opt.w4))

print(w1, w2, w3, w4)
iteration = 0

# keep track of the training progress
session_string = util.create_session_string('e2e', opt.network, opt.sampler_id, opt.epochs, opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session, w1, w2, w3, w4, opt.threshold)
log_dir = 'logs/' + session_string + '/' + opt.variant + '/'
if not os.path.isdir(log_dir): os.makedirs(log_dir)
train_log = open(log_dir + 'log_%s.txt' % (session_string), 'w', 1)
param_log = open(log_dir + 'param_%s.txt' % (session_string), 'w', 1)
param = ['train set: ', opt.datasets, '\n sampler id: ', str(opt.sampler_id),
				'\n learning rate: ', str(opt.learningrate),
				'\n loss: ', opt.loss,
				'\n', str(w1), '\t', str(w2), '\t', str(w3), '\t', str(w4)]
for p in param:
	param_log.write(p)

model_dir = 'models/' + opt.datasets + '/'
if not os.path.isdir(model_dir): os.makedirs(model_dir)
writer = SummaryWriter(log_dir +'vision', comment="model_vis")

# main training loop
for epoch in range(0, opt.epochs):

	print("=== Starting Epoch", epoch, "==================================")

	# store the network every so often
	torch.save(model.state_dict(),  'models/' + opt.datasets + '/weights_%s.net' % (session_string))
	torch.save(model, 'models/model_vision.pt')

	# main training loop in the current epoch
	for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in trainset_loader:
		gt_R = gt_R.numpy()
		gt_t = gt_t.numpy()

		# predict neural guidance
		if opt.fmat == False:
			correspondences = correspondences.float()
		log_probs = model(correspondences.cuda())
		probs = torch.exp(log_probs).view(log_probs.size(0), -1).cpu().detach().numpy()
		writer.add_graph(model, correspondences.cuda())

		# this tensor will contain the gradients for the entire batch
		log_probs_grad = torch.zeros(log_probs.size())
		avg_loss = 0

		# loop over batch
		for b in range(correspondences.size(0)):

			# we sample multiple times per input and keep the gradients and losses in the following lists
			log_prob_grads = []
			losses = []

			# loop over samples for approximating the expected loss
			for s in range(opt.samplecount):

				# gradient tensor of the current sample,this tensor will indicate which correspondences have been samples
				# this is multiplied with the loss of the sample to yield the gradients for log-probabilities
				gradients = torch.zeros(log_probs[b].size())
				# inlier mask of the best model
				inliers = torch.zeros(log_probs[b].size())

				if opt.fmat:
					# === CASE FUNDAMENTAL MATRIX =========================================
					if s == 0:  # denormalization is inplace, so do it for the first sample only
						# restore pixel coordinates
						util.denormalize_pts(correspondences[b, 0:2], im_size1[b])
						util.denormalize_pts(correspondences[b, 2:4], im_size2[b])

					# generate the sorted weights and points, prepare for weights-related samplers
					weights = probs[b]
					sorted_indices = np.argsort(weights)[::-1]

					# fetch the coordinates and reorder them according tto the inlier probabilities
					pts1 = correspondences[b, 0:2].squeeze().numpy().T
					pts2 = correspondences[b, 2:4].squeeze().numpy().T

					sorted_pts1 = pts1[sorted_indices]
					sorted_pts2 = pts2[sorted_indices]

					# use the reordered points for PROSAC and AR-Sampler, MAGSAC++ pybinding from c++
					if ((opt.sampler_id == 1) or (opt.sampler_id == 4)):
						F, mask, samples = pymagsac.findFundamentalMatrix(
							sorted_pts1, sorted_pts2,
							float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
							probabilities=weights.astype(np.float64),
							use_magsac_plus_plus=True,
							sigma_th=opt.threshold,
							sampler_id=opt.sampler_id,
							save_samples=True)

					else:
						F, mask, samples = pymagsac.findFundamentalMatrix(
							np.ascontiguousarray(pts1), np.ascontiguousarray(pts2),
							float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
							probabilities=weights.astype(np.float64),
							use_magsac_plus_plus=True,
							sigma_th=opt.threshold,
							sampler_id=opt.sampler_id,
							save_samples=True)

					# count inlier number
					incount = sum(mask)
					incount /= correspondences.size(2)

					if (incount == 0):
						F = np.identity(3)
					else:
						# update gradients and inliers
						if (opt.sampler_id == 1) or (opt.sampler_id == 4):
							# transform samples from rearranged indices to the original
							for sample in samples:
								sample = sorted_indices[sample]
								gradients[0, sample, 0] += 1

							# update the inlier mask with orignal indices according to the sorted indices
							sorted_index = sorted_indices[mask]
							inliers[0, sorted_index, 0] = 1

						else:
							for sample in samples:
								gradients[0, sample, 0] += 1
							inliers[0, :, 0] = mask

					# evaluation
					pts1 = correspondences[b, 0:2].numpy()
					pts2 = correspondences[b, 2:4].numpy()

					# fetch the orientation and sacle info
					scale_ratio = correspondences[b, 5].squeeze().numpy().T
					ang = correspondences[b, 6].squeeze().numpy().T

					# calculate the orientation loss
					ori_err = util.orientation_error(pts1, pts2, F, ang, b)
					ori_gt = util.orientation_error(pts1, pts2, gt_F[b].numpy(), ang, b)
					ori_loss = abs(ori_err - ori_gt)
					ori_loss /= correspondences.size(2)

					# calculate the scale loss
					sca_err = util.scale_error(pts1, pts2, F, scale_ratio, b)
					sca_gt = util.scale_error(pts1, pts2, gt_F[b].numpy(), scale_ratio, b)
					sca_loss = abs(sca_err - sca_gt)
					sca_loss /= correspondences.size(2)

					# compute fundamental matrix metrics if they are used as training signal
					if opt.loss is not 'pose':
						valid, F1, incount, epi_error = util.f_error(pts1, pts2, F, gt_F[b].numpy(), 0.75)

					E = K2[b].numpy().T.dot(F.dot(K1[b].numpy()))

					# normalize correspondences using the calibration parameters for the calculation of pose errors
					pts1_1 = cv2.undistortPoints(pts1.transpose(2, 1, 0), K1[b].numpy(), None)
					pts2_2 = cv2.undistortPoints(pts2.transpose(2, 1, 0), K2[b].numpy(), None)

				else:

					# === CASE ESSENTIAL MATRIX =========================================

					# focal length
					f1 = 0.5 * (K1.numpy()[b][0][0] + K1.numpy()[b][1][1])
					f2 = 0.5 * (K2.numpy()[b][0][0] + K2.numpy()[b][1][1])

					# generate the sorted weights and points, prepare for PROSAC
					weights = probs[b]

					sorted_indices = np.argsort(weights)[::-1]

					pts1 = correspondences[b, 0:2].squeeze().numpy().T
					pts2 = correspondences[b, 2:4].squeeze().numpy().T

					sorted_pts1 = pts1[sorted_indices]
					sorted_pts2 = pts2[sorted_indices]

					if ((opt.sampler_id == 1) or (opt.sampler_id == 4)):
						# print("use prosac")
						E, mask, samples = pymagsac.findEssentialMatrix(
							np.ascontiguousarray(sorted_pts1), np.ascontiguousarray(sorted_pts2),
							K1[b].numpy(),K2[b].numpy(),
							float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
							probabilities=weights,
							use_magsac_plus_plus=True,
							sigma_th=opt.threshold,
							sampler_id=opt.sampler_id,
							variance=opt.variance,
							save_samples=True)
					else:
						E, mask, samples = pymagsac.findEssentialMatrix(
							np.ascontiguousarray(pts1), np.ascontiguousarray(pts2),
							K1[b].numpy(), K2[b].numpy(),
							float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
							probabilities=weights,
							use_magsac_plus_plus=True,
							sigma_th=opt.threshold,
							sampler_id=opt.sampler_id,
							variance=opt.variance,
							save_samples=True)

					# count inlier number
					incount = sum(mask)
					incount /= correspondences.size(2)

					# for checking the success estimation
					if (incount == 0):
						E = np.identity(3)
					else:
					# update gradients and inliers
						if (opt.sampler_id == 1) or (opt.sampler_id == 4):
							# transform samples from rearranged indices to the original
							for sample in samples:
								sample = sorted_indices[sample]
								gradients[0, sample, 0] += 1

							# update the inlier mask with orignal indices according to the sorted indices
							sorted_index = sorted_indices[mask]
							inliers[0, sorted_index, 0] = 1

						else:
							for sample in samples:
								gradients[0, sample, 0] += 1
							inliers[0, :, 0] = mask

					pts1 = correspondences[b, 0:2].numpy()
					pts2 = correspondences[b, 2:4].numpy()

					# pick up the orientation and sacle info, then normalize them
					scale_ratio = correspondences[b, 5].squeeze().numpy().T
					ang = correspondences[b, 6].squeeze().numpy().T

					scale_ratio_n = scale_ratio * (f1 / f2)
					ang_n = ang * (f1 / f2)

					# calculate the orientation loss
					ori_err = util.orientation_error(pts1, pts2, E, ang_n, b)
					ori_gt = util.orientation_error(pts1, pts2, gt_E[b].numpy(), ang_n, b)
					ori_loss = abs(ori_err - ori_gt)
					ori_loss /= correspondences.size(2)

					# calculate the scale loss
					sca_err = util.scale_error(pts1, pts2, E, scale_ratio_n, b)
					sca_gt = util.scale_error(pts1, pts2, gt_E[b].numpy(), scale_ratio_n, b)
					sca_loss = abs(sca_err - sca_gt)
					sca_loss /= correspondences.size(2)

					# pts for recovering the pose
					pts1_1 = pts1.transpose(2, 1, 0)
					pts2_2 = pts2.transpose(2, 1, 0)
					epi_error=0
				# choose the user-defined training signal
				if opt.loss == 'inliers':
					loss = -incount
				# print(loss)
				elif opt.loss == 'f1' and opt.fmat:
					loss = -F1
				elif opt.loss == 'epi' and opt.fmat:
					loss = epi_error
				elif opt.loss == 'all':
					# evaluation of relative pose (essential matrix)
					inliers = inliers.byte().numpy().ravel()
					# previously commented
					# E = E.double().numpy()
					K = np.eye(3)
					R = np.eye(3)
					t = np.zeros((3, 1))

					cv2.recoverPose(E, pts1_1, pts2_2, K, R, t, inliers)  # from E to R and T
					dR, dT = util.pose_error(R, gt_R[b], t, gt_t[b])
					pose_loss = max(float(dR), float(dT))

					# turn down the combination for testing
					loss = w1 * pose_loss + w2 * epi_error + w3 * ori_loss + w4 * sca_loss
				log_prob_grads.append(gradients)
				losses.append(loss)

			# calculate the gradients of the expected loss
			baseline = sum(losses) / len(losses)  # expected loss
			for i, l in enumerate(losses):  # subtract the baseline for each sample to reduce gradient variance
				log_probs_grad[b] += log_prob_grads[i] * (l - baseline) / opt.samplecount

			avg_loss += baseline

		avg_loss /= correspondences.size(0)
		# for vision
		writer.add_scalar('loss', avg_loss, global_step=iteration)
		writer.flush()

		train_log.write('%d %f\n' % (iteration, avg_loss))

		# update model
		torch.autograd.backward((log_probs), (log_probs_grad.cuda()))
		optimizer.step()
		optimizer.zero_grad()

		print("Iteration: ", iteration, "Loss: ", avg_loss)

		iteration += 1
