import numpy as np
import cv2
import os
import torch
import pymagsac
import time
import pandas as pd
from networks.resnet import ResNet
from networks.dense import DenseNet
from networks.clnet import CLNet
from networks.gnnet import GNNet
from dataset import SparseDataset
import util


parser = util.create_parser(
    description="Test ARS-MAGSAC.")

opt = parser.parse_args()

# weights of different losses
w1 = float(opt.w1/(opt.w1 + opt.w2 + opt.w3 + opt.w4))
w2 = float(opt.w2/(opt.w1 + opt.w2 + opt.w3 + opt.w4))
w3 = float(opt.w3/(opt.w1 + opt.w2 + opt.w3 + opt.w4))
w4 = float(opt.w4/(opt.w1 + opt.w2 + opt.w3 + opt.w4))

print(w1,w2,w3,w4)

# construct folder that should contain pre-calculated correspondences
data_folder = opt.variant + '_data'
if opt.orb:
	data_folder += '_orb'
if opt.rootsift:
	data_folder += '_rs'

session_string = util.create_session_string('test', opt.network,opt.sampler_id, opt.epochs, opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session, w1, w2, w3, w4, opt.threshold)

# load a model, either directly provided by the user, or infer a pre-trained model from the command line parameters
model_file = opt.model
if len(model_file) == 0:
	model_file = util.create_session_string('e2e', opt.ransac, opt.epochs, opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session, w1, w2, w3, w4, opt.threshold)
	model_file = 'models/'+ opt.dataset + '/weights_' + model_file + '.net'
	print("No model file specified. Inferring pre-trained model from given parameters:")

# collect datasets to be used for testing
if opt.batchmode:
	datasets = util.test_datasets
	print("\n=== BATCH MODE: Doing evaluation for", len(datasets), "datasets. =================")
else:
	datasets = [opt.dataset]

# loop over datasets, perform a separate evaluation per dataset
print(torch.cuda.is_available())
for dataset in datasets:
	print('Starting evaluation for dataset:', dataset, data_folder, "\n")
	testset = SparseDataset([opt.src_pth + dataset + '/'+ data_folder+'/'], opt.ratio, opt.nfeatures, opt.fmat, opt.nosideinfo)

	# create or load model
	if opt.network == 0:
		model = ResNet(opt.resblocks)
	elif opt.network == 1:
		model = DenseNet()
	elif opt.network == 2:
		model = CLNet()
	else:
		model = GNNet(opt.resblocks)

	testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6, batch_size=opt.batchsize)

	testset_folder = opt.src_pth + dataset + '/' + data_folder
	test_files = os.listdir(testset_folder)

	# load the model
	model.load_state_dict(torch.load(model_file))
	model = model.cuda()
	model.eval()
	print("Successfully loaded model.")

	maskfile = 'weight_log/' + 'test/'+ dataset + '/log_%s.txt' % (session_string)
	avg_model_time = 0 # runtime of the network forward pass
	avg_ransac_time = 0 # runtime of RANSAC
	avg_counter = 0

	# essential matrix evaluation
	pose_losses = []

	# evaluation according to "deep fundamental matrix" (Ranftl and Koltun, ECCV 2018)
	avg_F1 = 0
	avg_inliers = 0
	epi_errors = []
	invalid_pairs = 0
	mask_data = []
	F1_record = []

	s = []
	time_record = []
	with torch.no_grad():
		for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in testset_loader:
			print("Processing batch", avg_counter+1, "of", len(testset_loader))

			gt_R = gt_R.numpy()
			gt_t = gt_t.numpy()
			start_time = time.time()
			if opt.fmat == False:
				correspondences = correspondences.float()

			# predict sampling weights
			log_probs = model(correspondences.cuda())
			#probs = torch.exp(log_probs).cpu()
			probs = torch.exp(log_probs).view(log_probs.size(0), -1).cpu().detach().numpy()
			avg_model_time += (time.time()-start_time) / opt.batchsize

			ransac_time = 0
			# loop over batch
			for b in range(correspondences.size(0)):
				gradients = torch.zeros(log_probs[b].size()) # not used in test mode, indicates which correspondence have been sampled
				inliers = torch.zeros(log_probs[b].size()) # inlier mask of winning model
				if opt.fmat:
					# === CASE FUNDAMENTAL MATRIX =========================================
					# restore pixel coordinates
					util.denormalize_pts(correspondences[b, 0:2], im_size1[b])
					util.denormalize_pts(correspondences[b, 2:4], im_size2[b])

					weights = probs[b]
					sorted_indices = np.argsort(weights)[::-1]

					# fetch the coordinates and reorder them according tto the inlier probabilities
					pts1 = correspondences[b, 0:2].squeeze().numpy().T
					pts2 = correspondences[b, 2:4].squeeze().numpy().T
					sorted_pts1 = pts1[sorted_indices]
					sorted_pts2 = pts2[sorted_indices]

					if opt.sampler_id ==1 or opt.sampler_id ==4:
						start_time = time.time()
						F, mask, samples = pymagsac.findFundamentalMatrix(
						sorted_pts1, sorted_pts2,
						float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
						probabilities=weights,
						use_magsac_plus_plus=True,
						sigma_th = opt.threshold,
						sampler_id =  opt.sampler_id,
						save_samples = False)
						current_time = time.time()-start_time
						ransac_time += current_time

					else:
						start_time = time.time()
						F, mask, samples = pymagsac.findFundamentalMatrix(
						np.ascontiguousarray(pts1), np.ascontiguousarray(pts2),
						float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
						probabilities=weights,
						use_magsac_plus_plus=True,
						sigma_th = opt.threshold,
						sampler_id =  opt.sampler_id,
						variance = opt.variance,
						save_samples = False)
						current_time = time.time() - start_time
						ransac_time += current_time


					# count inlier number
					incount = np.sum(mask)
					incount /= correspondences.size(2)

					# for checking the success estimation
					if (incount == 0):
						F = np.identity(3)
					else:
						# update gradients and inliers
						if (opt.sampler_id == 1) or (opt.sampler_id == 4):
							# update the inlier mask with original indices according to the sorted indices
							sorted_index = sorted_indices[mask]
							inliers[0, sorted_index, 0] = 1
						else:
							inliers[0, :, 0] = mask


					# essential matrix from fundamental matrix (for evaluation via relative pose)
					E = K2[b].numpy().T.dot(F.dot(K1[b].numpy()))
					pts1 = correspondences[b,0:2].numpy()
					pts2 = correspondences[b,2:4].numpy()

					# evaluation of F matrix via correspondences
					valid, F1, epi_inliers, epi_error = util.f_error(pts1, pts2, F, gt_F[b].numpy(), 0.75)

					if valid:
						avg_F1 += F1
						s.append(incount)
						F1_record.append(F1)
						avg_inliers += epi_inliers
						epi_errors.append(epi_error)
						time_record.append(current_time)
					else:
						# F matrix evaluation failed (ground truth model had no inliers)
						invalid_pairs += 1

					# normalize correspondences using the calibration parameters for the calculation of pose errors
					pts1_1 = cv2.undistortPoints(pts1.transpose(2, 1, 0), K1[b].numpy(), None)
					pts2_2 = cv2.undistortPoints(pts2.transpose(2, 1, 0), K2[b].numpy(), None)
				else:
					# === CASE ESSENTIAL MATRIX =========================================

					weights = probs[b]
					sorted_indices = np.argsort(weights)[::-1]

					# fetch the coordinates and reorder them according tto the inlier probabilities
					pts1 = correspondences[b, 0:2].squeeze().numpy().T
					pts2 = correspondences[b, 2:4].squeeze().numpy().T

					sorted_pts1 = pts1[sorted_indices]
					sorted_pts2 = pts2[sorted_indices]
					# call pymagsac
					if opt.sampler_id ==1 or opt.sampler_id ==4:

						start_time = time.time()
						E, mask, samples = pymagsac.findEssentialMatrix(
						np.ascontiguousarray(sorted_pts1), np.ascontiguousarray(sorted_pts2),
						K1[b].numpy(), K2[b].numpy(),
						float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
						probabilities= weights,
						use_magsac_plus_plus=True,
						sigma_th=opt.threshold,
						sampler_id=opt.sampler_id,
						save_samples = False)
						ransac_time += time.time()-start_time
					else:
						start_time = time.time()
						E, mask, samples = pymagsac.findEssentialMatrix(
						np.ascontiguousarray(pts1), np.ascontiguousarray(pts2),
						K1[b].numpy(), K2[b].numpy(),
						float(im_size1[b][0]), float(im_size1[b][1]), float(im_size2[b][0]), float(im_size2[b][1]),
						probabilities=weights,
						variance = opt.variance,
						use_magsac_plus_plus=True,
						sigma_th=opt.threshold,
						sampler_id=opt.sampler_id,
						save_samples = False)
						ransac_time += time.time()-start_time

					# count inlier number
					incount = np.sum(mask)
					#print(incount)
					incount /= correspondences.size(2)

					if (incount == 0):
						E = np.identity(3)
					else:
						# update gradients and inliers
						if (opt.sampler_id == 1) or (opt.sampler_id == 4):
							# update the inlier mask with original indices according to the sorted indices
							sorted_index = sorted_indices[mask]
							inliers[0, sorted_index, 0] = 1
						else:
							inliers[0, :, 0] = mask

					# pts for recovering the pose
					pts1 = correspondences[b,0:2].numpy()
					pts2 = correspondences[b,2:4].numpy()

					pts1_1 = pts1.transpose(2, 1, 0)
					pts2_2 = pts2.transpose(2, 1, 0)

				inliers = inliers.byte().numpy().ravel()

				K = np.eye(3)
				R = np.eye(3)
				t = np.zeros((3,1))

				# evaluation of relative pose (essential matrix)
				cv2.recoverPose(E, pts1_1, pts2_2, K, R, t, inliers)

				dR, dT = util.pose_error(R, gt_R[b], t, gt_t[b])
				pose_losses.append(max(float(dR), float(dT)))


			avg_ransac_time += ransac_time / opt.batchsize

			avg_counter += 1
	print(avg_counter)
	print("\nAvg. Model Time: %dms" % (avg_model_time / avg_counter*1000+0.00000001))
	print("Avg. RANSAC Time: %dms" % (avg_ransac_time / avg_counter*1000+0.00000001))
	print("debug5")

	if opt.fmat:
		dataframe = pd.DataFrame({'error': epi_errors, 'F1_score': F1_record, 'time': time_record, 'inlier_num': s})
	else:
		dataframe = pd.DataFrame({'error': pose_losses})

	w = 'thesis_plotting/result_save/tune/'+ opt.model +'/'
	if not os.path.isdir(w): os.makedirs(w)
	dataframe.to_csv(w+ dataset +'_result.csv', sep=',')

	# calculate AUC of pose losses
	thresholds = [5, 10, 20]
	AUC = util.AUC(losses = pose_losses, thresholds = thresholds, binsize=opt.evalbinsize)
	print("\n=== Relative Pose Accuracy ===========================")
	print("AUC for %ddeg/%ddeg/%ddeg: %.2f/%.2f/%.2f\n" % (thresholds[0], thresholds[1], thresholds[2], AUC[0], AUC[1], AUC[2]))
	if opt.fmat:
		print("\n=== F-Matrix Evaluation ==============================")
		if len(epi_errors) == 0:
			print("F-Matrix evaluation failed because no ground truth inliers were found.")
			print("Check inlier threshold?.")
		else:
			avg_F1 /= len(epi_errors)
			avg_inliers /= len(epi_errors)
			epi_errors.sort()
			mean_epi_err = sum(epi_errors) / len(epi_errors)
			median_epi_err = epi_errors[int(len(epi_errors)/2)]
			print("Invalid Pairs (ignored in the following metrics):", invalid_pairs)
			print("F1 Score: %.2f%%" % (avg_F1 * 100))
			print("%% Inliers: %.2f%%" % (avg_inliers * 100))
			print("Mean Epi Error: %.2f" % mean_epi_err)
			print("Median Epi Error: %.2f" % median_epi_err)

	# write evaluation results to file
	out_dir = 'results' + model_file[8:] + '/'
	if not os.path.isdir(out_dir): os.makedirs(out_dir)
	with open(out_dir + '%s.txt' % (session_string), 'a', 1) as f:
		f.write('%f %f %f %dms' % (AUC[0], AUC[1], AUC[2], (avg_ransac_time / avg_counter*1000)))
		if opt.fmat and len(epi_errors) > 0: f.write(' %f %f %f %f %f' % (avg_F1, avg_inliers, mean_epi_err, median_epi_err, avg_ransac_time))
		f.write('\n')
