import os
import torch
import torch.optim as optim
import init
from networks.resnet import ResNet
from networks.dense import DenseNet
from networks.clnet import CLNet
from networks.gnnet import GNNet
from dataset import SparseDataset
import util

from tensorboardX import SummaryWriter
# parse command line arguments
parser = util.create_parser(
	description="Train an initial network KL-divergence.")

opt = parser.parse_args()

# construct folder that should contain pre-calculated correspondences
data_folder = opt.variant + '_data'
if opt.orb:
	data_folder += '_orb'
if opt.rootsift:
	data_folder += '_rs'

train_data = opt.datasets.split(',') #support multiple training datasets used jointly
train_data = [opt.src_pth + '/' + ds + '/' + data_folder + '/' for ds in train_data]
print('Threshold:', opt.threshold)
print('Epoch number:',opt.epochs, '\n')
print('Using datasets:')
for d in train_data: print(d)

# create or load model
trainset = SparseDataset(train_data, opt.ratio, opt.nfeatures, opt.fmat, opt.nosideinfo)
# create or load model
if opt.network == 0:
	model = ResNet(opt.resblocks)
elif opt.network == 1:
	model = DenseNet()
elif opt.network == 2:
	model = CLNet()
else:
	model = GNNet(opt.resblocks)

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6, batch_size=opt.batchsize)

print("\nImage pairs: ", len(trainset), "\n")


if len(opt.model) > 0:
	model.load_state_dict(torch.load(opt.model))
model = model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=opt.learningrate)

iteration = 0

# keep track of the training progress
session_string = util.create_session_string('init', opt.network, 0, opt.epochs, opt.fmat, opt.orb, opt.rootsift, opt.ratio, opt.session, opt.w1, opt.w2, opt.w3, opt.w4, opt.threshold)
log_dir = 'logs/' + session_string + '/' + opt.variant + '/'
if not os.path.isdir(log_dir): os.makedirs(log_dir)
train_log = open(log_dir + 'log_init_%s.txt' % (session_string), 'w', 1)

param_log = open(log_dir + 'param_init_%s.txt' % (session_string), 'w', 1)
param = ['train set: ', opt.datasets, '\n sampler id: ', str(opt.sampler_id),
				'\n learning rate: ', str(opt.learningrate),
				'\n loss: ', opt.loss,
				'\n', str(opt.w1), '\t', str(opt.w2), '\t', str(opt.w3), '\t', str(opt.w4)]
for p in param:
	param_log.write(p)

model_dir = log_dir + 'models/'
if not os.path.isdir(model_dir): os.makedirs(model_dir)

writer = SummaryWriter(log_dir +'vision_init', comment="model_vis")

# in the initalization we optimize the KLDiv of the predicted distribution and the target distgribution (see NG-RANSAC supplement A, Eq. 12)
distLoss = torch.nn.KLDivLoss(reduction='sum')

# main training loop
for epoch in range(0, opt.epochs):	

	print("=== Starting Epoch", epoch, "==================================")

	# store the network every so often
	torch.save(model.state_dict(), 'models/' + opt.datasets + '/weights_%s.net' % (session_string))

	# main training loop in the current epoch
	for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in trainset_loader:
	
		correspondences = correspondences.float()
		log_probs = model(correspondences.cuda())
		probs = torch.exp(log_probs).cpu()
		target_probs = torch.zeros(probs.size())
		#loop over batch
		for b in range(correspondences.size(0)):
			
			if opt.fmat:

				# === CASE FUNDAMENTAL MATRIX =========================================

				util.denormalize_pts(correspondences[b, 0:2], im_size1[b])
				util.denormalize_pts(correspondences[b, 2:4], im_size2[b])

				init.gtdist(correspondences[b], target_probs[b], gt_F[b], opt.threshold, True)
			else:

				# === CASE ESSENTIAL MATRIX =========================================

				init.gtdist(correspondences[b], target_probs[b], gt_E[b].float(), opt.threshold, False)
			
		log_probs.squeeze_()
		target_probs.squeeze_()

		# KL divergence
		loss = distLoss(log_probs, target_probs.cuda()) / correspondences.size(0) 

		# update model(gradient decending)
		loss.backward()
		optimizer.step() #update parameterstorch.utils.data.DataLoader
		optimizer.zero_grad() #gradient = 0

		print("Iteration: ", iteration, "Loss: ", float(loss))
		train_log.write('%d %f\n' % (iteration, loss))

		iteration += 1

		# for vision

		writer.add_scalar('loss',loss,global_step = iteration)
		writer.flush()

		del log_probs, probs, target_probs, loss
