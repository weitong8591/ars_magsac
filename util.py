import numpy as np
import cv2
import math
import argparse


def normalize_pts(pts, im_size):
    """Normalize image coordinate using the image size.

    Pre-processing of correspondences before passing them to the network to be
    independent of image resolution.
    Re-scales points such that max image dimension goes from -0.5 to 0.5.
    In-place operation.

    Keyword arguments:
    pts -- 3-dim array conainting x and y coordinates in the last dimension, first dimension should have size 1.
    im_size -- image height and width
    """

    pts[0, :, 0] -= float(im_size[1]) / 2
    pts[0, :, 1] -= float(im_size[0]) / 2
    pts /= float(max(im_size))


def denormalize_pts(pts, im_size):
    """Undo image coordinate normalization using the image size.

    In-place operation.

    Keyword arguments:
    pts -- N-dim array conainting x and y coordinates in the first dimension
    im_size -- image height and width
    """
    pts *= max(im_size)
    pts[0] += im_size[1] / 2
    pts[1] += im_size[0] / 2


def AUC(losses, thresholds, binsize):
    """Compute the AUC up to a set of error thresholds.

    Return multiple AUC corresponding to multiple threshold provided.

    Keyword arguments:
    losses -- list of losses which the AUC should be calculated for
    thresholds -- list of threshold values up to which the AUC should be calculated
    binsize -- bin size to be used for the cumulative histogram when calculating the AUC, the finer the more accurate
    """

    bin_num = int(max(thresholds) / binsize)
    bins = np.arange(bin_num + 1) * binsize

    hist, _ = np.histogram(losses, bins)  # histogram up to the max threshold
    hist = hist.astype(np.float32) / len(losses)  # normalized histogram
    hist = np.cumsum(hist)  # cumulative normalized histogram

    # calculate AUC for each threshold
    return [np.mean(hist[:int(t / binsize)]) for t in thresholds]


def pose_error(R, gt_R, t, gt_t):
    """Compute the angular error between two rotation matrices and two translation vectors.

    Keyword arguments:
    R -- 2D numpy array containing an estimated rotation
    gt_R -- 2D numpy array containing the corresponding ground truth rotation
    t -- 2D numpy array containing an estimated translation as column
    gt_t -- 2D numpy array containing the corresponding ground truth translation
    """

    # calculate angle between provided rotations
    dR = np.matmul(R, np.transpose(gt_R))
    dR = cv2.Rodrigues(dR)[0]
    dR = np.linalg.norm(dR) * 180 / math.pi

    # calculate angle between provided translations
    dT = float(np.dot(gt_t.T, t))
    dT /= float(np.linalg.norm(gt_t))

    if dT > 1 or dT < -1:
        print("Domain warning! dT:", dT)
        dT = max(-1, min(1, dT))
    dT = math.acos(dT) * 180 / math.pi

    return dR, dT


def f_error(pts1, pts2, F, gt_F, threshold):
    """Compute multiple evaluaton measures for a fundamental matrix.

    Return (False, 0, 0, 0) if the evaluation fails due to not finding inliers for the ground truth model,
    else return() True, F1 score, % inliers, mean epipolar error of inliers).

    Follows the evaluation procedure in:
    "Deep Fundamental Matrix Estimation"
    Ranftl and Koltun
    ECCV 2018

    Keyword arguments:
    pts1 -- 3D numpy array containing the feature coordinates in image 1, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    pts2 -- 3D numpy array containing the feature coordinates in image 2, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    F -- 2D numpy array containing an estimated fundamental matrix
    gt_F -- 2D numpy array containing the corresponding ground truth fundamental matrix
    threshold -- inlier threshold for the epipolar error in pixels
    """

    EPS = 0.00000000001
    num_pts = pts1.shape[1]

    # 2D coordinates to 3D homogeneous coordinates
    hom_pts1 = np.concatenate((pts1[:, :, 0], np.ones((1, num_pts))), axis=0)
    hom_pts2 = np.concatenate((pts2[:, :, 0], np.ones((1, num_pts))), axis=0)

    def epipolar_error(hom_pts1, hom_pts2, F):
        """Compute the symmetric epipolar error."""
        res = 1 / np.linalg.norm(F.T.dot(hom_pts2)[0:2], axis=0)
        res += 1 / np.linalg.norm(F.dot(hom_pts1)[0:2], axis=0)
        res *= abs(np.sum(hom_pts2 * np.matmul(F, hom_pts1), axis=0))
        return res

    # determine inliers based on the epipolar error
    est_res = epipolar_error(hom_pts1, hom_pts2, F)
    gt_res = epipolar_error(hom_pts1, hom_pts2, gt_F)
    est_inliers = (est_res < threshold)
    gt_inliers = (gt_res < threshold)

    true_positives = est_inliers & gt_inliers
    gt_inliers = float(gt_inliers.sum())
    print(gt_inliers)
    if gt_inliers > 0:
        est_inliers = float(est_inliers.sum())
        true_positives = float(true_positives.sum())
        precision = true_positives / (est_inliers + EPS)
        recall = true_positives / (gt_inliers + EPS)
        F1 = 2 * precision * recall / (precision + recall + EPS)
        inliers = est_inliers / num_pts
        print(inliers, est_inliers, num_pts)
        epi_mask = (gt_res < 1)
        if epi_mask.sum() > 0:
            epi_error = float(est_res[epi_mask].mean())
        else:
            # no ground truth inliers for the fixed 1px threshold used for epipolar errors
            return False, 0, 0, 0
        return True, F1, inliers, epi_error
    else:
        # no ground truth inliers for the user provided threshold
        return False, 0, 0, 0


# orientation error calculation for E or F matrix

def orientation_error(pts1, pts2, M, ang, b):
    num_pts = pts1.shape[1]

    # 2D coordinates to 3D homogeneous coordinates

    hom_pts1 = np.concatenate((pts1[:, :, 0], np.ones((1, num_pts))), axis=0)
    hom_pts2 = np.concatenate((pts2[:, :, 0], np.ones((1, num_pts))), axis=0)

    # calculate the ang between n1 and n2
    l1 = M.T.dot(hom_pts2)[0:2]
    l2 = M.dot(hom_pts1)[0:2]

    n1 = [l1[0][b], l1[1][b]]
    n2 = [l2[0][b], l2[1][b]]

    n1_norm = 1 / np.linalg.norm(n1, axis=0)
    n1 = np.dot(n1, n1_norm)

    n2_norm = 1 / np.linalg.norm(n2, axis=0)
    n2 = np.dot(n2, n2_norm)

    alpha = np.arccos(n1.T.dot(n2))

    ori_error = abs(alpha - ang[b])

    return ori_error


def scale_error(pts1, pts2, M, scale_ratio, b):
    num_pts = pts1.shape[1]

    # 2D coordinates to 3D homogeneous coordinates
    hom_pts1 = np.concatenate((pts1[:, :, 0], np.ones((1, num_pts))), axis=0)
    hom_pts2 = np.concatenate((pts2[:, :, 0], np.ones((1, num_pts))), axis=0)

    # calculate the ang between n1 and n2
    l1 = M.T.dot(hom_pts2)[0:2]
    l2 = M.dot(hom_pts1)[0:2]

    l1_norm = np.linalg.norm(np.dot(scale_ratio[b], l1))
    l2_norm = np.linalg.norm(l2)
    # n1 = [l1[0][b], l1[1][b]]
    # n2 = [l2[0][b], l2[1][b]]

    # n1_norm = 1 / np.linalg.norm(n1, axis=0)
    # n1 = np.dot(n1, n1_norm)

    # n2_norm = 1 / np.linalg.norm(n2, axis=0)
    # n2 = np.dot(n2, n2_norm)

    return abs(l1_norm - l2_norm)


def rootSift(desc):
    """Apply root sift normalization to a given set of descriptors.

    See details in:
    "Three Things Everyone Should Know to Improve Object Retrieval"
    Arandjelovic and Zisserman
    CVPR 2012

    Keyword arguments:
    desc -- 2D numpy array containing the descriptors in its rows
    """

    desc_norm = np.linalg.norm(desc, ord=1, axis=1)
    desc_norm += 1  # avoid division by zero
    desc_norm = np.expand_dims(desc_norm, axis=1)
    desc_norm = np.repeat(desc_norm, desc.shape[1], axis=1)

    desc = np.divide(desc, desc_norm)
    return np.sqrt(desc)


def create_parser(description):
    """Create a default command line parser with the most common options.

    Keyword arguments:
    description -- description of the main functionality of a script/program
    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fmat', '-fmat', action='store_true',
                        help='estimate the fundamental matrix, instead of the essential matrix')

    parser.add_argument('--rootsift', '-rs', action='store_true',
                        help='use RootSIFT normalization')

    parser.add_argument('--orb', '-orb', action='store_true',
                        help='use ORB instead of SIFT')

    parser.add_argument('--nfeatures', '-nf', type=int, default=2000,
                        help='fixes number of features by clamping/replicating, set to -1 for dynamic feature count but then batchsize (-bs) has to be set to 1')

    parser.add_argument('--batchmode', '-bm', action='store_true',
                        help='loop over all test datasets defined in util.py')

    parser.add_argument('--ratio', '-r', type=float, default=1.0,
                        help='apply Lowes ratio filter with the given ratio threshold, 1.0 does nothing')

    parser.add_argument('--nosideinfo', '-nos', action='store_true',
                        help='Do not provide side information (matching ratios) to the network. The network should be trained and tested consistently.')

    parser.add_argument('--threshold', '-t', type=float, default=0.001,
                        help='inlier threshold. Recommended values are 0.001 for E matrix estimation, and 0.1 or 1.0 for F matrix estimation')

    parser.add_argument('--resblocks', '-rb', type=int, default=12,
                        help='number of res blocks of the network')

    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='batch size')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, useful to separate different runs of a script')

    # parser.add_argument('--ransac', '-ra',
    #                    choices=['ransac', 'ngransac', 'pymagsac'], default='pymagsac',
    #                    help='which ransac method you use')

    parser.add_argument('--datasets', '-ds', default='kitti_new',
                        help='which datasets to use, separate multiple datasets by comma')

    parser.add_argument('--variant', '-v', default='train',
                        help='subfolder of the dataset to use')

    parser.add_argument('--hyps', '-hyps', type=int, default=16,
                        help='number of hypotheses, init.e. number of RANSAC iterations')

    parser.add_argument('--evalbinsize', '-eb', type=float, default=5,
                        help='bin size when calculating the AUC evaluation score, 5 was used by Yi et al., and therefore also in the NG-RANSAC paper for reasons of comparability; for accurate AUC values, set to e.g. 0.1')

    parser.add_argument('--samplecount', '-ss', type=int, default=4,
                        help='number of samples when approximating the expectation')

    parser.add_argument('--learningrate', '-lr', type=float, default=0.00001,
                        help='learning rate')

    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs')

    parser.add_argument('--model', '-m', default='',
                        help='load a model to contuinue training or leave empty to create a new model')

    parser.add_argument('--loss', '-l', choices=['inliers', 'f1', 'all', 'epi'], default='all',
                        help='Loss to use as a reward signal; '
                             '"all" means using weighted sum of errors, '
                             'e.g., w1=1 means only pose rror, max of translational and rotational angle error, '
                             "with non-zero w3, w4 means combing our proposed affine  loss."
                             '"inliers" maximizes the inlier count (self-supervised training), '
                             '"f1" is the alignment of estimated inliers and ground truth inliers (only for fundamental matrixes, '''
                             'init.e. -fmat), '
                             '"epi" is the mean epipolar error of inliers to ground truth epi lines (only for fundamental matrixes, init.e. -fmat)')

    parser.add_argument('--variance', '-var', type=float, default=0.8,
                        help='subfolder of the dataset to use')

    #parser.add_argument('--plus', '-pl', action='store_true',
    #                    help='define we use magsac plus plus or not')

    parser.add_argument('--w1', '-w1', type=float, default=1.0,
                        help='the weight of pose error')

    parser.add_argument('--w2', '-w2', type=float, default=0.0,
                        help='the weight of epipolar error')

    parser.add_argument('--w3', '-w3', type=float, default=0.0,
                        help='the weight of orientation error')

    parser.add_argument('--w4', '-w4', type=float, default=0.0,
                        help='the weight of scale error')

    parser.add_argument('--sampler_id', '-id', type=int, default=1,
                        help='which sampler you use')

    parser.add_argument('--network', '-net', type=int, default=0,
                        help='which network you train on, 0--ResNet like NGRANSAC, 1--DenseNet, 2--CLNet, 3 ResNet+GNN')

    parser.add_argument('--src_pth', '-src', default='traindata/',
                        help='source path of the correspondences')

    parser.add_argument('-img_pth', '-isrc', default='traindata/',
                        help='source path')

    parser.add_argument('--pairnum', '-pn', type=int, default=5000,
	                    help='number of pairs you want to test')

    return parser


def create_session_string(prefix, network_id, sampler_id, epochs, fmat, orb, rootsift, ratio, session, w1, w2, w3, w4, threshold):
    """Create an identifier string from the most common parameter options.

    Keyword arguments:
    prefix -- custom string appended at the beginning of the session string
    sampler_id -- the idddenticcation of which sample you use
    epochs -- how many epochs you trained
    fmat -- bool indicating whether fundamental matrices or essential matrices are estimated
    orb -- bool indicating whether ORB features or SIFT features are used
    rootsift -- bool indicating whether RootSIFT normalization is used
    ratio -- threshold for Lowe's ratio filter
    session -- custom string appended at the end of the session string
    """
    session_string = prefix + '_'
    session_string += 'net_'+str(network_id) + '_'
    session_string += 'sampler_'+str(sampler_id) + '_'
    session_string += 'epoch_'+str(epochs) + '_'
    if fmat:
        session_string += 'F_'
    else:
        session_string += 'E_'

    if orb: session_string += 'orb_'
    if rootsift: session_string += 'rs_'
    session_string += 'r%.2f_' % ratio
    session_string += 't%.2f_' % threshold
    if (w1 != 0): session_string += 'w1_%.2f_' % w1
    if (w2 != 0): session_string += 'w2_%.2f_' % w2
    if (w3 != 0): session_string += 'w3_%.2f_' % w3
    if (w4 != 0): session_string += 'w4_%.2f_' % w4
    # if des: session_string += 'des_'
    session_string += session

    return session_string


# list of all test datasets for Deep MAGSAC++
outdoor_test_datasets = [
    'buckingham_palace',
    'brandenburg_gate',
    'colosseum_exterior',
    'grand_place_brussels',
    'notre_dame_front_facade',
    'palace_of_westminster',
    'pantheon_exterior',
    'prague_old_town_square',
    'sacre_coeur',
    'taj_mahal',
    'trevi_fountain',
    'westminster_abbey'
]


test_datasets = outdoor_test_datasets
