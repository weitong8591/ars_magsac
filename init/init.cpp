#include <torch/extension.h>
#include <opencv2/opencv.hpp>

#include <omp.h>

#include <iostream>
#include "opencv-five-point.h"
#include "opencv-fundam.h"

/**

from NG-RANSAC
* @brief Calculate a ground truth probability distribution for a set of correspondences.
*
* Method uses using the distance to a ground truth model which
* can be an essential matrix or a fundamental matrix.
* For more information, see paper supplement A, Eq. 12
*
* @param correspondences 3-dim float tensor, dim 1: xy coordinates of left and right image (normalized by calibration parameters when used for an essential matrix, i.e. f_mat=false), dim 2: correspondences, dim 3: dummy dimension
* @param out_probabilities 3-dim float tensor, dim 1: ground truth probability, dim 2: correspondences, dim 3: dummy dimension
* @param gt_model 2-dim float tensor, ground truth model, essential matrix or fundamental matrix
* @param threshold determines the softness of the distribution, the inlier threshold used at test time is a good choice
* @param f_mat indicator whether ground truth model is an essential or fundamental matrix)
* @return void
*/
void gtdist(
	at::Tensor correspondences,
	at::Tensor out_probabilities,
	at::Tensor gt_model,
	float threshold,
	bool f_mat
)
{
	// we compute the sequared error, so we use the squared threshold
	threshold *= threshold;

	int cCount = out_probabilities.size(1); // number of correspondences

	// access to PyTorch tensors
	at::TensorAccessor<float, 3> cAccess = correspondences.accessor<float, 3>();
	at::TensorAccessor<float, 3> pAccess = out_probabilities.accessor<float, 3>();
	at::TensorAccessor<float, 2> MAccess = gt_model.accessor<float, 2>();

	// read correspondences
	std::vector<cv::Point2d> pts1, pts2; // coordinates in image 1 and 2

	for(int c = 0; c < cCount; c++)
	{
		pts1.push_back(cv::Point2d(cAccess[0][c][0], cAccess[1][c][0]));
		pts2.push_back(cv::Point2d(cAccess[2][c][0], cAccess[3][c][0]));
	}

	// read essential matrix
	cv::Mat_<double> gtModel = cv::Mat_<double>::zeros(3, 3);

	for(int x = 0; x < gtModel.cols; x++)
	for(int y = 0; y < gtModel.rows; y++)
		gtModel(y, x) = MAccess[y][x];

	// compute epipolar errors
	cv::Mat gtErr;
	if(f_mat)
	{
		cv::Mat m1(pts1);
		m1.convertTo(m1, CV_32F);
		cv::Mat m2(pts2);
		m2.convertTo(m2, CV_32F);

		compute_fundamental_error(m1, m2, gtModel, gtErr);
	}
	else
		compute_essential_error(pts1, pts2, gtModel, gtErr);

	// compute ground truth correspondence weights (see paper supplement A, Eq. 12)
	std::vector<float> weights(gtErr.rows);
	float normalizer = std::sqrt(2 * 3.1415926 * threshold);
	float probSum = 0;

	for(int j = 0; j < gtErr.rows; j++)
	{
		weights[j] = std::exp(-gtErr.at<float>(j, 0) / 2 / threshold) / normalizer;
		probSum += weights[j];
	}

	// write out results
	for(int j = 0; j < gtErr.rows; j++)
		pAccess[0][j][0] = weights[j] / probSum;
}


// register C++ functions for use in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("gtdist", &gtdist, "Ground truth distribution for initialization for essential matrix estimation.");
}
