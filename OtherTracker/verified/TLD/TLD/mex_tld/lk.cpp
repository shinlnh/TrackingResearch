#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <limits>
#include <vector>

#include "mex.h"

namespace {

constexpr int kWinSize = 4;
constexpr int kPatchSize = 10;

cv::Mat matlabGrayToMat(const mxArray* array)
{
    if (mxGetNumberOfDimensions(array) != 2) {
        mexErrMsgIdAndTxt("tld:lk:invalidImageDims", "Expected a 2D grayscale image.");
    }

    const mwSize rows = mxGetM(array);
    const mwSize cols = mxGetN(array);
    cv::Mat image(static_cast<int>(rows), static_cast<int>(cols), CV_8UC1);

    switch (mxGetClassID(array)) {
        case mxUINT8_CLASS: {
            const auto* src = static_cast<const unsigned char*>(mxGetData(array));
            for (mwSize c = 0; c < cols; ++c) {
                for (mwSize r = 0; r < rows; ++r) {
                    image.at<unsigned char>(static_cast<int>(r), static_cast<int>(c)) = src[r + c * rows];
                }
            }
            break;
        }
        case mxDOUBLE_CLASS: {
            const auto* src = mxGetPr(array);
            for (mwSize c = 0; c < cols; ++c) {
                for (mwSize r = 0; r < rows; ++r) {
                    double value = src[r + c * rows];
                    value = std::max(0.0, std::min(255.0, value));
                    image.at<unsigned char>(static_cast<int>(r), static_cast<int>(c)) = static_cast<unsigned char>(value);
                }
            }
            break;
        }
        case mxSINGLE_CLASS: {
            const auto* src = static_cast<const float*>(mxGetData(array));
            for (mwSize c = 0; c < cols; ++c) {
                for (mwSize r = 0; r < rows; ++r) {
                    float value = src[r + c * rows];
                    value = std::max(0.0f, std::min(255.0f, value));
                    image.at<unsigned char>(static_cast<int>(r), static_cast<int>(c)) = static_cast<unsigned char>(value);
                }
            }
            break;
        }
        default:
            mexErrMsgIdAndTxt(
                "tld:lk:invalidImageType",
                "Unsupported image type. Expected uint8, single, or double.");
    }

    return image;
}

std::vector<cv::Point2f> matlabPointsToVector(const mxArray* array)
{
    if (mxGetM(array) != 2) {
        mexErrMsgIdAndTxt("tld:lk:invalidPoints", "Expected a 2xN point matrix.");
    }

    const mwSize count = mxGetN(array);
    const double* data = mxGetPr(array);
    std::vector<cv::Point2f> points(count);
    for (mwSize i = 0; i < count; ++i) {
        points[i] = cv::Point2f(static_cast<float>(data[2 * i]), static_cast<float>(data[2 * i + 1]));
    }
    return points;
}

float patchNcc(const cv::Mat& imageI, const cv::Mat& imageJ, const cv::Point2f& pointI, const cv::Point2f& pointJ)
{
    cv::Mat patchI;
    cv::Mat patchJ;
    cv::getRectSubPix(imageI, cv::Size(kPatchSize, kPatchSize), pointI, patchI);
    cv::getRectSubPix(imageJ, cv::Size(kPatchSize, kPatchSize), pointJ, patchJ);

    cv::Mat patchI32;
    cv::Mat patchJ32;
    patchI.convertTo(patchI32, CV_32F);
    patchJ.convertTo(patchJ32, CV_32F);

    cv::Mat response;
    cv::matchTemplate(patchI32, patchJ32, response, cv::TM_CCOEFF_NORMED);
    return response.at<float>(0, 0);
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    const double nanValue = std::numeric_limits<double>::quiet_NaN();

    if (nrhs == 0) {
        mexPrintf("Lucas-Kanade\n");
        return;
    }

    const int mode = static_cast<int>(mxGetScalar(prhs[0]));
    if (mode == 0) {
        return;
    }

    if (mode != 2 || (nrhs != 5 && nrhs != 6)) {
        mexErrMsgIdAndTxt("tld:lk:usage", "Usage: lk(2, imgI, imgJ, ptsI, ptsJ[, level])");
    }

    const int level = (nrhs == 6) ? static_cast<int>(mxGetScalar(prhs[5])) : 5;

    const cv::Mat imageI = matlabGrayToMat(prhs[1]);
    const cv::Mat imageJ = matlabGrayToMat(prhs[2]);
    const std::vector<cv::Point2f> pointsI = matlabPointsToVector(prhs[3]);
    std::vector<cv::Point2f> pointsJ = matlabPointsToVector(prhs[4]);

    if (pointsI.size() != pointsJ.size()) {
        mexErrMsgIdAndTxt("tld:lk:pointCountMismatch", "ptsI and ptsJ must contain the same number of points.");
    }

    std::vector<unsigned char> statusForward;
    std::vector<unsigned char> statusBackward;
    std::vector<float> errorForward;
    std::vector<float> errorBackward;
    std::vector<cv::Point2f> backwardPoints = pointsI;

    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::calcOpticalFlowPyrLK(
        imageI, imageJ, pointsI, pointsJ, statusForward, errorForward,
        cv::Size(kWinSize, kWinSize), level, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    cv::calcOpticalFlowPyrLK(
        imageJ, imageI, pointsJ, backwardPoints, statusBackward, errorBackward,
        cv::Size(kWinSize, kWinSize), level, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

    const mwSize pointCount = static_cast<mwSize>(pointsI.size());
    plhs[0] = mxCreateDoubleMatrix(4, pointCount, mxREAL);
    double* output = mxGetPr(plhs[0]);

    for (mwSize i = 0; i < pointCount; ++i) {
        const mwSize offset = 4 * i;
        if (statusForward[i] && statusBackward[i]) {
            output[offset] = pointsJ[i].x;
            output[offset + 1] = pointsJ[i].y;
            output[offset + 2] = cv::norm(pointsI[i] - backwardPoints[i]);
            output[offset + 3] = patchNcc(imageI, imageJ, pointsI[i], pointsJ[i]);
        } else {
            output[offset] = nanValue;
            output[offset + 1] = nanValue;
            output[offset + 2] = nanValue;
            output[offset + 3] = nanValue;
        }
    }

    (void)nlhs;
}
