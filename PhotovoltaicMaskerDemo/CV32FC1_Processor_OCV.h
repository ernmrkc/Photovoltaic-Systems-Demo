#pragma once

#ifndef CV32FC1_PROCESSOR_OCV_H
#define CV32FC1_PROCESSOR_OCV_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class CV32FC1_Processor_OCV
{
private:
	/********************************/
	/*    Enumeration Definition    */
	/********************************/

	enum ConvolutionMode
	{
		Blur,
		Sharpen
	};

	enum SobelEdgeDetectionMode
	{
		Horizontal_X,
		Vertical_Y,
		Both_XY
	};

	enum MorphologicalOperationMode
	{
		Dilation,
		Erosion,
		Edge
	};

public:
	/**
	* Constructs a CV32FC1_Processor_OCV object and initializes it by loading an image from the specified path.
	* The loaded image is converted to a single-channel (grayscale) 32-bit floating point format.
	* @param imagePath Path to the image file to be loaded.
	*/
	explicit CV32FC1_Processor_OCV(const string &imagePath);
	
	/**
	 * Constructs a CV32FC1_Processor_OCV object and initializes it with the provided cv::Mat image.
	 * The input image is expected to be in a format compatible with the processor's operations,
	 * typically single-channel (grayscale) 32-bit floating point format.
	 * @param inputImage cv::Mat object containing the input image data.
	 */
	explicit CV32FC1_Processor_OCV(const Mat &inputImage);

	/**
	 * Retrieves the processed image stored within the CV32FC1_Processor_OCV object.
	 * The returned image is in a single-channel (grayscale) 32-bit floating point format.
	 * @return A cv::Mat object containing the processed image data.
	 */
	Mat getImage() const;

	/**
	 * Binarizes the stored image using the given threshold value.
	 * @param fThresholdValue Threshold for binarization [0.0, 1.0].
	 */
	void performThreshold(float fThresholdValue);

	/**
	 * Applies a convolution operation based on the specified mode.
	 * @param mode The convolution mode to apply (e.g., Blur, Sharpen).
	 */
	void performConvolution(ConvolutionMode mode);
	
	/**
	 * Performs Sobel edge detection on the image in the specified direction.
	 * @param mode Direction for edge detection (Horizontal, Vertical, or Both).
	 */
	void performSobelEdgeDetection(SobelEdgeDetectionMode mode);
	
	/**
	 * Applies a morphological operation to the image, such as dilation or erosion.
	 * @param mode Type of morphological operation (e.g., Dilation, Erosion).
	 * @param nMorphCount Number of times the operation is applied.
	 * @param fThresholdValue Threshold value used for preprocessing.
	 */
	void performMorphologicalOperation(MorphologicalOperationMode mode, int nMorphCount, float fThresholdValue);
	
	/**
	 * Applies median filtering to smooth the image.
	 * @param kernelSize Size of the kernel used for median filtering.
	 */
	void performMedianFiltering(int kernelSize);
	
	/**
	 * Applies locally adaptive thresholding to the image.
	 * @param fAdaptiveBias Bias factor for local threshold adjustment.
	 */
	void performLocallyAdaptiveThreshold(float fAdaptiveBias);

private:

	/*************************/
	/*    Mask Definition    */
	/*************************/

	Mat kernel_blur = (cv::Mat_<float>(3, 3) <<
					   0.0f, 0.125f, 0.0f,
					   0.125f, 0.5f, 0.125f,
					   0.0f, 0.125f, 0.0f);

	Mat kernel_sharpen = (cv::Mat_<float>(3, 3) <<
						  0.0f, -1.0f, 0.0f,
						  -1.0f, 5.0f, -1.0f,
						  0.0f, -1.0f, 0.0f);

	Mat image_CV32FC1;
};



#endif // CV32FC1_PROCESSOR_OCV_H
