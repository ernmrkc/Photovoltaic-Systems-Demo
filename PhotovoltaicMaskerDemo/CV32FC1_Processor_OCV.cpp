#include "CV32FC1_Processor_OCV.h"

CV32FC1_Processor_OCV::CV32FC1_Processor_OCV(const string &imagePath)
{
    image_CV32FC1 = imread(imagePath, IMREAD_GRAYSCALE);
    if(image_CV32FC1.empty())
    {
        throw runtime_error("Could not load image: " + imagePath);
    }
    image_CV32FC1.convertTo(image_CV32FC1, CV_32F, 1.0 / 255.0);
}

CV32FC1_Processor_OCV::CV32FC1_Processor_OCV(const Mat &inputImage)
{
    if(inputImage.channels() > 1)
    {
        cvtColor(inputImage, image_CV32FC1, COLOR_BGR2GRAY);
    }
    image_CV32FC1.convertTo(image_CV32FC1, CV_32F, 1.0 / 255.0);
}

Mat CV32FC1_Processor_OCV::getImage() const
{
    return image_CV32FC1.clone();
}

void CV32FC1_Processor_OCV::performThreshold(float fThresholdValue)
{
    threshold(image_CV32FC1, image_CV32FC1, fThresholdValue, 1.0, THRESH_BINARY);
}

void CV32FC1_Processor_OCV::performConvolution(ConvolutionMode mode)
{
    Mat kernel;
    if(mode == Blur)
    {
        kernel = kernel_blur;
    }
    else if(mode == Sharpen)
    {
        kernel = kernel_sharpen;
    }

    Mat result;
    filter2D(image_CV32FC1, result, CV_32F, kernel);

    image_CV32FC1 = result;
}

void CV32FC1_Processor_OCV::performSobelEdgeDetection(SobelEdgeDetectionMode mode)
{
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F; 

    if(mode == Horizontal_X || mode == Both_XY)
    {
        Sobel(image_CV32FC1, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        convertScaleAbs(grad_x, abs_grad_x);
    }

    if(mode == Vertical_Y || mode == Both_XY)
    {
        Sobel(image_CV32FC1, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
        convertScaleAbs(grad_y, abs_grad_y);
    }

    if(mode == Both_XY)
    {
        Mat grad;
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        image_CV32FC1 = grad;
    }
    else if(mode == Horizontal_X)
    {
        image_CV32FC1 = abs_grad_x;
    }
    else if(mode == Vertical_Y)
    {
        image_CV32FC1 = abs_grad_y;
    }
}

void CV32FC1_Processor_OCV::performMorphologicalOperation(MorphologicalOperationMode mode, int nMorphCount, float fThresholdValue)
{
    Mat binaryImage;
    threshold(image_CV32FC1, binaryImage, fThresholdValue, 1.0, THRESH_BINARY);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat result = binaryImage.clone(); 

    switch(mode)
    {
        case Dilation:
            for(int i = 0; i < nMorphCount; ++i)
            {
                dilate(result, result, kernel);
            }
            break;
        case Erosion:
            for(int i = 0; i < nMorphCount; ++i)
            {
                erode(result, result, kernel);
            }
            break;
        case Edge:
            Mat dilated, eroded;
            dilate(binaryImage, dilated, kernel);
            erode(binaryImage, eroded, kernel);
            subtract(dilated, eroded, result);
            break;
    }

    image_CV32FC1 = result;
}

void CV32FC1_Processor_OCV::performMedianFiltering(int kernelSize)
{
    if(kernelSize % 2 == 0 || kernelSize < 1)
    {
        cerr << "Kernel size must be odd and positive." << endl;
        return;
    }

    Mat result;
    medianBlur(image_CV32FC1, result, kernelSize);

    image_CV32FC1 = result;
}

void CV32FC1_Processor_OCV::performLocallyAdaptiveThreshold(float fAdaptiveBias)
{
    double maxValue = 1.0;
    int blockSize = 5;
    double C = fAdaptiveBias - 1.0;

    Mat result;
    adaptiveThreshold(image_CV32FC1, result, maxValue, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, C);
    image_CV32FC1 = result;
}