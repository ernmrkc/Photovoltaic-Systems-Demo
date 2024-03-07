#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "CV32FC1_Processor_OCV.h"

using namespace std;
using namespace cv;

int main()
{
    // Read Image
    string image_path = "Solar-Images\\Photovoltaic-System.jpg";
    Mat original_image = imread(image_path);
    imshow("original image", original_image);

    // Processing the image with OpenCV
    CV32FC1_Processor_OCV processor(original_image);
    processor.performThreshold(0.3);
    Mat processedImage = processor.getImage();
    imshow("processed image", processedImage);

    waitKey(0);
    destroyAllWindows();

}


