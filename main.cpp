#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#define _PRESENTATION

int main(int argc, char** argv) {
    std::string filename;
    std::string outText;

    if (argc >= 2) {
        filename = argv[1];
    }
    else {
        filename = "test_plate.jpg";
    }

    cv::Mat img, gray, tresh;

    img = cv::imread(filename, cv::IMREAD_COLOR);
    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

    cv::resize(img, img, cv::Size(500, (500 * img.size[0]) / img.size[1]));

#ifdef _PRESENTATION
    namedWindow("Input image", cv::WINDOW_AUTOSIZE);
    imshow("Input image", img);
    cv::waitKey(0);
#endif

    // Grayify
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

#ifdef _PRESENTATION
    namedWindow("gray", cv::WINDOW_AUTOSIZE);
    imshow("gray", gray);
    cv::waitKey(0);
#endif

    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
#ifdef _PRESENTATION
    imshow("gray", gray);
    cv::waitKey(0);
#endif

    cv::erode(gray, gray, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

#ifdef _PRESENTATION
    imshow("gray", gray);
    cv::waitKey(0);
#endif

    cv::threshold(gray, tresh, 0, 255, cv::THRESH_BINARY_INV +  cv::THRESH_OTSU);
    
#ifdef _PRESENTATION
    namedWindow("tresh", cv::WINDOW_AUTOSIZE);
    imshow("tresh", tresh);
    cv::waitKey(0);
#endif

    // Getting contours

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(tresh, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours
#ifdef _PRESENTATION
    cv::namedWindow("contours", cv::WINDOW_AUTOSIZE);
#endif
    cv::Mat contourImg(tresh.size(), CV_8UC3, cv::Scalar(0,0,0));
    std::vector<cv::Rect> boundingRectangles;
    for (size_t i = 0; i < contours.size(); i++) {
#ifdef _PRESENTATION
        cv::drawContours(contourImg, contours, i, cv::Scalar(0, 255, 0)); // Draw contours
#endif
        cv::Rect boundingRect = cv::boundingRect(contours[i]);
#ifdef _PRESENTATION
        std::cout << "Height: " << boundingRect.height << " Width: " << boundingRect.width << std::endl;
#endif
        if (boundingRect.height >= 75 && boundingRect.height <= 85) {
            if (boundingRect.width >= 14 && boundingRect.width <= 58){
#ifdef _PRESENTATION
                cv::rectangle(contourImg, boundingRect, cv::Scalar(255, 0, 0));
#endif
                boundingRectangles.push_back(boundingRect);
            }
        }
#ifdef _PRESENTATION
        std::cout << "Area:" << cv::contourArea(contours[i]) << std::endl;
        cv::imshow("contours", contourImg);
        cv::waitKey(0);
#endif
    }

    // sorting chars from left to right
    sort(boundingRectangles.begin(), boundingRectangles.end(), [](const cv::Rect& lhs, const cv::Rect& rhs) {
        return lhs.x <  rhs.x;
    });

    api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    api->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);

    for (size_t i = 0; i < boundingRectangles.size(); i++) {
        cv::Mat character = gray(boundingRectangles[i]);
#ifdef _PRESENTATION
        cv::imshow(std::to_string(i), character);
#endif
        api->SetImage(character.data, character.cols, character.rows, 1, character.step);
        outText.append(api->GetUTF8Text());
#ifdef _PRESENTATION
        cv::waitKey(0);
#endif
    }
    

    std::cout << outText << std::endl;

    cv::waitKey(0);

    api->End();

    return 0;
}


