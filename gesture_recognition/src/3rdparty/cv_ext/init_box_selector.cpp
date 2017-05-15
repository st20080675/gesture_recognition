/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
// License Agreement
// For Open Source Computer Vision Library
// (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//M*/

/*
// Original file: https://github.com/Itseez/opencv_contrib/blob/292b8fa6aa403fb7ad6d2afadf4484e39d8ca2f1/modules/tracking/samples/tracker.cpp
// + Author: Klaus Haag
// * Refactor file: Move target selection to separate class/file
*/

#include "init_box_selector.hpp"

void InitBoxSelector::onMouse(int event, int x, int y, int, void*)
{
    if (!selectObject)
    {
        switch (event)
        {
        case cv::EVENT_LBUTTONDOWN:
            //set origin of the bounding box
            startSelection = true;
            initBox.x = x;
            initBox.y = y;
            break;
        case cv::EVENT_LBUTTONUP:
            //set width and height of the bounding box
            initBox.width = std::abs(x - initBox.x);
            initBox.height = std::abs(y - initBox.y);
            startSelection = false;
            selectObject = true;
            break;
        case cv::EVENT_MOUSEMOVE:
            if (startSelection && !selectObject)
            {
                //draw the bounding box
                cv::Mat currentFrame;
                image.copyTo(currentFrame);
                cv::rectangle(currentFrame, cv::Point((int)initBox.x, (int)initBox.y), cv::Point(x, y), cv::Scalar(255, 0, 0), 2, 1);
                cv::imshow(windowTitle.c_str(), currentFrame);
            }
            break;
        }
    }
}

bool InitBoxSelector::selectBox(cv::Mat& frame, cv::Rect& initBox)
{
    frame.copyTo(image);
    startSelection = false;
    selectObject = false;
    cv::imshow(windowTitle.c_str(), image);
    cv::setMouseCallback(windowTitle.c_str(), onMouse, 0);

    /*
    while (selectObject == false)
    {
        char c = (char)cv::waitKey(10);

        if (c == 27)
            return false;
    }

    initBox = InitBoxSelector::initBox;
    cv::setMouseCallback(windowTitle.c_str(), 0, 0);
    cv::destroyWindow(windowTitle.c_str());

    return true;
    */


    // ****sun ting******* use skin detection and face detection to auto init boundingBox
    bool flag_face_detected = 0;
    // ***** skin detection *******************
    std::ifstream skin_file("/home/sunting/Documents/program/Gesture_recognition/skindetector/skin_model_bool.txt", std::ios::in);
        double skin_prob_map[32768];
        for (int i = 0; i<32768; i++)
        {
            skin_file>>skin_prob_map[i];
        }
        skin_file.close();

        cv::Mat image_gray;
        cvtColor( image, image_gray, CV_BGR2GRAY );

        cv::Mat skin_mask = cv::Mat::zeros(image_gray.rows, image_gray.cols, image_gray.type());

        // the skin index is: 1+floor(R/8)+floor(G/8)*32+floor(B/8)*32*32
        // cv::Mat read in color image in BGR channel order

        for (int i_col = 0; i_col < image.cols; i_col++)
        {
            for (int i_row = 0; i_row < image.rows; i_row++)
            {
                // opencv store color image in BGR order
                int current_idx = (int)(floor(image.at<cv::Vec3b>(i_row,i_col)[2]/8)+floor(image.at<cv::Vec3b>(i_row,i_col)[1]/8)*32+floor(image.at<cv::Vec3b>(i_row,i_col)[0]/8)*32*32);
                skin_mask.at<uchar>(i_row,i_col) = skin_prob_map[current_idx];

            }

        }


       // face detection
        std::string face_cascade_dir = "/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml";
           //std::string face_cascade_dir = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml";
           //std::string face_cascade_dir = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";
           cv::CascadeClassifier face_cascade;
           if( !face_cascade.load( face_cascade_dir ) ){ printf("--(!)Error loading face cascade classifier\n"); return 0; };

           cv::Mat gray_image;
           std::vector< cv::Rect> face_bbx_vec;
           cvtColor(image, gray_image, CV_BGR2GRAY);
           face_cascade.detectMultiScale(gray_image, face_bbx_vec);

           int face_num = face_bbx_vec.size();
           int best_face_idx = 0;

           if (face_num > 0)
           {
               flag_face_detected = 1;

               if (face_num > 1)
               {
                   float face_score = countNonZero(skin_mask(face_bbx_vec[0]));
                   float temp_face_score;
                   for (int i=1; i<face_num; i++)
                   {
                       temp_face_score = countNonZero(skin_mask(face_bbx_vec[i]));

                       if (temp_face_score > face_score)
                       {
                           best_face_idx = i;
                           face_score = temp_face_score;
                       }
                   }
               }

               initBox.x = std::max((int)(face_bbx_vec[best_face_idx].x - 0.5*face_bbx_vec[best_face_idx].width),(int)0);
               initBox.y = std::max((int)(face_bbx_vec[best_face_idx].y - face_bbx_vec[best_face_idx].height/5),(int)0);
               initBox.width = std::min(int(face_bbx_vec[best_face_idx].width * 2), int(image.cols - initBox.x));
               initBox.height = std::min(int(face_bbx_vec[best_face_idx].width * 7), int(image.rows - initBox.y));

               std::cout<< "the initial bounding box is: " << initBox.x << ", " << initBox.y << ", " << initBox.width << ", " << initBox.height << std::endl;
           }

    std::cout<< "the face detection is finished" << std::endl;

    

    if (!flag_face_detected)
    {
        while (selectObject == false)
        {
            char c = (char)cv::waitKey(10);

            if (c == 27)
                return false;
        }

        initBox = InitBoxSelector::initBox;
        cv::setMouseCallback(windowTitle.c_str(), 0, 0);
        cv::destroyWindow(windowTitle.c_str());

    }

#ifdef  write_intBox_to_file // **************** write to the file **********************
    std::ofstream init_Box_file;
    init_Box_file.open("init_Box.txt",std::ios::app);

    time_t _tm =time(NULL );
    struct tm * curtime = localtime ( &_tm );

    init_Box_file << asctime(curtime);
    init_Box_file << initBox.x << "  " << initBox.y << "  " << initBox.width << "  " << initBox.height << std::endl;
    init_Box_file.close();
#endif // ********************* end of writing file ********************************

    return true;
}

const std::string InitBoxSelector::windowTitle = "Draw Bounding Box";
bool InitBoxSelector::startSelection = false;
bool InitBoxSelector::selectObject = false;
cv::Mat InitBoxSelector::image;
cv::Rect InitBoxSelector::initBox;
