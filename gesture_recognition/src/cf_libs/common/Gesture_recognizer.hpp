#ifndef GESTURE_RECOGNIZER_HPP
#define GESTURE_RECOGNIZER_HPP

#define Gesture_recognition
//#define write_time_to_file


#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/traits.hpp>
#include <memory>
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <thread>


// ** gesture recognition class definition ****************************
#ifdef Gesture_recognition
class Gesture_recognizer
{

private:

    bool m_flag_use_opt_flow = 0;
    bool m_flag_face_detected = 0;
    bool m_flag_xy_extend_hand = 0;
    bool m_flag_xy_command = 0;
    bool m_flag_down_command = 0;
    int m_z_command_proposal = 0;


    cv::Vec2d m_bx_extend_margin = cv::Vec2d(1.6,0.2);
    float m_hand_bbx_to_width_tracking_bbx = 0.25;
    float m_static_position_dis_threshold_to_min_tracking_bbx = 0.6;
    float m_horizontal_dis_threshold_to_min_tracking_bbx = 0.3;
    float m_scale_smooth_ratio_hand_size = 0.2;
    double m_skin_weight_motion = 0.013;
    double m_skin_weight_static = 0.5;
    double m_threshold_face_skin = 0.1;
    double m_valid_state_ratio = 0.6;
    int m_max_body_height;
    int m_max_body_width;

    int m_state_num = 60;
    int m_max_position_status[60][3] ;
    int m_front_hand_status[60][3];
    float m_face_scale[2] = {0, 0};
    float m_tracking_scale[2] = {0, 0};

public:

    cv::Mat m_potential_hand_mask;
    cv::Mat m_image_with_result;
    cv::Rect m_tracking_bbx_extend;
    cv::Rect m_face_bbx;
    cv::Rect m_face_bbx_extend;
    cv::Point m_tracking_bbx_center;
    cv::Point m_command_center;
    cv::Vec2f m_hand_size;
    cv::Point3f m_command_proposal;
    //cv::Vec2f m_max_position_states;
    cv::Vec2f m_obj_motion;
    // for optical flow;
    cv::UMat m_prev_gray_crop;
    cv::Mat m_flow;
    cv::Mat m_flow_crop;

    Gesture_recognizer();
    ~Gesture_recognizer();

    void reinitial();
    void run(const cv::Mat& image, cv::Rect_<float> &boundingBox, float &newScale, cv::Point_<float> &newPos);
    void calculate_tracking_bbx_extend(const cv::Mat& image, cv::Rect_<float> &boundingBox);
    void face_detection(const cv::Mat& image, const cv::Rect_<float> &boundingBox);
    void recalculate_bbx_and_point_using_face_bbx(const cv::Mat& image, cv::Rect_<float> &boundingBox, float &newScale, cv::Point_<float> &newPos);
    cv::Mat calculate_potential_hand_mask(const cv::Mat& image);
    void skin_detection(const cv::Mat& image, const cv::Rect_<float> &boundingBox);
    void calculate_optical_flow(const cv::Mat& image);
    void state_buffer_management();
    void hand_detection(const cv::Rect_<float> &boundingBox);
    void command_generator(const cv::Rect_<float> &boundingBox);
    void command_regularization();
    void drawCommand(const cv::Rect_<float> &boundingBox);

};


#endif



// ** end of gesture recognition class definition *********************
#endif // GESTURE_RECOGNIZER_HPP
