#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1;


extern std::string IMAGE_TOPIC;             // 图片的topic
extern std::string IMU_TOPIC;               // IMU的topic
extern std::string FISHEYE_MASK;            // 鱼眼掩模
extern std::vector<std::string> CAM_NAMES;  // 相机名称数组
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;                            // 每秒更新的帧数
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;                    // 立体追踪
extern int EQUALIZE;                        // 是否对图片进行均衡化
extern int FISHEYE;                         // 是否为鱼眼相机
extern bool PUB_THIS_FRAME;                 // 是否发送这一帧

void readParameters(ros::NodeHandle &n);
