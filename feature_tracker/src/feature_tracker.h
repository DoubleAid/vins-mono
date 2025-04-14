#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

// 判断一个特征点是不是在图像边界
// 用于过滤掉位于图像边缘之外的特征点，避免越界访问。
bool inBorder(const cv::Point2f &pt);

// 根据光流跟踪状态 status（1 成功，0 失败），删除跟踪失败的特征点或 ID。
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    // 处理新输入图像的主函数，执行光流跟踪、特征检测等。
    void readImage(const cv::Mat &_img,double _cur_time);
    // 创建掩码 mask，避免在已有特征点附近检测新点。
    void setMask();
    // 在图像中检测新特征点（如使用 cv::goodFeaturesToTrack），补充到 forw_pts 中，并分配新 ID。
    void addPoints();
    // 更新特征点 ID 和跟踪次数 track_cnt。可能逻辑
    bool updateID(unsigned int i);
    // 从标定文件（如 camera.yaml）加载相机内参和畸变参数，初始化 m_camera（相机模型）
    void readIntrinsicParameter(const string &calib_file);
    // 显示去畸变后的图像，用于调试畸变参数是否正确。
    void showUndistortion(const string &name);
    // 通过基础矩阵（Fundamental Matrix）和 RANSAC 剔除误匹配的特征点（外点剔除）。
    void rejectWithF();
    // 将像素坐标 cur_pts 去畸变，得到归一化坐标 cur_un_pts（无畸变的 3D 点，z=1）。
    void undistortedPoints();

    cv::Mat mask;             // 特征检测掩码，控制新特征点的生成位置。
    cv::Mat fisheye_mask;     // 鱼眼相机的掩码，屏蔽边缘畸变区域。
    // prev_img 上一帧图像。
    // cur_img：当前帧
    // forw_img：光流跟踪的目标帧
    cv::Mat prev_img, cur_img, forw_img;
    
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;      // 特征点坐标。
    vector<cv::Point2f> prev_un_pts, cur_un_pts;          // 去畸变后的归一化坐标点。
    vector<cv::Point2f> pts_velocity;                     // 特征点的速度，可能用于估计运动或预测下一帧的位置。也可以用于聚类区分静态和动态
    vector<int> ids;
    vector<int> track_cnt;                                // 跟踪计数，记录每个特征点被连续跟踪的帧数。
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;                        // 相机模型指针，用于去畸变和投影操作。
    double cur_time;
    double prev_time;

    static int n_id;                                      // 静态变量，可能用于生成唯一的特征ID。
};
