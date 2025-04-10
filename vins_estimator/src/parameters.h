#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

// 相机参数
const double FOCAL_LENGTH = 460.0;          // 焦距（像素单位，根据相机标定结果修改）
const int WINDOW_SIZE = 10;                 // 滑动窗口大小（保留的关键帧数）
const int NUM_OF_CAM = 1;                   // 相机数量（1表示单目）
const int NUM_OF_F = 1000;                  // 最大跟踪特征点数
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;                   // 特征点初始深度（单目初始化时使用，单位米）
extern double MIN_PARALLAX;                 // 最小视差（低于此值不触发初始化，单位像素）
extern int ESTIMATE_EXTRINSIC;              // 外参标定模式

extern double ACC_N, ACC_W;                 // ACC_N 加速度计白噪声（标定值，单位 m/s²/√Hz） ACC_W 加速度计随机游走噪声（标定值，单位 m/s³/√Hz）
extern double GYR_N, GYR_W;                 // GYR_N 陀螺仪白噪声（标定值，单位 rad/s/√Hz） GYR_W 陀螺仪随机游走噪声（标定值， unit rad/s²/√Hz）

extern std::vector<Eigen::Matrix3d> RIC;    // 相机到IMU的旋转外参
extern std::vector<Eigen::Vector3d> TIC;    // 相机到IMU的平移外参
extern Eigen::Vector3d G;                   // 重力向量（初始化为 [0, 0, 9.8]）

extern double BIAS_ACC_THRESHOLD;           // 加速度计偏置异常阈值（单位 m/s²）
extern double BIAS_GYR_THRESHOLD;           // 陀螺仪偏置异常阈值（单位 rad/s）
extern double SOLVER_TIME;                  // 单次优化最大时间（秒）
extern int NUM_ITERATIONS;                  // 单次优化最大迭代次数
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;        // 轨迹输出路径（TUM格式）
extern std::string IMU_TOPIC;               // 
extern double TD;                           // IMU与相机的时间戳偏移（单位秒）
extern double TR;                           // 卷帘快门每行曝光时间（单位秒）
extern int ESTIMATE_TD;                     // 是否在线估计时间偏移（0-不估计，1-估计）
extern int ROLLING_SHUTTER;                 // 是否为卷帘快门相机（0-全局快门，1-卷帘快门）
// 相机参数，高度和参数
extern double ROW, COL;


void readParameters(ros::NodeHandle &n);

// 定义优化变量中不同状态量的​​参数化维度​​（用于 Ceres 或 g2o 优化时的参数块大小）。
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,                          // 位姿参数化维度：旋转（四元数，4D） + 平移（3D）
    SIZE_SPEEDBIAS = 9,                     // 速度与零偏参数化维度：速度（3D） + 加速度计零偏（3D） + 陀螺仪零偏（3D）
    SIZE_FEATURE = 1                        // 特征点参数化维度：逆深度（1D）
};

// 定义 ​​IMU 状态向量​​ 的存储顺序和索引偏移量。
// 一个 IMU 状态量包含 15 维参数
enum StateOrder
{
    O_P = 0,                                // 平移（Position）的起始索引（0~2）
    O_R = 3,                                // 旋转（Rotation，李代数 so3）的起始索引（3~5）
    O_V = 6,                                // 速度（Velocity）的起始索引（6~8）
    O_BA = 9,                               // 加速度计零偏（Accelerometer Bias）的起始索引（9~11）
    O_BG = 12                               // 陀螺仪零偏（Gyroscope Bias）的起始索引（12~14）
};

// 定义 ​​IMU 噪声向量​​ 的存储顺序和索引偏移量。
enum NoiseOrder
{
    O_AN = 0,                               // 加速度计白噪声（Accelerometer Noise）起始索引（0~2）
    O_GN = 3,                               // 陀螺仪白噪声（Gyroscope Noise）起始索引（3~5）
    O_AW = 6,                               // 加速度计随机游走噪声（Accelerometer Random Walk）起始索引（6~8）
    O_GW = 9                                // 陀螺仪随机游走噪声（Gyroscope Random Walk）起始索引（9~11）
};
