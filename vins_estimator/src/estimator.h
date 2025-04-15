#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator
{
  public:
    Estimator();

    // 用于设置算法参数，比如相机内参、IMU噪声参数等。
    void setParameter();

    // interface
    // 处理 IMU 数据，执行 ​​预积分​​ 和 ​​运动学传播​​，更新当前帧状态（速度、位置、姿态）。
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    // 处理图像数据，完成 ​​特征跟踪​​、​​初始化​​、​​滑动窗口管理​​、​​非线性优化​​。
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    // 设置重定位信息（匹配特征点、参考帧位姿），用于 ​​全局位姿校正​​。
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    // 重置系统状态（清除滑动窗口、预积分、特征跟踪等），用于系统重启或故障恢复。
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    // 标记优化阶段（INITIAL 或 NON_LINEAR），控制初始化与非线性优化的切换。
    SolverFlag solver_flag;
    // 处理边缘化策略，决定是边缘化旧帧还是次新帧。
    MarginalizationFlag  marginalization_flag;
    // 重力向量（世界坐标系），初始化为 [0, 0, 9.81]，在优化中被估计。
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    // 相机到 IMU 的外参（旋转和平移），在标定后固定或作为优化变量。
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    // 滑动窗口中各帧的 ​​位置​​、​​速度​​、​​旋转矩阵​​（世界坐标系）。
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];

    // 滑动窗口中各帧的​​加速度计零偏​​
    Vector3d Bas[(WINDOW_SIZE + 1)];
    
    // 滑动窗口中各帧的陀螺仪零偏​​。
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    
    // 相机与 IMU 的 ​​时间偏移​​，用于时间同步校准。
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    // 存储两帧图像间的 IMU 预积分量（相对运动增量）。
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // 当前滑动窗口中的帧数。
    int frame_count;
    // 统计特征点跟踪中的异常情况。
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    // 管理特征点（跟踪、三角化、构建视觉残差）。
    FeatureManager f_manager;
    // 运动估计器，可能用于计算相对运动或初始位姿。
    MotionEstimator m_estimator;
    // 初始化相机到IMU的旋转外参。
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    // 回环检测相关的窗口索引。
    int loop_window_index;

    // 存储上一次边缘化的信息和参数块，用于先验约束。
    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;
    // 存储所有图像帧的信息，可能用于初始化或重定位。
    map<double, ImageFrame> all_image_frame;
    // 临时预积分器，可能用于初始化阶段。
    IntegrationBase *tmp_pre_integration;

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
