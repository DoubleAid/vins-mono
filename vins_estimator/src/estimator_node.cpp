#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;                                // 估计器

std::condition_variable con;
double current_time = -1;                           // -1 表示初始化第一个数据
queue<sensor_msgs::ImuConstPtr> imu_buf;            // imu数据缓存
queue<sensor_msgs::PointCloudConstPtr> feature_buf; // 图像特征缓存
queue<sensor_msgs::PointCloudConstPtr> relo_buf;    // 重定位数据缓存，回环检测
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;                              // 当前位置（世界坐标系下）。
Eigen::Quaterniond tmp_Q;                           // 当前姿态四元数（世界坐标系下）。
Eigen::Vector3d tmp_V;                              // 当前速度（世界坐标系下）。
Eigen::Vector3d tmp_Ba;                             // 加速度计和陀螺仪的偏置（需在优化器中估计）。
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;                              // 上一时刻的 ​​原始加速度计测量值​​（未去偏置、未转换坐标系）。
Eigen::Vector3d gyr_0;                              // 上一时刻的 ​​原始陀螺仪测量值​​（未去偏置）。
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// IMU 运动学传播 （传统积分，中值积分）
// predict 函数的作用是基于 ​​IMU 测量值​​ 进行 ​​状态预测​​，通过积分更新当前时刻的姿态、位置和速度。以下是其核心功能的分步解析：
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    // 如果是初始化状态，记录当前时间并返回
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    // 计算时间差 dt 并更新 latest_time
    double dt = t - latest_time;
    latest_time = t;

    // 提取 IMU 线性加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    // 提取 IMU 角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    // 加速度去偏置 + 转换到世界坐标系
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    // 角速度去偏置
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // 四元数更新
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
    // 加速度更新
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;
    // 平均加速度（中值积分）
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // 根据中值积分加速度计算当前位置和速度
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;
    // 更新上一次的数据
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    // 更新时间
    latest_time = current_time;
    // 更新最新的当前位姿
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    // 更新偏置
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    // 更新当前的重力加速度
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    // 更新当前的线速度和角速度
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    // 将队列里还没有进行处理的imu数据重新进行中值积分
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

// 获取匹配的图片和IMU
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        // 如果 imu 缓存或者 特征缓存为空，就认为已经结束了
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // 已经获取到图片了，但没有获取到IMU的信息
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // 已经找不到这个 特征的 IMU帧的时间了，就舍弃掉这个特征
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // 仍保留这个数据
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// IMU回调函数
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 如果时间乱序了，直接不要了
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    // 更新最新时间戳
    last_imu_t = imu_msg->header.stamp.toSec();

    // 将IMU数据存储到IMU队列中
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        // 预积分 
        // 执行 IMU 前向传播（预测）
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        // 系统已通过初始化，进入稳定的 ​​非线性优化阶段​​。此时发布的里程计是可靠的（未初始化时的数据可能不稳定，故不发布）。
        
        // IMU 数据 → 运动学传播 → 发布预测值 → 后端优化 → 修正预测值 → 发布优化值。
        
        // 在非线性优化阶段，通过高频发布 IMU 预测值，满足实时性需求。
        // 平衡实时性与精度，是实时 SLAM 系统的典型设计模式。
        
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// 特征点回调函数
// 将特征点存储到特征队列中
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// 图像中断，重新启动
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        // 晴空特征队列和IMU队列
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
// 视觉惯性里程计
void process()
{
    while (true)
    {
        // 存储测量数据，包含IMU数据队列和图像数据配对的列表。
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        // 等待条件变量 con，直到 getMeasurements() 获取到一批 ​​时间对齐的 IMU 和图像数据​​。
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            // 获取图片像素数据
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            // 遍历 imu
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                // 添加 相机和IMU的时间戳偏移值，进行时间同步
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                { 
                    // IMU的时间还没有到达照片的时间，就直接做 IMU的预积分

                    // 第一个数据
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;    // 更新IMU的线性加速度
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;       // 更新IMU的角速度
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // 处理 IMU 数据直到图像时间戳
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    // 如果 图片的时间 介于 上次IMU和本次IMU之间，就做线性插值获取图片时间IMU的积分值
                    double dt_1 = img_t - current_time;     // 左边的delta
                    double dt_2 = t - img_t;                // 右边的delta
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);       // 右边的比例
                    double w2 = dt_1 / (dt_1 + dt_2);       // 左边的比例
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;     // 左边乘上一个的加速度，右边乘本次的加速度 计算到图片是的预计份
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;     
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // set relocalization frame
            // 回环检测数据处理​
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            // 获取循环检测的最后一个缓存数据
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                // 解析匹配点（特征点对应关系）
                // 匹配点的列表的列表
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                // 获取位置信息
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                // 读取四元数
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                // 将四元数转换成矩阵
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                // 获取回环关键帧的索引
                frame_index = relo_msg->channels[0].values[7];
                // 设置回环帧
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            // 解析图像特征消息
            // 将当前帧的视觉特征输入状态估计器，用于：​视觉惯性对齐​​（初始化阶段）和 滑动窗口非线性优化​​（重投影误差 + IMU 预积分约束）。
            TicToc t_s;
            // image 最外边是特征的ID，然后后面是 一系列 camera id 和 特征点数据
            // 也就是同一特征不同相机观测的数据
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                // 解析特征ID和相机ID（编码在 channels[0] 中）
                // feature_id 唯一标识一个 ​​物理世界中的特征点​​。即使同一特征点被多个相机观测到，其 feature_id 是相同的。
                // camera_id 标识观测到特征点的 ​​相机编号​​，用于多相机系统（如双目、多目相机）。
                // 将 feature_id 和 camera_id 合并为一个整数 v。例如：
                // 若 NUM_OF_CAM = 2（双目系统），feature_id = 100 的特征点在左相机（camera_id = 0）的编码值为 100 * 2 + 0 = 200。
                // 解码时，v = 200 对应 feature_id = 200 / 2 = 100，camera_id = 200 % 2 = 0。
                int v = img_msg->channels[0].values[i] + 0.5;           // +0.5 用于四舍五入
                int feature_id = v / NUM_OF_CAM;                        // 特征ID
                int camera_id = v % NUM_OF_CAM;                         // 相机ID（多相机系统）
                double x = img_msg->points[i].x;                        // 获取归一化坐标 [x, y, 1]
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];            // 在像素上投影
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];     // 光流速度
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;       // 构建特征点数据 七维向量
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);        // 
            }
            // 执行视觉惯性紧耦合优化
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            // 发布估计结果
            pubOdometry(estimator, header);                             // 发布里程计位姿
            pubKeyPoses(estimator, header);                             // 发布关键帧位姿
            pubCameraPose(estimator, header);                           // 发布相机位姿 IMU坐标系到Camera位姿
            pubPointCloud(estimator, header);                           // 发布三维地图点
            pubTF(estimator, header);                                   // TF 变换，用于Rviz
            pubKeyframe(estimator);                                     // 关键帧数据，用于回环检测和建图
            if (relo_msg != NULL)
                pubRelocalization(estimator);                           // 发布回环检测结果
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        // 如果当前的状态已经是非线性优化状态，即正常状态，调用 update() 函数进行更新
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    // 初始化 ros 节点
    ros::init(argc, argv, "vins_estimator");
    // 创建一个 ​​ROS 节点句柄（NodeHandle）​​，并指定其命名空间为 ​​私有命名空间
    ros::NodeHandle n("~");
    // 设置日志级别为 Info，屏蔽 Debug 信息
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    // 从 ROS 参数服务器读取配置文件（如相机内参、IMU噪声）
    readParameters(n);  
    // 将参数加载到 Estimator 类中（例如设置重力、相机-IMU外参）
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // 初始化用于发布话题的 ROS Publisher（如轨迹、位姿、点云等）
    registerPub(n);

    // 订阅 IMU 数据（高频率，2000 为队列长度，使用 TCP 协议且禁用 Nagle 算法降低延迟）
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    
    // 订阅特征点数据（来自 feature_tracker 节点提取的特征）
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    
    // 订阅重启信号（当特征跟踪失败时，触发系统重置）
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);

    // 订阅回环检测的匹配点（用于重定位）
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    // 创建线程执行 process() 函数
    std::thread measurement_process{process};

    // ros 运行
    ros::spin();

    return 0;
}
