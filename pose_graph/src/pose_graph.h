#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <stdio.h>
#include <ros/ros.h>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"


#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph
{
public:
	PoseGraph();					// 构造函数，初始化位姿图参数（如序列计数器、全局索引等）。
	~PoseGraph();					// 析构函数，释放资源（如关键帧内存、可视化对象等）。
	void registerPub(ros::NodeHandle &n);	// 注册 ROS 发布器（如路径、位姿图可视化话题）。
	void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);		// 核心函数​​：将实时关键帧插入位姿图，若 flag_detect_loop 为 true，触发回环检测。
	void loadKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);		// 从文件加载历史关键帧到位姿图，用于重启后的状态恢复。
	void loadVocabulary(std::string voc_path);		// 加载 DBoW2 词袋模型文件，加速回环检测中的特征匹配。
	void updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1 > &_loop_info);		// 更新检测到回环后的关键帧约束（如相对位姿、协方差）。
	KeyFrame* getKeyFrame(int index);		// 根据索引从 keyframelist 中获取关键帧对象。
	nav_msgs::Path path[10];				// 存储各序列（最多10个）的优化后路径，用于 RViz 可视化。
	nav_msgs::Path base_path;				// 基础序列（通常是第一个序列）的路径，用于多地图对齐。
	CameraPoseVisualization* posegraph_visualization;		// 可视化对象，管理相机位姿和回环连接的显示。
	void savePoseGraph();		// 将当前位姿图数据（关键帧位姿、回环约束）保存到磁盘。
	void loadPoseGraph();		// 从磁盘加载历史位姿图数据，恢复系统状态。
	void publish();				// 发布路径、关键帧位姿等可视化信息到 ROS 话题。
	Vector3d t_drift;			// 平移漂移量，用于多序列间的坐标系对齐。
	double yaw_drift;			// 偏航角漂移量，修正方向误差。
	Matrix3d r_drift;			// 旋转漂移矩阵，与 t_drift 共同描述序列间的变换。
	// world frame( base sequence or first sequence)<----> cur sequence frame  
	Vector3d w_t_vio;			// 世界坐标系到 VIO 估计的平移偏移（用于多地图融合）。
	Matrix3d w_r_vio;			// 世界坐标系到 VIO 估计的旋转偏移。


private:
	int detectLoop(KeyFrame* keyframe, int frame_index);
	void addKeyFrameIntoVoc(KeyFrame* keyframe);
	void optimize4DoF();
	void updatePath();
	list<KeyFrame*> keyframelist;		// ​​核心容器​​：存储所有关键帧的链表。
	std::mutex m_keyframelist;			// 保护 keyframelist 的互斥锁，防止多线程竞争。
	std::mutex m_optimize_buf;			// 保护优化队列 optimize_buf 的锁。
	std::mutex m_path;
	std::mutex m_drift;
	std::thread t_optimization;			// 优化线程，执行 optimize4DoF 函数。
	std::queue<int> optimize_buf;		// 待优化的关键帧索引队列。

	int global_index;
	int sequence_cnt;
	vector<bool> sequence_loop;			// 标记各序列是否已检测到回环。
	map<int, cv::Mat> image_pool;		// 缓存关键帧图像，用于回环检测时的特征匹配可视化。
	int earliest_loop_index;			// 最早检测到回环的关键帧索引，用于局部优化范围限制。
	int base_sequence;

	BriefDatabase db;					// 存储关键帧的 BRIEF 描述子数据库，用于快速匹配。
	BriefVocabulary* voc;				// DBoW2 词汇表，将描述子映射到视觉单词。

	ros::Publisher pub_pg_path;			// 发布优化后的全局路径。
	ros::Publisher pub_base_path;		// 发布基础序列路径。
	ros::Publisher pub_pose_graph;		// 发布位姿图结构（如关键帧、回环边）。
	ros::Publisher pub_path[10];		// 发布各子序列的路径。
};

template <typename T>
T NormalizeAngle(const T& angle_degrees) {
  if (angle_degrees > T(180.0))
  	return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
  	return angle_degrees + T(360.0);
  else
  	return angle_degrees;
};

class AngleLocalParameterization {
 public:

  template <typename T>
  bool operator()(const T* theta_radians, const T* delta_theta_radians,
                  T* theta_radians_plus_delta) const {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  static ceres::LocalParameterization* Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};

template <typename T> 
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

	T y = yaw / T(180.0) * T(M_PI);
	T p = pitch / T(180.0) * T(M_PI);
	T r = roll / T(180.0) * T(M_PI);


	R[0] = cos(y) * cos(p);
	R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
	R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
	R[3] = sin(y) * cos(p);
	R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
	R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
	R[6] = -sin(p);
	R[7] = cos(p) * sin(r);
	R[8] = cos(p) * cos(r);
};

template <typename T> 
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
	inv_R[0] = R[0];
	inv_R[1] = R[3];
	inv_R[2] = R[6];
	inv_R[3] = R[1];
	inv_R[4] = R[4];
	inv_R[5] = R[7];
	inv_R[6] = R[2];
	inv_R[7] = R[5];
	inv_R[8] = R[8];
};

template <typename T> 
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
	r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
	r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
	r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct FourDOFError
{
	FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
				  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){}

	template <typename T>
	bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x));
		residuals[1] = (t_i_ij[1] - T(t_y));
		residuals[2] = (t_i_ij[2] - T(t_z));
		residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          FourDOFError, 4, 1, 3, 1, 3>(
	          	new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;

};

struct FourDOFWeightError
{
	FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
				  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){
				  	weight = 1;
				  }

	template <typename T>
	bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
		residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
		residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
		residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) * T(weight) / T(10.0);

		return true;
	}

	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          FourDOFWeightError, 4, 1, 3, 1, 3>(
	          	new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
	double weight;

};