#include "Estimator/Estimator.h"
#include "pcl/io/ply_io.h"
#include "pcl/filters/passthrough.h"
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Quaternion.h>
#include "cyber_msgs/Heading.h"
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <geographic_msgs/GeoPointStamped.h>

typedef pcl::PointXYZINormal PointType;


//全局点云显示
pcl::VoxelGrid<PointType> downFullMap;
pcl::VoxelGrid<PointType> downLocalMap;
pcl::PointCloud<PointType>::Ptr globalMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr globalMapDS(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr readinMap(new pcl::PointCloud<PointType>());
std::mutex _mutexMapQueue;
std::queue<pcl::PointCloud<PointType>> _globalMapQueue;

//局部子图存储
bool record_data = false;
std::ofstream poseFile;
std::ofstream SlamPoseevoFile;
std::ofstream GpsPoseevoFile;
std::ofstream SlamPoseFile;
std::ofstream GpsPoseFile;
std::string savePath = "/home/lzg/lab/Learn/lio-livox_ws/src/globalmap_data/";
int mapNUm = 0;
int saveMapNum = 0;
std::mutex _mutexLocalMapQueue;
pcl::VoxelGrid<PointType> downlocalMap;
pcl::PassThrough<PointType> disFilter;
double last_keypose[2]={0,0};
Eigen::Matrix4d transformLocToMap = Eigen::Matrix4d::Identity();
Eigen::Quaterniond quatLocToMap = Eigen::Quaterniond::Identity();
Eigen::Vector3d poseLocToMap = Eigen::Vector3d::Zero();

pcl::PointCloud<PointType>::Ptr localMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr localMapDS(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr localMapDSInv(new pcl::PointCloud<PointType>());
std::queue<Eigen::Matrix4d> _scanPoseQueue;
std::queue<Eigen::Quaterniond> _scanQuatQueue;
std::queue<Eigen::Vector3d> _scanVectQueue;
std::queue<double> _scanTimeQueue;

int WINDOWSIZE;
bool LidarIMUInited = false;
boost::shared_ptr<std::list<Estimator::LidarFrame>> lidarFrameList;
pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
Estimator* estimator;

ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserOdometryPath;
ros::Publisher pubFullLaserCloud;
ros::Publisher pubOutlierLaserCloud;
tf::StampedTransform laserOdometryTrans;
tf::TransformBroadcaster* tfBroadcaster;
ros::Publisher pubGps;
ros::Publisher pubGlobalMap;
ros::Publisher pubCurScan;

bool newfullCloud = false;

Eigen::Matrix4d transformAftMapped = Eigen::Matrix4d::Identity();

std::mutex _mutexLidarQueue;
std::queue<sensor_msgs::PointCloud2ConstPtr> _lidarMsgQueue;
std::mutex _mutexIMUQueue;
std::queue<sensor_msgs::ImuConstPtr> _imuMsgQueue;
Eigen::Matrix4d exTlb;
Eigen::Matrix3d exRlb, exRbl;
Eigen::Vector3d exPlb, exPbl;
Eigen::Vector3d GravityVector;
float filter_parameter_corner = 0.2;
float filter_parameter_surf = 0.4;
int IMU_Mode = 2;
sensor_msgs::NavSatFix gps;
int pushCount = 0;
double startTime = 0;

nav_msgs::Path laserOdoPath;

/** \brief publish odometry infomation
  * \param[in] newPose: pose to be published
  * \param[in] timefullCloud: time stamp
  */
void pubOdometry(const Eigen::Matrix4d& newPose, double& timefullCloud){
  nav_msgs::Odometry laserOdometry;

  Eigen::Matrix3d Rcurr = newPose.topLeftCorner(3, 3);
  Eigen::Quaterniond newQuat(Rcurr);
  Eigen::Vector3d newPosition = newPose.topRightCorner(3, 1);
  laserOdometry.header.frame_id = "/world";
  laserOdometry.child_frame_id = "/livox_frame";
  laserOdometry.header.stamp = ros::Time().fromSec(timefullCloud);
  laserOdometry.pose.pose.orientation.x = newQuat.x();
  laserOdometry.pose.pose.orientation.y = newQuat.y();
  laserOdometry.pose.pose.orientation.z = newQuat.z();
  laserOdometry.pose.pose.orientation.w = newQuat.w();
  laserOdometry.pose.pose.position.x = newPosition.x();
  laserOdometry.pose.pose.position.y = newPosition.y();
  laserOdometry.pose.pose.position.z = newPosition.z();
  pubLaserOdometry.publish(laserOdometry);

  geometry_msgs::PoseStamped laserPose;
  laserPose.header = laserOdometry.header;
  laserPose.pose = laserOdometry.pose.pose;
  laserOdoPath.header.stamp = laserOdometry.header.stamp;
  laserOdoPath.poses.push_back(laserPose);
  laserOdoPath.header.frame_id = "/world";
  pubLaserOdometryPath.publish(laserOdoPath);

  laserOdometryTrans.frame_id_ = "/world";
  laserOdometryTrans.child_frame_id_ = "/livox_frame";
  laserOdometryTrans.stamp_ = ros::Time().fromSec(timefullCloud);
  laserOdometryTrans.setRotation(tf::Quaternion(newQuat.x(), newQuat.y(), newQuat.z(), newQuat.w()));
  laserOdometryTrans.setOrigin(tf::Vector3(newPosition.x(), newPosition.y(), newPosition.z()));
  tfBroadcaster->sendTransform(laserOdometryTrans);

	gps.header.stamp = ros::Time().fromSec(timefullCloud);
	gps.header.frame_id = "world";
	gps.latitude = newPosition.x();
	gps.longitude = newPosition.y();
	gps.altitude = newPosition.z();
	gps.position_covariance = {
					Rcurr(0, 0), Rcurr(1, 0), Rcurr(2, 0),
					Rcurr(0, 1), Rcurr(1, 1), Rcurr(2, 1),
					Rcurr(0, 2), Rcurr(1, 2), Rcurr(2, 2)
	};
	pubGps.publish(gps);

}

void fullCallBack(const sensor_msgs::PointCloud2ConstPtr &msg){
  // push lidar msg to queue
	std::unique_lock<std::mutex> lock(_mutexLidarQueue);
  _lidarMsgQueue.push(msg);
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg){
  // push IMU msg to queue
  std::unique_lock<std::mutex> lock(_mutexIMUQueue);
  _imuMsgQueue.push(imu_msg);
}

double gps_yaw = 0;
double gps_timestamp;
void gpsFixCallBack(const sensor_msgs::NavSatFixConstPtr &fix_msg){
    gps_timestamp = fix_msg->header.stamp.toSec();
    geographic_msgs::GeoPointStampedPtr gps_msg(new geographic_msgs::GeoPointStamped());
    gps_msg->header = fix_msg->header;
    gps_msg->position.latitude = fix_msg->latitude;
    gps_msg->position.longitude = fix_msg->longitude;
    gps_msg->position.altitude = fix_msg->altitude;

    geodesy::UTMPoint utm_point;
    geodesy::fromMsg(gps_msg->position, utm_point);
    double temp_x,temp_y,temp_z;
    double temp_rx,temp_ry,temp_rz; 
    temp_x = utm_point.easting - 349350; //SJTU
    temp_y = utm_point.northing - 3432459;
    temp_z=0;

    geometry_msgs::Quaternion temp_quat = tf::createQuaternionMsgFromRollPitchYaw(0,0,gps_yaw);
    GpsPoseevoFile << std::fixed << std::setprecision(4)                
            << fix_msg->header.stamp.toSec() << ' '
            << temp_x << ' '
            << temp_y << ' '
            << 0 << ' '
            // << temp_quat.x << ' '
            // << temp_quat.y << ' '
            // << temp_quat.z << ' '
            // << temp_quat.w << std::endl;
            << 0<< ' '
            << 0<< ' '
            << 0 << ' '
            << 1 << std::endl;
    GpsPoseFile<< std::fixed << std::setprecision(4) 
            << fix_msg->header.stamp.toSec() << ' '
            << temp_x << ' '
            << temp_y << ' '
            << 0 << ' '
            << 0 << ' '
            << 0 << ' '
            << gps_yaw<< ' '
            << std::endl;

}  

void gpdHeadCallBack(const cyber_msgs::HeadingConstPtr &heading_msg) {
  double temp_rx = 0.0, temp_ry = 0.0, temp_rz;
  gps_yaw = heading_msg->data;
  if (gps_yaw >= M_PI)
    gps_yaw -= 2 * M_PI;
  else if (gps_yaw <= -M_PI)
    gps_yaw += 2 * M_PI;
}

/** \brief get IMU messages in a certain time interval
  * \param[in] startTime: left boundary of time interval
  * \param[in] endTime: right boundary of time interval
  * \param[in] vimuMsg: store IMU messages
  */
bool fetchImuMsgs(double startTime, double endTime, std::vector<sensor_msgs::ImuConstPtr> &vimuMsg){
  std::unique_lock<std::mutex> lock(_mutexIMUQueue);
  double current_time = 0;
  vimuMsg.clear();
  while(true)
  {
    if(_imuMsgQueue.empty())
      break;
    if(_imuMsgQueue.back()->header.stamp.toSec()<endTime ||
       _imuMsgQueue.front()->header.stamp.toSec()>=endTime)
      break;
    sensor_msgs::ImuConstPtr& tmpimumsg = _imuMsgQueue.front();
    double time = tmpimumsg->header.stamp.toSec();
    if(time<=endTime && time>startTime){
      vimuMsg.push_back(tmpimumsg);
      current_time = time;
      _imuMsgQueue.pop();
      if(time == endTime) break;
    } else{
      if(time<=startTime){
        _imuMsgQueue.pop();
      } else{
        double dt_1 = endTime - current_time;
        double dt_2 = time - endTime;
        ROS_ASSERT(dt_1 >= 0);
        ROS_ASSERT(dt_2 >= 0);
        ROS_ASSERT(dt_1 + dt_2 > 0);
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        sensor_msgs::ImuPtr theLastIMU(new sensor_msgs::Imu);
        theLastIMU->linear_acceleration.x = w1 * vimuMsg.back()->linear_acceleration.x + w2 * tmpimumsg->linear_acceleration.x;
        theLastIMU->linear_acceleration.y = w1 * vimuMsg.back()->linear_acceleration.y + w2 * tmpimumsg->linear_acceleration.y;
        theLastIMU->linear_acceleration.z = w1 * vimuMsg.back()->linear_acceleration.z + w2 * tmpimumsg->linear_acceleration.z;
        theLastIMU->angular_velocity.x = w1 * vimuMsg.back()->angular_velocity.x + w2 * tmpimumsg->angular_velocity.x;
        theLastIMU->angular_velocity.y = w1 * vimuMsg.back()->angular_velocity.y + w2 * tmpimumsg->angular_velocity.y;
        theLastIMU->angular_velocity.z = w1 * vimuMsg.back()->angular_velocity.z + w2 * tmpimumsg->angular_velocity.z;
        theLastIMU->header.stamp.fromSec(endTime);
        vimuMsg.emplace_back(theLastIMU);
        break;
      }
    }
  }
  return !vimuMsg.empty();
}

/** \brief Remove Lidar Distortion
  * \param[in] cloud: lidar cloud need to be undistorted
  * \param[in] dRlc: delta rotation
  * \param[in] dtlc: delta displacement
  */
void RemoveLidarDistortion(pcl::PointCloud<PointType>::Ptr& cloud,
                           const Eigen::Matrix3d& dRlc, const Eigen::Vector3d& dtlc){
  int PointsNum = cloud->points.size();
  for (int i = 0; i < PointsNum; i++) {
    Eigen::Vector3d startP;
    float s = cloud->points[i].normal_x;
    Eigen::Quaterniond qlc = Eigen::Quaterniond(dRlc).normalized();
    Eigen::Quaterniond delta_qlc = Eigen::Quaterniond::Identity().slerp(s, qlc).normalized();
    const Eigen::Vector3d delta_Plc = s * dtlc;
    startP = delta_qlc * Eigen::Vector3d(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z) + delta_Plc;
    Eigen::Vector3d _po = dRlc.transpose() * (startP - dtlc);

    cloud->points[i].x = _po(0);
    cloud->points[i].y = _po(1);
    cloud->points[i].z = _po(2);
    cloud->points[i].normal_x = 1.0;
  }
}


bool TryMAPInitialization() {

  Eigen::Vector3d average_acc = -lidarFrameList->begin()->imuIntegrator.GetAverageAcc();
  double info_g = std::fabs(9.805 - average_acc.norm());
  average_acc = average_acc * 9.805 / average_acc.norm();

  // calculate the initial gravity direction
  double para_quat[4];
  para_quat[0] = 1;
  para_quat[1] = 0;
  para_quat[2] = 0;
  para_quat[3] = 0;


  ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
  ceres::Problem problem_quat;
  
  problem_quat.AddParameterBlock(para_quat, 4, quatParam);

  problem_quat.AddResidualBlock(Cost_Initial_G::Create(average_acc),
                                nullptr,
                                para_quat);

  ceres::Solver::Options options_quat;
  ceres::Solver::Summary summary_quat;
  ceres::Solve(options_quat, &problem_quat, &summary_quat);

  Eigen::Quaterniond q_wg(para_quat[0], para_quat[1], para_quat[2], para_quat[3]);


  //build prior factor of LIO initialization
  Eigen::Vector3d prior_r = Eigen::Vector3d::Zero();
  Eigen::Vector3d prior_ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d prior_bg = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> prior_v;
  int v_size = lidarFrameList->size();
  for(int i = 0; i < v_size; i++) {
    prior_v.push_back(Eigen::Vector3d::Zero());
  }
  Sophus::SO3d SO3_R_wg(q_wg.toRotationMatrix());
  prior_r = SO3_R_wg.log();
  
  for (int i = 1; i < v_size; i++){
    auto iter = lidarFrameList->begin();
    auto iter_next = lidarFrameList->begin();
    std::advance(iter, i-1);
    std::advance(iter_next, i);

    Eigen::Vector3d velo_imu = (iter_next->P - iter->P + iter_next->Q*exPlb - iter->Q*exPlb) / (iter_next->timeStamp - iter->timeStamp);
    prior_v[i] = velo_imu;
  }
  prior_v[0] = prior_v[1];

  double para_v[v_size][3];
  double para_r[3];
  double para_ba[3];
  double para_bg[3];

  for(int i = 0; i < 3; i++) {
    para_r[i] = 0;
    para_ba[i] = 0;
    para_bg[i] = 0;
  }

  for(int i = 0; i < v_size; i++) {
    for(int j = 0; j < 3; j++) {
      para_v[i][j] = prior_v[i][j];
    }
  }

  Eigen::Matrix<double, 3, 3> sqrt_information_r = 2000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_ba = 1000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_bg = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> sqrt_information_v = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);
  problem.AddParameterBlock(para_r, 3);
  problem.AddParameterBlock(para_ba, 3);
  problem.AddParameterBlock(para_bg, 3);
  for(int i = 0; i < v_size; i++) {
    problem.AddParameterBlock(para_v[i], 3);
  }
  
  // add CostFunction
  problem.AddResidualBlock(Cost_Initialization_Prior_R::Create(prior_r, sqrt_information_r),
                           nullptr,
                           para_r);
  
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_ba, sqrt_information_ba),
                           nullptr,
                           para_ba);
  problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_bg, sqrt_information_bg),
                           nullptr,
                           para_bg);

  for(int i = 0; i < v_size; i++) {
    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_v[i], sqrt_information_v),
                             nullptr,
                             para_v[i]);
  }

  for(int i = 1; i < v_size; i++) {
    auto iter = lidarFrameList->begin();
    auto iter_next = lidarFrameList->begin();
    std::advance(iter, i-1);
    std::advance(iter_next, i);

    Eigen::Vector3d pi = iter->P + iter->Q*exPlb;
    Sophus::SO3d SO3_Ri(iter->Q*exRlb);
    Eigen::Vector3d ri = SO3_Ri.log();
    Eigen::Vector3d pj = iter_next->P + iter_next->Q*exPlb;
    Sophus::SO3d SO3_Rj(iter_next->Q*exRlb);
    Eigen::Vector3d rj = SO3_Rj.log();

    problem.AddResidualBlock(Cost_Initialization_IMU::Create(iter_next->imuIntegrator,
                                                                   ri,
                                                                   rj,
                                                                   pj-pi,
                                                                   Eigen::LLT<Eigen::Matrix<double, 9, 9>>
                                                                    (iter_next->imuIntegrator.GetCovariance().block<9,9>(0,0).inverse())
                                                                    .matrixL().transpose()),
                             nullptr,
                             para_r,
                             para_v[i-1],
                             para_v[i],
                             para_ba,
                             para_bg);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = 6;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Vector3d r_wg(para_r[0], para_r[1], para_r[2]);
  GravityVector = Sophus::SO3d::exp(r_wg) * Eigen::Vector3d(0, 0, -9.805);

  Eigen::Vector3d ba_vec(para_ba[0], para_ba[1], para_ba[2]);
  Eigen::Vector3d bg_vec(para_bg[0], para_bg[1], para_bg[2]);
  // std::cout<<"ba_vec="<<para_ba[0]<<", "<<para_ba[1]<<", "<<para_ba[2]<<std::endl;
  // std::cout<<"bg_vec="<<para_bg[0]<<", "<<para_bg[1]<<", "<<para_bg[2]<<std::endl;
  // std::cout<<"ba_vec.norm()="<<ba_vec.norm()<<std::endl;
  // std::cout<<"bg_vec.norm()="<<bg_vec.norm()<<std::endl;
  if(ba_vec.norm() > 0.5 || bg_vec.norm() > 0.5) {
    ROS_WARN("Too Large Biases! Initialization Failed!");
    return false;
  }

  for(int i = 0; i < v_size; i++) {
    auto iter = lidarFrameList->begin();
    std::advance(iter, i);
    iter->ba = ba_vec;
    iter->bg = bg_vec;
    Eigen::Vector3d bv_vec(para_v[i][0], para_v[i][1], para_v[i][2]);
    if((bv_vec - prior_v[i]).norm() > 2.0) {
      ROS_WARN("Too Large Velocity! Initialization Failed!");
      std::cout<<"delta v norm: "<<(bv_vec - prior_v[i]).norm()<<std::endl;
      return false;
    }
    iter->V = bv_vec;
  }

  for(size_t i = 0; i < v_size - 1; i++){
    auto laser_trans_i = lidarFrameList->begin();
    auto laser_trans_j = lidarFrameList->begin();
    std::advance(laser_trans_i, i);
    std::advance(laser_trans_j, i+1);
    laser_trans_j->imuIntegrator.PreIntegration(laser_trans_i->timeStamp, laser_trans_i->bg, laser_trans_i->ba);
  }


  // //if IMU success initialized
  WINDOWSIZE = Estimator::SLIDEWINDOWSIZE;
  while(lidarFrameList->size() > WINDOWSIZE){
	  lidarFrameList->pop_front();
  }
	Eigen::Vector3d Pwl = lidarFrameList->back().P;
	Eigen::Quaterniond Qwl = lidarFrameList->back().Q;
	lidarFrameList->back().P = Pwl + Qwl*exPlb;
	lidarFrameList->back().Q = Qwl * exRlb;

	// std::cout << "\n=============================\n| Initialization Successful |"<<"\n=============================\n" << std::endl;
  
  return true;
}


/** \brief Mapping main thread
  */
void process(){
  double time_last_lidar = -1;
  double time_curr_lidar = -1;
  //std::cout<<"  haha"<<std::endl;
  Eigen::Matrix3d delta_Rl = Eigen::Matrix3d::Identity();
  Eigen::Vector3d delta_tl = Eigen::Vector3d::Zero();
	Eigen::Matrix3d delta_Rb = Eigen::Matrix3d::Identity();
	Eigen::Vector3d delta_tb = Eigen::Vector3d::Zero();
  std::string poseFilePath = savePath + "localmap/odom.txt";
  std::string SlamPoseevoFilePath = savePath + "slam_evo.txt";
  std::string GpsPoseevoFilePath = savePath + "gps_evo.txt";
  std::string SlamPoseFilePath = savePath + "slam.txt";
  std::string GpsPoseFilePath = savePath + "gps.txt";
  std::string local_filepath = savePath+"localmap/";

  char dst_dir[255];
  strcpy(dst_dir, local_filepath.c_str());
  int state = access(dst_dir, R_OK|W_OK);
  if(state == 0){
    std::cout<<"file exist, delete all files!"<<std::endl;
    std::string delete_local_str = "rm " + local_filepath + "*";
    strcpy(dst_dir, delete_local_str.c_str());
    std::system(dst_dir);
  }else{
    std::string mkdir_str = "mkdir " + local_filepath;
    strcpy(dst_dir, mkdir_str.c_str());
    std::system(dst_dir);
    std::cout<<"Create file path "<<local_filepath<<std::endl;
  }
  poseFile.open(poseFilePath);

  //记录轨迹
  if(record_data == true){
    SlamPoseevoFile.open(SlamPoseevoFilePath);
    GpsPoseevoFile.open(GpsPoseevoFilePath);
    SlamPoseFile.open(SlamPoseFilePath);
    GpsPoseFile.open(GpsPoseFilePath);
  }

 
  //25
  transformAftMapped.topLeftCorner(3,3) = Eigen::Matrix3d(Eigen::AngleAxisd(0.3563,Eigen::Vector3d::UnitZ())*
                             Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())*
                             Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX()));
  
  transformAftMapped.topRightCorner(3,1) = Eigen::Vector3d(1701.4710,1083.2363,0);

  //32
  // transformAftMapped.topLeftCorner(3,3) = Eigen::Matrix3d(Eigen::AngleAxisd(0.3256,Eigen::Vector3d::UnitZ())*
  //                            Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())*
  //                            Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX()));
  
  // transformAftMapped.topRightCorner(3,1) = Eigen::Vector3d(1739.4742,1455.2052,0);

  //11-14-17
  // transformAftMapped.topLeftCorner(3,3) = Eigen::Matrix3d(Eigen::AngleAxisd(-1.170103673,Eigen::Vector3d::UnitZ())*
  //                            Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())*
  //                            Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX()));
  
  // transformAftMapped.topRightCorner(3,1) = Eigen::Vector3d(1584.7611,1402.1700,0);

  //11-14-16
  // transformAftMapped.topLeftCorner(3,3) = Eigen::Matrix3d(Eigen::AngleAxisd(-1.222463551,Eigen::Vector3d::UnitZ())*
  //                            Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())*
  //                            Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX())); //-2.7409
  
  // transformAftMapped.topRightCorner(3,1) = Eigen::Vector3d(1583.7022,1401.7214,0);

// transformAftMapped.topLeftCorner(3,3) = Eigen::Matrix3d(Eigen::AngleAxisd(-2.8074,Eigen::Vector3d::UnitZ())*
//                              Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())*
//                              Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX()));
  
// transformAftMapped.topRightCorner(3,1) = Eigen::Vector3d(1927.1312,1527.2840,0);


  std::vector<sensor_msgs::ImuConstPtr> vimuMsg;
  while(ros::ok()){
    newfullCloud = false;
    laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
	  std::unique_lock<std::mutex> lock_lidar(_mutexLidarQueue);
    if(!_lidarMsgQueue.empty()){
      // get new lidar msg
      time_curr_lidar = _lidarMsgQueue.front()->header.stamp.toSec();
      pcl::fromROSMsg(*_lidarMsgQueue.front(), *laserCloudFullRes);
      _lidarMsgQueue.pop();
      newfullCloud = true;
    }
    lock_lidar.unlock();

    if(newfullCloud){

      nav_msgs::Odometry debugInfo;
      debugInfo.pose.pose.position.x = 0;
      debugInfo.pose.pose.position.y = 0;
      debugInfo.pose.pose.position.z = 0;
      if(IMU_Mode > 0 && time_last_lidar > 0){
        // get IMU msg int the Specified time interval
        vimuMsg.clear();
        int countFail = 0;
        while (!fetchImuMsgs(time_last_lidar, time_curr_lidar, vimuMsg)) {
          countFail++;
          if (countFail > 100){
            break;
          }
          std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        }
      }
      // this lidar frame init
      Estimator::LidarFrame lidarFrame;
      lidarFrame.laserCloud = laserCloudFullRes;
      lidarFrame.timeStamp = time_curr_lidar;

	    boost::shared_ptr<std::list<Estimator::LidarFrame>> lidar_list;
	    if(!vimuMsg.empty()){
	    	if(!LidarIMUInited) {
	    		// if get IMU msg successfully, use gyro integration to update delta_Rl
			    lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
			    lidarFrame.imuIntegrator.GyroIntegration(time_last_lidar);
			    delta_Rb = lidarFrame.imuIntegrator.GetDeltaQ().toRotationMatrix();
			    delta_Rl = exTlb.topLeftCorner(3, 3) * delta_Rb * exTlb.topLeftCorner(3, 3).transpose();

			    // predict current lidar pose
			    lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
			                   + transformAftMapped.topRightCorner(3,1);
			    Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
			    lidarFrame.Q = m3d;

			    lidar_list.reset(new std::list<Estimator::LidarFrame>);
			    lidar_list->push_back(lidarFrame);
		    }else{
			    // if get IMU msg successfully, use pre-integration to update delta lidar pose
			    lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
			    lidarFrame.imuIntegrator.PreIntegration(lidarFrameList->back().timeStamp, lidarFrameList->back().bg, lidarFrameList->back().ba);

			    const Eigen::Vector3d& Pwbpre = lidarFrameList->back().P;
			    const Eigen::Quaterniond& Qwbpre = lidarFrameList->back().Q;
			    const Eigen::Vector3d& Vwbpre = lidarFrameList->back().V;

			    const Eigen::Quaterniond& dQ =  lidarFrame.imuIntegrator.GetDeltaQ();
			    const Eigen::Vector3d& dP = lidarFrame.imuIntegrator.GetDeltaP();
			    const Eigen::Vector3d& dV = lidarFrame.imuIntegrator.GetDeltaV();
			    double dt = lidarFrame.imuIntegrator.GetDeltaTime();

			    lidarFrame.Q = Qwbpre * dQ;
			    lidarFrame.P = Pwbpre + Vwbpre*dt + 0.5*GravityVector*dt*dt + Qwbpre*(dP);
			    lidarFrame.V = Vwbpre + GravityVector*dt + Qwbpre*(dV);
			    lidarFrame.bg = lidarFrameList->back().bg;
			    lidarFrame.ba = lidarFrameList->back().ba;

			    Eigen::Quaterniond Qwlpre = Qwbpre * Eigen::Quaterniond(exRbl);
			    Eigen::Vector3d Pwlpre = Qwbpre * exPbl + Pwbpre;

			    Eigen::Quaterniond Qwl = lidarFrame.Q * Eigen::Quaterniond(exRbl);
			    Eigen::Vector3d Pwl = lidarFrame.Q * exPbl + lidarFrame.P;

			    delta_Rl = Qwlpre.conjugate() * Qwl;
			    delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
			    delta_Rb = dQ.toRotationMatrix();
			    delta_tb = dP;

			    lidarFrameList->push_back(lidarFrame);
			    lidarFrameList->pop_front();
			    lidar_list = lidarFrameList;
	    	}
	    }else{
	    	if(LidarIMUInited)
	    	  break;
	    	else{
			    // predict current lidar pose
          //通过上一帧增量估计此帧位置
			    lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
			                   + transformAftMapped.topRightCorner(3,1);
			    Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
			    lidarFrame.Q = m3d;

			    lidar_list.reset(new std::list<Estimator::LidarFrame>);
			    lidar_list->push_back(lidarFrame);
          // std::cout<<"transformAftMapped.topLeftCorner(3,3)="<<transformAftMapped.topLeftCorner(3,3)<<std::endl;
          // std::cout<<"transformAftMapped.topRightCorner(3,1)="<<transformAftMapped.topRightCorner(3,1).transpose()<<std::endl;
          //  std::cout<<"delta_tb="<<delta_tb.transpose()<<std::endl;
          // std::cout<<"delta_Rb="<<delta_Rb<<std::endl;
	    	}
	    }

	    // remove lidar distortion
	    RemoveLidarDistortion(laserCloudFullRes, delta_Rl, delta_tl);

      // optimize current lidar pose with IMU
      //通过特征做匹配，求结果，lidar_list为最终结果
      estimator->EstimateLidarPose(*lidar_list, exTlb, GravityVector, debugInfo);

      pcl::PointCloud<PointType>::Ptr laserCloudCornerMap(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr laserCloudSurfMap(new pcl::PointCloud<PointType>());

	    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
	    transformTobeMapped.topLeftCorner(3,3) = lidar_list->front().Q * exRbl; //exRbl IMU和lidar变换关系
	    transformTobeMapped.topRightCorner(3,1) = lidar_list->front().Q * exPbl + lidar_list->front().P;
      
      //z轴置为0
      //transformTobeMapped(2,3)= 0;

      // std::cout<<"  --lidar_list->front().Q="<<lidar_list->front().Q.w()<<" "<<lidar_list->front().Q.vec().transpose()<<std::endl;
      // std::cout<<"  --lidar_list->front().P="<<lidar_list->front().P.transpose()<<std::endl;
      // std::cout<<"  --exRbl="<<exRbl<<std::endl;
      // std::cout<<"  --transformTobeMapped="<<transformTobeMapped<<std::endl;


	    // update delta transformation
      //delta_Rb,delta_tb是求这一帧相对上一帧增量
	    delta_Rb = transformAftMapped.topLeftCorner(3, 3).transpose() * lidar_list->front().Q.toRotationMatrix();
	    delta_tb = transformAftMapped.topLeftCorner(3, 3).transpose() * (lidar_list->front().P - transformAftMapped.topRightCorner(3, 1));
	    Eigen::Matrix3d Rwlpre = transformAftMapped.topLeftCorner(3, 3) * exRbl;
	    Eigen::Vector3d Pwlpre = transformAftMapped.topLeftCorner(3, 3) * exPbl + transformAftMapped.topRightCorner(3, 1);
	    delta_Rl = Rwlpre.transpose() * transformTobeMapped.topLeftCorner(3,3);
	    delta_tl = Rwlpre.transpose() * (transformTobeMapped.topRightCorner(3,1) - Pwlpre);
	    transformAftMapped.topLeftCorner(3,3) = lidar_list->front().Q.toRotationMatrix();
	    transformAftMapped.topRightCorner(3,1) = lidar_list->front().P;

	    // publish odometry rostopic
	    pubOdometry(transformTobeMapped, lidar_list->front().timeStamp); //transformTobeMapped是最后的结果
      
      //发布当前帧给后端优化使用
      sensor_msgs::PointCloud2 scanCloudMsg;
      pcl::toROSMsg(*(lidar_list->front().laserCloud), scanCloudMsg);
      scanCloudMsg.header.frame_id = "/world";
      scanCloudMsg.header.stamp.fromSec(lidar_list->front().timeStamp);
      pubCurScan.publish(scanCloudMsg);

      // publish lidar points
      int laserCloudFullResNum = lidar_list->front().laserCloud->points.size();
      pcl::PointCloud<PointType>::Ptr laserCloudAfterEstimate(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr laserCloudOutlier(new pcl::PointCloud<PointType>());
      laserCloudAfterEstimate->reserve(laserCloudFullResNum);
      for (int i = 0; i < laserCloudFullResNum; i++) {
        PointType temp_point;
        MAP_MANAGER::pointAssociateToMap(&lidar_list->front().laserCloud->points[i], &temp_point, transformTobeMapped); //点云转换到全局坐标
        if((lidar_list->front().laserCloud->points[i].normal_z == 0) || (lidar_list->front().laserCloud->points[i].normal_z > 9))
          laserCloudOutlier->push_back(temp_point);
        else
          laserCloudAfterEstimate->push_back(temp_point);


      }
      //std::cout<<"  laserCloudAfterEstimate="<<laserCloudAfterEstimate->points.size()<<std::endl;

      sensor_msgs::PointCloud2 laserCloudMsg;
      pcl::toROSMsg(*laserCloudAfterEstimate, laserCloudMsg);
      laserCloudMsg.header.frame_id = "/world";
      laserCloudMsg.header.stamp.fromSec(lidar_list->front().timeStamp);
      pubFullLaserCloud.publish(laserCloudMsg); //最后输出的地图，原代码地图数量有限

      sensor_msgs::PointCloud2 laserCloudMsg1;
      pcl::toROSMsg(*laserCloudOutlier, laserCloudMsg1);
      laserCloudMsg1.header.frame_id = "/world";
      laserCloudMsg1.header.stamp.fromSec(lidar_list->front().timeStamp);
      pubOutlierLaserCloud.publish(laserCloudMsg1); //最后输出的地图，原代码地图数量有限

      //此处为全局地图显示
      _mutexMapQueue.lock();
      _globalMapQueue.push(*laserCloudAfterEstimate);
      _scanPoseQueue.push(transformTobeMapped);
      _scanQuatQueue.push(lidar_list->front().Q);
      _scanVectQueue.push(lidar_list->front().P);
      _scanTimeQueue.push(lidar_list->front().timeStamp);
      _mutexMapQueue.unlock();     

	    // if tightly coupled IMU message, start IMU initialization
	    if(IMU_Mode > 1 && !LidarIMUInited){
		    // update lidar frame pose
		    lidarFrame.P = transformTobeMapped.topRightCorner(3,1);
		    Eigen::Matrix3d m3d = transformTobeMapped.topLeftCorner(3,3);
		    lidarFrame.Q = m3d;

		    // static int pushCount = 0;
		    if(pushCount == 0){
			    lidarFrameList->push_back(lidarFrame);
			    lidarFrameList->back().imuIntegrator.Reset();
			    if(lidarFrameList->size() > WINDOWSIZE)
				    lidarFrameList->pop_front();
		    }else{
			    lidarFrameList->back().laserCloud = lidarFrame.laserCloud;
			    lidarFrameList->back().imuIntegrator.PushIMUMsg(vimuMsg);
			    lidarFrameList->back().timeStamp = lidarFrame.timeStamp;
			    lidarFrameList->back().P = lidarFrame.P;
			    lidarFrameList->back().Q = lidarFrame.Q;
		    }
		    pushCount++;
		    if (pushCount >= 3){
			    pushCount = 0;
			    if(lidarFrameList->size() > 1){
				    auto iterRight = std::prev(lidarFrameList->end());
				    auto iterLeft = std::prev(std::prev(lidarFrameList->end()));
				    iterRight->imuIntegrator.PreIntegration(iterLeft->timeStamp, iterLeft->bg, iterLeft->ba);
			    }

          if (lidarFrameList->size() == int(WINDOWSIZE / 1.5)) {
					  startTime = lidarFrameList->back().timeStamp;
			    }

			    if (!LidarIMUInited && lidarFrameList->size() == WINDOWSIZE && lidarFrameList->front().timeStamp >= startTime){
            std::cout<<"**************Start MAP Initialization!!!******************"<<std::endl;
				    if(TryMAPInitialization()){
              LidarIMUInited = true;
					    pushCount = 0;
              startTime = 0;
				    }
            std::cout<<"**************Finish MAP Initialization!!!******************"<<std::endl;
			    }

		    }
	    }
      time_last_lidar = time_curr_lidar;

    }
  }
}

//全局地图显示 记录全局和局部地图（帧叠加）
// double temp_time;
// void publishGlobalMap(){
//     _mutexMapQueue.lock();
//     while(!_globalMapQueue.empty()){
//       pcl::PointCloud<PointType> temp_pointcloud;
//       Eigen::Matrix4d temp_transform = _scanPoseQueue.front();
//       Eigen::Quaterniond temp_quaternion = _scanQuatQueue.front();
//       Eigen::Vector3d temp_position = _scanVectQueue.front();
//       temp_pointcloud = _globalMapQueue.front();
//       temp_time = _scanTimeQueue.front();
//       _globalMapQueue.pop();
//       _scanPoseQueue.pop();
//       _scanQuatQueue.pop();
//       _scanVectQueue.pop();
//       _scanTimeQueue.pop();

//       *globalMap += temp_pointcloud;
//       *localMap += temp_pointcloud;
//       if(mapNUm == 0)
//         transformLocToMap = temp_transform;
//         quatLocToMap = temp_quaternion;
//         poseLocToMap = temp_position;
//       mapNUm++;
//     } 
//     _mutexMapQueue.unlock();

//     //局部地图添加保存
//     if(mapNUm>10){
//       downlocalMap.setLeafSize(0.1, 0.1, 0.1);
//       downlocalMap.setInputCloud(localMap);
//       downlocalMap.filter(*localMapDS);
//       Eigen::Matrix4d transformLocToMapInv = transformLocToMap.inverse();
//       for (int i = 0; i < localMapDS->points.size(); i++) {
//         PointType temp_point;
//         MAP_MANAGER::pointAssociateToMap(&localMapDS->points[i], &temp_point, transformLocToMapInv); 
//         localMapDSInv->push_back(temp_point);
//       }

//       std::string file_name = savePath + "localmap/localmap_"+ std::to_string(saveMapNum)+".ply";
//       // pcl::io::savePLYFileASCII(file_name ,*localMapDSInv);
//       pcl::io::savePLYFileASCII(file_name ,*localMapDS);
//       // poseFilePath<<std::fixed<<std::setprecision(4)<<transformLocToMap<<std::endl;
//       poseFile<<transformLocToMap<<std::endl;

//       if(record_data == true){
//         tf::Quaternion q(quatLocToMap.x(),quatLocToMap.y(),quatLocToMap.z(),quatLocToMap.w());
//         double roll,pitch,yaw;
//         tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
//         if(record_data == true){
//           SlamPoseevoFile << std::fixed << std::setprecision(4)                
//                   << temp_time << ' '
//                   << poseLocToMap[0] << ' '
//                   << poseLocToMap[1] << ' '
//                   << 0 << ' '
//                   // << quatLocToMap.x() << ' '
//                   // << quatLocToMap.y() << ' '
//                   // << quatLocToMap.z() << ' '
//                   // << quatLocToMap.w() << std::endl;
//                     << 0<< ' '
//                   << 0<< ' '
//                   << 0 << ' '
//                   << 1 << std::endl;
//           SlamPoseFile<< std::fixed << std::setprecision(4) 
//                   <<temp_time << ' '
//                   << poseLocToMap[0] << ' '
//                   << poseLocToMap[1] << ' '
//                   << 0 << ' '
//                   << roll << ' '
//                   << pitch << ' '
//                   << yaw<< ' '
//                   << std::endl;
//         }

//       }
//       localMap.reset(new pcl::PointCloud<PointType>());
//       localMapDS.reset(new pcl::PointCloud<PointType>());
//       localMapDSInv.reset(new pcl::PointCloud<PointType>());
//       mapNUm = 0;
//       saveMapNum++;
//     }

//     downFullMap.setLeafSize(0.5, 0.5, 0.5);
//     downFullMap.setInputCloud(globalMap);
//     downFullMap.filter(*globalMapDS);
//     *globalMap = *globalMapDS;
//     sensor_msgs::PointCloud2 tempCloud;
//     pcl::toROSMsg(*globalMap,tempCloud);
//     tempCloud.header.stamp = ros::Time::now();
//     tempCloud.header.frame_id = "/world";
//     pubGlobalMap.publish(tempCloud);
// }

//全局地图显示 记录全局和局部地图（局部切分）
double temp_time;
void publishGlobalMap(){
    clock_t startTime = clock();
    _mutexMapQueue.lock();
    while(!_globalMapQueue.empty()){
      pcl::PointCloud<PointType> temp_pointcloud;
      Eigen::Matrix4d temp_transform = _scanPoseQueue.front();
      Eigen::Quaterniond temp_quaternion = _scanQuatQueue.front();
      Eigen::Vector3d temp_position = _scanVectQueue.front();
      temp_pointcloud = _globalMapQueue.front();
      temp_time = _scanTimeQueue.front();
      _globalMapQueue.pop();
      _scanPoseQueue.pop();
      _scanQuatQueue.pop();
      _scanVectQueue.pop();
      _scanTimeQueue.pop();

      *readinMap += temp_pointcloud;
      transformLocToMap = temp_transform;
      quatLocToMap = temp_quaternion;
      poseLocToMap = temp_position;
    } 
    _mutexMapQueue.unlock();
    *globalMap += *readinMap;
    *localMap += *readinMap;
    readinMap.reset(new pcl::PointCloud<PointType>());

    //局部地图添加保存
    if(sqrt(pow((poseLocToMap[0] - last_keypose[0]),2) + pow((poseLocToMap[1] - last_keypose[1]),2))>3){
      clock_t t1 = clock();
      //std::cout<<"dis="<<sqrt(pow((poseLocToMap[0] - last_keypose[0]),2) + pow((poseLocToMap[1] - last_keypose[1]),2))<<" m"<<std::endl;
      // *localMap = *globalMap;
      // disFilter.setInputCloud(localMap);

      // disFilter.setFilterFieldName("x");
      // disFilter.setFilterLimits(poseLocToMap[0]-80, poseLocToMap[0]+80);
      // disFilter.filter(*localMap);

      // disFilter.setFilterFieldName("y");
      // disFilter.setFilterLimits(poseLocToMap[1]-80, poseLocToMap[1]+80);
      // disFilter.filter(*localMap);

      // disFilter.setFilterFieldName("z");
      // disFilter.setFilterLimits(-80, 80);
      // disFilter.filter(*localMap);

      downFullMap.setLeafSize(1, 1, 1);
      downFullMap.setInputCloud(globalMap);
      downFullMap.filter(*globalMapDS);
      *globalMap = *globalMapDS;

      downLocalMap.setLeafSize(0.8, 0.8, 0.4);
      downLocalMap.setInputCloud(localMap);
      downLocalMap.filter(*localMapDS);
      *localMap = *localMapDS;
      localMapDS.reset(new pcl::PointCloud<PointType>());

      clock_t t2 = clock();
      for(int i = 0; i < localMap->size(); i++){
        double dis = sqrt(pow((localMap->points[i].x - poseLocToMap[0]),2) + pow((localMap->points[i].y - poseLocToMap[1]),2));
        if(dis < 80){
          localMapDS->points.push_back(localMap->points[i]);
        }
      }
      *localMap = *localMapDS;
      pcl::PointCloud<pcl::PointXYZI>::Ptr local_output(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::copyPointCloud(*localMap,*local_output);

      clock_t t3 = clock();
      //std::cout<<"disfilter="<<(double)(t3 - t2)/CLOCKS_PER_SEC * 1000<<" ms"<<std::endl;

      std::string file_name = savePath + "localmap/localmap_"+ std::to_string(saveMapNum)+".ply";
      // pcl::io::savePLYFileASCII(file_name ,*localMapDSInv);
      if(record_data == true){
        pcl::io::savePLYFileASCII(file_name ,*local_output);
      }
      // poseFilePath<<std::fixed<<std::setprecision(4)<<transformLocToMap<<std::endl;
      poseFile<<transformLocToMap<<std::endl;

      if(record_data == true){
        tf::Quaternion q(quatLocToMap.x(),quatLocToMap.y(),quatLocToMap.z(),quatLocToMap.w());
        double roll,pitch,yaw;
        tf::Matrix3x3(q).getRPY(roll,pitch,yaw);
        SlamPoseevoFile << std::fixed << std::setprecision(4)                
                  << temp_time << ' '
                  << poseLocToMap[0] << ' '
                  << poseLocToMap[1] << ' '
                  << 0 << ' '
                  // << quatLocToMap.x() << ' '
                  // << quatLocToMap.y() << ' '
                  // << quatLocToMap.z() << ' '
                  // << quatLocToMap.w() << std::endl;
                    << 0<< ' '
                  << 0<< ' '
                  << 0 << ' '
                  << 1 << std::endl;
        SlamPoseFile<< std::fixed << std::setprecision(4) 
                  <<temp_time << ' '
                  << poseLocToMap[0] << ' '
                  << poseLocToMap[1] << ' '
                  << 0 << ' '
                  << roll << ' '
                  << pitch << ' '
                  << yaw<< ' '
                  << std::endl;
      }
      last_keypose[0] = poseLocToMap[0];
      last_keypose[1] = poseLocToMap[1];
      localMapDS.reset(new pcl::PointCloud<PointType>());
      localMapDSInv.reset(new pcl::PointCloud<PointType>());
      mapNUm = 0;
      saveMapNum++;
      clock_t t4 = clock();
      //std::cout<<"Procsee time="<<(double)(t4 - t1)/CLOCKS_PER_SEC * 1000<<" ms"<<std::endl;
    }

    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*globalMap,tempCloud);
    tempCloud.header.stamp = ros::Time::now();
    tempCloud.header.frame_id = "/world";
    pubGlobalMap.publish(tempCloud);
    clock_t t2 = clock();
    //std::cout<<"All cost time="<<(double)(t2 - startTime)/CLOCKS_PER_SEC * 1000<<" ms"<<std::endl;
}

//显示线程
void visualizationThread(){
  ros::Rate rate(4);
  while(ros::ok()){
    rate.sleep();
    publishGlobalMap();
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "PoseEstimation");
  ros::NodeHandle nodeHandler("~");

  ros::param::get("~filter_parameter_corner",filter_parameter_corner);
  ros::param::get("~filter_parameter_surf",filter_parameter_surf);
  ros::param::get("~IMU_Mode",IMU_Mode);
	std::vector<double> vecTlb;
	ros::param::get("~Extrinsic_Tlb",vecTlb);

  // set extrinsic matrix between lidar & IMU
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
	R << vecTlb[0], vecTlb[1], vecTlb[2],
	     vecTlb[4], vecTlb[5], vecTlb[6],
	     vecTlb[8], vecTlb[9], vecTlb[10];
	t << vecTlb[3], vecTlb[7], vecTlb[11];
  Eigen::Quaterniond qr(R);
  R = qr.normalized().toRotationMatrix();
  exTlb.topLeftCorner(3,3) = R;
  exTlb.topRightCorner(3,1) = t;
  exRlb = R;
  exRbl = R.transpose();
  exPlb = t;
  exPbl = -1.0 * exRbl * exPlb;

  ros::Subscriber subFullCloud = nodeHandler.subscribe<sensor_msgs::PointCloud2>("/livox_full_cloud", 10, fullCallBack);
  // if(record_data == true){
  //   std::cout<<"hah"<<std::endl;
    ros::Subscriber subGpsFix = nodeHandler.subscribe<sensor_msgs::NavSatFix>("/strong/fix", 10, gpsFixCallBack);
    ros::Subscriber subGpsHeading = nodeHandler.subscribe<cyber_msgs::Heading>("/strong/heading", 10, gpdHeadCallBack);
  // }
  ros::Subscriber sub_imu;
  if(IMU_Mode > 0)
    sub_imu = nodeHandler.subscribe("/livox/imu_3WEDH5900101251", 2000, imu_callback, ros::TransportHints().unreliable());
  if(IMU_Mode < 2)
    WINDOWSIZE = 1;
  else
    WINDOWSIZE = 20;

  pubFullLaserCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_full_cloud_mapped", 10);
  pubOutlierLaserCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_outlier_cloud_mapped", 10);
  pubLaserOdometry = nodeHandler.advertise<nav_msgs::Odometry> ("/livox_odometry_mapped", 5);
  pubLaserOdometryPath = nodeHandler.advertise<nav_msgs::Path> ("/livox_odometry_path_mapped", 5);
	pubGps = nodeHandler.advertise<sensor_msgs::NavSatFix>("/lidar", 1000);
  pubGlobalMap = nodeHandler.advertise<sensor_msgs::PointCloud2>("/global_map", 1000);
  pubCurScan = nodeHandler.advertise<sensor_msgs::PointCloud2>("/cur_scan", 1000);

  tfBroadcaster = new tf::TransformBroadcaster();

  laserCloudFullRes.reset(new pcl::PointCloud<PointType>);
  estimator = new Estimator(filter_parameter_corner, filter_parameter_surf);
	lidarFrameList.reset(new std::list<Estimator::LidarFrame>);

  std::thread thread_process{process};
  std::thread thread_visual{visualizationThread};
  ros::spin();

  if(record_data == true){
    std::string globalFilepath = savePath + "global.ply";
    pcl::io::savePLYFileASCII(globalFilepath, *globalMapDS);
    std::cout<<"save map!"<<std::endl;
  }
  return 0;
}

