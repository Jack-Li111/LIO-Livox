#include <unistd.h>
#include <iostream>
#include <cstring>
#include <ros/ros.h>
#include "Estimator/Estimator.h"
#include <mutex>
#include <queue>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/TimeReference.h>
#include <nav_msgs/Odometry.h>
#include <nmea_msgs/Sentence.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <geographic_msgs/GeoPointStamped.h>
#include <geodesy/utm.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <pcl/filters/voxel_grid.h>
#include <cmath>
#include <algorithm>
#include <tf/transform_broadcaster.h>
#include <pcl/io/ply_io.h>

std::mutex mutexLidar;
std::queue<sensor_msgs::PointCloud2ConstPtr> lidarQueue;
std::mutex mutexOdometry;
std::queue<nav_msgs::OdometryConstPtr> odomQueue;
std::mutex mutexGps;
std::queue<sensor_msgs::NavSatFixConstPtr> gpsQueue;
std::mutex mutexGpsTime;
std::queue<sensor_msgs::TimeReferenceConstPtr> gpsTimeQueue;

gtsam::ISAM2 *isam;
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCoveriance;
bool aLoopIsClosed = false;
ros::Time timeLaserInfoStamp;
ros::Publisher pubKeyPoses;
ros::Publisher pubRecentKeyFrames;
ros::Publisher pubRecentKeyFrame;
ros::Publisher pubLaserCloudSurround;
ros::Publisher pubLaserOdometry;
std::string odometryFrame;
std::string worldFrame;
std::mutex mtx;

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudKeyFrames;
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudKeyPoses3D(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());

tf::TransformBroadcaster *tf_broadcaster;
tf::StampedTransform *registration_transform;

pcl::PointCloud<pcl::PointXYZI>::Ptr localMap(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr localMapDS(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr localMapDSInv(new pcl::PointCloud<pcl::PointXYZI>());
std::string savePath = "/home/lzg/lab/Learn/lio-livox_ws/src/globalmap_data_gps/";
int saveMapNum = 0;
int mapNUm = 0;
std::ofstream poseFile;

float pointDistance(pcl::PointXYZI p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}
float pointDistance(pcl::PointXYZI p1, pcl::PointXYZI p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

//降采样存起来处理
void fullCallBack(const sensor_msgs::PointCloud2ConstPtr &msg){
  
  std::unique_lock<std::mutex> lock(mutexLidar);
  lidarQueue.push(msg);
  
  
}

void poseCallBack(const nav_msgs::OdometryConstPtr &msg){
  
  std::unique_lock<std::mutex> lock(mutexOdometry);
  odomQueue.push(msg);
  
  
}

void gpsCallBack(const sensor_msgs::NavSatFixConstPtr &msg){
  
  std::unique_lock<std::mutex> lock(mutexGps);
  gpsQueue.push(msg);
  
  
}

void gpsTimeCallBack(const sensor_msgs::TimeReferenceConstPtr &msg){
  
  std::unique_lock<std::mutex> lock(mutexGpsTime);
  gpsTimeQueue.push(msg);
  
  
}

gtsam::Pose3 odomToPose3(nav_msgs::Odometry odomIn){
  tf::Quaternion quat;
  tf::quaternionMsgToTF(odomIn.pose.pose.orientation, quat);
  double roll,pitch,yaw;
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
                       gtsam::Point3(odomIn.pose.pose.position.x, odomIn.pose.pose.position.y, odomIn.pose.pose.position.z));
}

gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

bool odomInit = false;
int count_num = 0;
nav_msgs::Odometry lastOdom;
void addOdomFactor(nav_msgs::Odometry odomIn){

  if(!odomInit){
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1,1,M_PI*M_PI,1e8,1e8,1e8).finished());
    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, odomToPose3(odomIn), priorNoise));
    initialEstimate.insert(0, odomToPose3(odomIn));
    lastOdom = odomIn;
    odomInit = true;
  }else{
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::Pose3 poseFrom = odomToPose3(lastOdom);
    gtsam::Pose3 poseTo = odomToPose3(odomIn);
    lastOdom = odomIn;
    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(count_num-1, count_num, poseFrom.between(poseTo), odometryNoise));
    initialEstimate.insert(count_num, poseTo); //??
  }
  count_num ++;
  // std::cout<<"count_num="<<count_num<<std::endl;
}

pcl::PointXYZI llaToUtm(double latitude, double longitude, double altitude){
  pcl::PointXYZI result;
  geographic_msgs::GeoPointStampedPtr gps_msg(new geographic_msgs::GeoPointStamped());
  gps_msg->position.latitude = latitude;
  gps_msg->position.longitude = longitude;
  gps_msg->position.altitude = altitude;

  geodesy::UTMPoint utm_point;
  geodesy::fromMsg(gps_msg->position, utm_point);
  result.x = utm_point.easting - 349350;
  result.y = utm_point.northing - 3432459;
  result.z = utm_point.altitude;

  return result;
}

double lastGpsTime = -1, curGpsTime = -1;
sensor_msgs::NavSatFix lastGps, curGps;
bool findLastGpsPose,findCurGpsPose = false;

// void addOdomGpsFactor(double slamTime){
//   std::cout<<"haha"<<std::endl;
//   std::unique_lock<std::mutex> lock_gps(mutexGps);
//   if(gpsQueue.empty())
//     return;
//   std::cout<<std::fixed<<std::setprecision(3)<<"gpsTimeQueue.front()->time_ref.toSec()="<<gpsQueue.front()->header.stamp.toSec()<<std::endl;
//   std::cout<<std::fixed<<std::setprecision(3)<<"slamTime="<<slamTime<<std::endl;
//   while(!gpsQueue.empty()){
//     if(gpsQueue.front()->header.stamp.toSec()<slamTime){
//       lastGpsTime = gpsQueue.front()->header.stamp.toSec();
//       lastGps = *gpsQueue.front();
//       gpsQueue.pop();
//     }else{
//       curGpsTime = gpsQueue.front()->header.stamp.toSec();
//       curGps = *gpsQueue.front();
//       findCurGpsPose = true;
//       break;
//     }
//   }
//   lock_gps.unlock();
//   
//   if(findCurGpsPose == false || lastGpsTime == -1){
//     return;
//   }
//   
//   findCurGpsPose = false;
//   double noise_x, noise_y, noise_z;
//   noise_x = (lastGps.position_covariance[0] + curGps.position_covariance[0])/2;
//   noise_y = (lastGps.position_covariance[4] + curGps.position_covariance[4])/2;
//   noise_z = (lastGps.position_covariance[8] + curGps.position_covariance[8])/2;
//   std::cout<<"  noise:"<<noise_x<<" "<<noise_y<<" "<<noise_z<<std::endl;
//   if (noise_x > 2.0 || noise_y > 2.0)
//     return;
//   
//   std::vector<double> gpsFusionPose(3,0), gpsLastPose, gpsCurPose;
//   gpsLastPose = llaToUtm(lastGps.latitude, lastGps.longitude, lastGps.altitude); 
//   gpsCurPose = llaToUtm(curGps.latitude, curGps.longitude, curGps.altitude); 
//   std::cout<<"  lastGps:"<<lastGps.latitude<<" "<<lastGps.longitude<<" "<<lastGps.altitude<<std::endl;
//   std::cout<<"  curGps:"<<curGps.latitude<<" "<<curGps.longitude<<" "<<curGps.altitude<<std::endl;
//   std::cout<<"  gpsLastPose.size()="<<gpsLastPose.size()<<std::endl;
//   for(size_t i = 0; i < gpsLastPose.size(); i++){
//       gpsFusionPose[i] = gpsLastPose[i] * (curGpsTime - slamTime)/(curGpsTime - lastGpsTime) + 
//                          gpsCurPose[i] * (slamTime - lastGpsTime)/(curGpsTime - lastGpsTime);
//   }
//   std::cout<<"  gpspose:"<<gpsFusionPose[0]<<" "<<gpsFusionPose[1]<<" "<<gpsFusionPose[2]<<std::endl;
//   std::cout<<"  gpsLastPose:"<<gpsLastPose[0]<<" "<<gpsLastPose[1]<<" "<<gpsLastPose[2]<<std::endl;
//   std::cout<<"  gpsCurPose:"<<gpsCurPose[0]<<" "<<gpsCurPose[1]<<" "<<gpsCurPose[2]<<std::endl;
//   std::cout<<"  GpsTime:"<<curGpsTime<<" "<<lastGpsTime<<std::endl;
//   std::cout<<"  slamTime:"<<slamTime<<std::endl;
//   gtsam::Vector Vector3(3);
//   Vector3 << std::max(noise_x, 1.0),std::max(noise_y, 1.0), std::max(noise_z, 1.0);
//   gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
//    std::cout<<"count_num="<<count_num<<std::endl;
//   gtsam::GPSFactor gps_factor(count_num-1, gtsam::Point3(gpsFusionPose[0], gpsFusionPose[1], gpsFusionPose[2]), gps_noise);
//   gtSAMgraph.add(gps_factor);
//   std::cout<<"  添加GPS因子"<<std::endl;
//   aLoopIsClosed = true;
// }

void addOdomGpsFactor(double slamTime){
  if(gpsQueue.empty()){
    return;
  }
  if(cloudKeyPoses3D->points.empty())
    return;
  else{
    if(pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 20.0)
      return;
  }
  std::cout<<"  poseCoveriance:"<<poseCoveriance(3,3)<<" "<<poseCoveriance(4,4)<<std::endl;
  if(poseCoveriance(3,3) < 25.0 && poseCoveriance(4,4) < 25.0)
    return;
  
  static pcl::PointXYZI gpslastPose;
  while(!gpsQueue.empty()){
    std::cout<<"  time:"<<gpsQueue.front()->header.stamp.toSec() - slamTime<<std::endl;
    if(gpsQueue.front()->header.stamp.toSec() - slamTime < -0.05){
      gpsQueue.pop();
    }
    else if(gpsQueue.front()->header.stamp.toSec() - slamTime > 0.05){
      break;
    }else{
      curGps = *gpsQueue.front();
      gpsQueue.pop();
      double noise_x, noise_y, noise_z;
      noise_x = curGps.position_covariance[0];
      noise_y = curGps.position_covariance[4];
      noise_z = curGps.position_covariance[8];
      std::cout<<"  noise:"<<noise_x<<" "<<noise_y<<" "<<noise_z<<std::endl;
      if (noise_x > 2.0 || noise_y > 2.0)
        return;
      
      pcl::PointXYZI gpsCurPose; 
      gpsCurPose = llaToUtm(curGps.latitude, curGps.longitude, curGps.altitude); 
      std::cout<<"  curGps:"<<curGps.latitude<<" "<<curGps.longitude<<" "<<curGps.altitude<<std::endl;

      if(pointDistance(gpsCurPose, gpslastPose) < 20.0)
        continue;
      else
        gpslastPose = gpsCurPose;

      gtsam::Vector Vector3(3);
      // Vector3 << std::max(noise_x, 1.0),std::max(noise_y, 1.0), std::max(noise_z, 1.0);
      Vector3 << 5,5, 5;
      gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
      std::cout<<"count_num="<<count_num<<std::endl;
      gtsam::GPSFactor gps_factor(count_num-1, gtsam::Point3(gpsCurPose.x, gpsCurPose.y, 1.0), gps_noise);
      gtSAMgraph.add(gps_factor);
      std::cout<<"  添加GPS因子"<<std::endl;
      aLoopIsClosed = true;
      break;
    }


  }


}

void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // update key poses
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            //updatePath(cloudKeyPoses6D->points[i]);
        }

        aLoopIsClosed = false;
    }
}

sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<pcl::PointXYZI>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr transformPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn, PointTypePose* transformIn)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::PointXYZI *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
}

void publishFrames()
{
    if (cloudKeyPoses3D->points.empty())
        return;
    if (pubRecentKeyFrame.getNumSubscribers() != 0)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZI>());
        PointTypePose thisPose6D = cloudKeyPoses6D->points[cloudKeyPoses6D->points.size()-1];
        //对发布点云作处理 Author: lzg 去除近处噪点
        pcl::PointCloud<pcl::PointXYZI>::Ptr temp_CloudKeyFrames(new pcl::PointCloud<pcl::PointXYZI>());
        // pcl::PointCloud<pcl::PointXYZI>::Ptr fil_temp_CloudKeyFrames(new pcl::PointCloud<pcl::PointXYZI>());
        // temp_cornerCloudKeyFrames = segPointCloud(laserCloudCornerLastDS);
        // temp_surfCloudKeyFrames = segPointCloud(laserCloudSurfLastDS);

        *cloudOut += *transformPointCloud(cloudKeyFrames[cloudKeyFrames.size()-1],  &thisPose6D);
        publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, worldFrame);

        nav_msgs::Odometry laserOdometry;
        geometry_msgs::Quaternion newQuat = tf::createQuaternionMsgFromRollPitchYaw(thisPose6D.roll, thisPose6D.pitch, thisPose6D.yaw);
        laserOdometry.header.frame_id = "/world";
        laserOdometry.child_frame_id = "/opt_frame";
        laserOdometry.header.stamp = ros::Time::now();
        laserOdometry.pose.pose.orientation.x = newQuat.x;
        laserOdometry.pose.pose.orientation.y = newQuat.y;
        laserOdometry.pose.pose.orientation.z = newQuat.z;
        laserOdometry.pose.pose.orientation.w = newQuat.w;
        laserOdometry.pose.pose.position.x = thisPose6D.x;
        laserOdometry.pose.pose.position.y = thisPose6D.y;
        laserOdometry.pose.pose.position.z = thisPose6D.z;
        pubLaserOdometry.publish(laserOdometry);

        registration_transform->stamp_ = ros::Time::now();
        registration_transform->setRotation(tf::Quaternion(newQuat.x, newQuat.y, newQuat.z, newQuat.w));
        registration_transform->setOrigin(tf::Vector3(thisPose6D.x, thisPose6D.y, thisPose6D.z));
        tf_broadcaster->sendTransform(*registration_transform);
    }
}

void process(){
  while(ros::ok()){
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn(new pcl::PointCloud<pcl::PointXYZI>());
    nav_msgs::Odometry slamOdomtry;
    if((!lidarQueue.empty()) && (!odomQueue.empty())){
      std::unique_lock<std::mutex> lock_lidar(mutexLidar);
      double time_cur_lidar = lidarQueue.front()->header.stamp.toSec() + 1639464233.424; //!!gps和lidar时间没同步
      timeLaserInfoStamp = lidarQueue.front()->header.stamp;
      pcl::fromROSMsg(*lidarQueue.front(), *cloudIn);
      lidarQueue.pop();
      lock_lidar.unlock();
      std::unique_lock<std::mutex> lock_odom(mutexOdometry);
      slamOdomtry = *odomQueue.front();
      odomQueue.pop();
      lock_odom.unlock();;
      std::lock_guard<std::mutex> lock(mtx);
      
      addOdomFactor(slamOdomtry);
      addOdomGpsFactor(time_cur_lidar);
      
      isam->update(gtSAMgraph, initialEstimate);
      isam->update();
      
      if (aLoopIsClosed == true)
      {
          isam->update();
          isam->update();
          isam->update();
          isam->update();
          isam->update();
      }
      
      gtSAMgraph.resize(0);
      initialEstimate.clear();
      
      //save key poses
      pcl::PointXYZI thisPose3D;
      PointTypePose thisPose6D;
      gtsam::Pose3 latestEstimate;
      
      isamCurrentEstimate = isam->calculateEstimate();
      latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);
      // cout << "****************************************************" << endl;
      // isamCurrentEstimate.print("Current estimate: ");

      thisPose3D.x = latestEstimate.translation().x();
      thisPose3D.y = latestEstimate.translation().y();
      thisPose3D.z = latestEstimate.translation().z();
      thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
      cloudKeyPoses3D->push_back(thisPose3D);

      thisPose6D.x = thisPose3D.x;
      thisPose6D.y = thisPose3D.y;
      thisPose6D.z = thisPose3D.z;
      thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
      thisPose6D.roll  = latestEstimate.rotation().roll();
      thisPose6D.pitch = latestEstimate.rotation().pitch();
      thisPose6D.yaw   = latestEstimate.rotation().yaw();
      thisPose6D.time = time_cur_lidar;
      cloudKeyPoses6D->push_back(thisPose6D);

      poseCoveriance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
      

      // save updated transform
      // transformTobeMapped[0] = latestEstimate.rotation().roll();
      // transformTobeMapped[1] = latestEstimate.rotation().pitch();
      // transformTobeMapped[2] = latestEstimate.rotation().yaw();
      // transformTobeMapped[3] = latestEstimate.translation().x();
      // transformTobeMapped[4] = latestEstimate.translation().y();
      // transformTobeMapped[5] = latestEstimate.translation().z();

      // save all the points
      pcl::PointCloud<pcl::PointXYZI>::Ptr thisKeyFrame(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::VoxelGrid<pcl::PointXYZI> downSizePointCloud;
      downSizePointCloud.setLeafSize(0.2,0.2,0.2);
      downSizePointCloud.setInputCloud(cloudIn);
      downSizePointCloud.filter(*thisKeyFrame);
      cloudKeyFrames.push_back(thisKeyFrame);
      
      // save path for visualization
      //updatePath(thisPose6D);
      correctPoses();
      
      publishFrames();
      
    }
  }

}
pcl::PointCloud<pcl::PointXYZI>::Ptr globalMapKeyFramesSave(new pcl::PointCloud<pcl::PointXYZI>());
void publishGlobalMap()
{
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr globalMapKeyPoses(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr globalMapKeyFrames(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<pcl::PointXYZI>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), 1000.0, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(3.0 , 3.0 , 3.0); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > 1000.0)
            continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *transformPointCloud(cloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    std::cout<<"  globalMapKeyFramesDS.size="<<globalMapKeyFramesDS->points.size()<<std::endl;
    *globalMapKeyFramesSave = *globalMapKeyFramesDS;
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, worldFrame);
}

void visualization(){
  ros::Rate rate(0.5);
  while (ros::ok()){
    rate.sleep();
    publishGlobalMap();
  }
  std::string glonalmap_name = savePath + "globalmap.ply";
  pcl::io::savePLYFile(glonalmap_name,*globalMapKeyFramesSave);
  std::cout<<"Save all map"<<std::endl;

  //帧叠加
  // for (int i = 0; i < (int)cloudKeyPoses3D->size(); ++i){
  //   int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
  //   *localMap += *transformPointCloud(cloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
  //   mapNUm++;
  //   if(mapNUm>10){
  //     pcl::VoxelGrid<pcl::PointXYZI> downLocalMap;
  //     downLocalMap.setLeafSize(0.1, 0.1, 0.1);
  //     downLocalMap.setInputCloud(localMap);
  //     downLocalMap.filter(*localMapDS);
  //     std::string file_name = savePath + "localmap/localmap_"+ std::to_string(saveMapNum)+".ply";
  //     pcl::io::savePLYFileASCII(file_name ,*localMapDS);
  //     Eigen::Matrix4d transformLocToMap  = Eigen::Matrix4d::Identity();
  //     transformLocToMap.topLeftCorner(3,3) = Eigen::Matrix3d(Eigen::AngleAxisd(cloudKeyPoses6D->points[thisKeyInd].yaw,Eigen::Vector3d::UnitZ())*
  //                            Eigen::AngleAxisd(cloudKeyPoses6D->points[thisKeyInd].pitch,Eigen::Vector3d::UnitY())*
  //                            Eigen::AngleAxisd(cloudKeyPoses6D->points[thisKeyInd].roll,Eigen::Vector3d::UnitX()));
  
  //     transformLocToMap.topRightCorner(3,1) = Eigen::Vector3d(cloudKeyPoses6D->points[thisKeyInd].x, cloudKeyPoses6D->points[thisKeyInd].y, cloudKeyPoses6D->points[thisKeyInd].z);

  //     poseFile<<transformLocToMap<<std::endl;

  //     localMap.reset(new pcl::PointCloud<pcl::PointXYZI>());
  //     localMapDS.reset(new pcl::PointCloud<pcl::PointXYZI>());
  //     mapNUm = 0;
  //     saveMapNum++;
  //   }
  //     //局部地图添加保存
  // }
  double last_keypose[2]={0,0};
  //局部切分
  for (int i = 0; i < (int)cloudKeyPoses3D->size(); ++i){
    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
    *localMap += *transformPointCloud(cloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
    mapNUm++;
    Eigen::Vector3d poseLocToMap(cloudKeyPoses6D->points[thisKeyInd].x, cloudKeyPoses6D->points[thisKeyInd].y, cloudKeyPoses6D->points[thisKeyInd].z);

    if(sqrt(pow((poseLocToMap[0] - last_keypose[0]),2) + pow((poseLocToMap[1] - last_keypose[1]),2))>3){
      pcl::VoxelGrid<pcl::PointXYZI> downLocalMap;
      downLocalMap.setLeafSize(0.8, 0.8, 0.4);
      downLocalMap.setInputCloud(localMap);
      downLocalMap.filter(*localMapDS);
      *localMap = *localMapDS;
      localMapDS.reset(new pcl::PointCloud<pcl::PointXYZI>());
      for(int i = 0; i < localMap->size(); i++){
        double dis = sqrt(pow((localMap->points[i].x - poseLocToMap[0]),2) + pow((localMap->points[i].y - poseLocToMap[1]),2));
        if(dis < 80){
          localMapDS->points.push_back(localMap->points[i]);
        }
      }
      std::string file_name = savePath + "localmap/localmap_"+ std::to_string(saveMapNum)+".ply";
      pcl::io::savePLYFileASCII(file_name ,*localMapDS);
      Eigen::Matrix4d transformLocToMap  = Eigen::Matrix4d::Identity();
      transformLocToMap.topLeftCorner(3,3) = Eigen::Matrix3d(Eigen::AngleAxisd(cloudKeyPoses6D->points[thisKeyInd].yaw,Eigen::Vector3d::UnitZ())*
                             Eigen::AngleAxisd(cloudKeyPoses6D->points[thisKeyInd].pitch,Eigen::Vector3d::UnitY())*
                             Eigen::AngleAxisd(cloudKeyPoses6D->points[thisKeyInd].roll,Eigen::Vector3d::UnitX()));
  
      transformLocToMap.topRightCorner(3,1) = Eigen::Vector3d(cloudKeyPoses6D->points[thisKeyInd].x, cloudKeyPoses6D->points[thisKeyInd].y, cloudKeyPoses6D->points[thisKeyInd].z);

      poseFile<<transformLocToMap<<std::endl;

      last_keypose[0] = poseLocToMap[0];
      last_keypose[1] = poseLocToMap[1];
      localMapDS.reset(new pcl::PointCloud<pcl::PointXYZI>());
      localMapDSInv.reset(new pcl::PointCloud<pcl::PointXYZI>());
      mapNUm = 0;
      saveMapNum++;
    }
      //局部地图添加保存
  }
  std::cout<<"all local map saved"<<std::endl;
}

int main(int argc, char** argv){
  ros::init(argc, argv, "mapOptmization");
  ros::NodeHandle nh;
  ros::Subscriber subFullCloud = nh.subscribe("cur_scan", 10,  fullCallBack);
  ros::Subscriber subCloudPose = nh.subscribe("livox_odometry_mapped", 10, poseCallBack);
  ros::Subscriber subGps = nh.subscribe("fix", 10, gpsCallBack);
  ros::Subscriber subGpsTime = nh.subscribe("time_reference", 10, gpsTimeCallBack);

  std::string poseFilePath = savePath + "localmap/odom.txt";
  poseFile.open(poseFilePath);
  
  odometryFrame = "/livox_frame";
  worldFrame = "/world";
  pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_livox/mapping/map_local", 1);
  pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>("lio_livox/mapping/cloud_registered", 1);
  pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("lio_livox/mapping/map_global", 1);
  pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("lio_livox/mapping/opt_odom", 1);

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.1;
  parameters.relinearizeSkip = 1;
  isam = new gtsam::ISAM2(parameters);

  tf_broadcaster = new tf::TransformBroadcaster;
  registration_transform = new tf::StampedTransform;
  registration_transform->frame_id_ = "/world";
  registration_transform->child_frame_id_ = "/opt_frame";

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

  std::thread threadProcess{process};
  std::thread threadVisualization{visualization};
  ros::spin();

  threadVisualization.join();
  threadProcess.join();
  
  return 0;
}