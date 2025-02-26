#include "LidarFeatureExtractor/LidarFeatureExtractor.h"

typedef pcl::PointXYZINormal PointType;

ros::Publisher pubFullLaserCloud;
ros::Publisher pubSharpCloud;
ros::Publisher pubFlatCloud;
ros::Publisher pubNonFeature;

LidarFeatureExtractor* lidarFeatureExtractor;
pcl::PointCloud<PointType>::Ptr laserCloud;
pcl::PointCloud<PointType>::Ptr laserConerCloud;
pcl::PointCloud<PointType>::Ptr laserSurfCloud;
pcl::PointCloud<PointType>::Ptr laserNonFeatureCloud;
int Lidar_Type = 0;
int N_SCANS = 6;
bool Feature_Mode = false;
bool Use_seg = false;


//去除动态物体并提取特征点
void lidarCallBackHorizon(const livox_ros_driver::CustomMsgConstPtr &msg) {

  sensor_msgs::PointCloud2 msg2; //每帧24000个点，10hz
  //Use_seg=1，使用分割模式去除动态障碍物，否则不去除。
  //laserConerCloud角点，laserSurfCloud平面点，laserNonFeatureCloud？ N_SCANS:horizon激光雷达线数，最大为6，在每个点里都有?为啥是6
  if(Use_seg){
    lidarFeatureExtractor->FeatureExtract_with_segment(msg, laserCloud, laserConerCloud, laserSurfCloud, laserNonFeatureCloud, msg2,N_SCANS);
  }//laserCloud normal_z包含类别 1，2，3分别代表 laserConerFeature，laserSurfFeature，laserNonFeature
  else{
    lidarFeatureExtractor->FeatureExtract(msg, laserCloud, laserConerCloud, laserSurfCloud,N_SCANS);
  } 

  sensor_msgs::PointCloud2 laserCloudMsg;
  pcl::toROSMsg(*laserCloud, laserCloudMsg);
  laserCloudMsg.header = msg->header;
  //laserCloudMsg.header.stamp.fromNSec(msg->timebase+msg->points.back().offset_time);
  laserCloudMsg.header.stamp.fromNSec(msg->header.stamp.toNSec()+msg->points.back().offset_time);
  pubFullLaserCloud.publish(laserCloudMsg);
  // std::cout<<std::fixed<<std::setprecision(4)<<"scan_time0="<<msg->timebase<<std::endl;
  // std::cout<<std::fixed<<std::setprecision(4)<<"scan_time1="<<msg->points.back().offset_time<<std::endl;
  // std::cout<<std::fixed<<std::setprecision(4)<<"scan_time2="<<laserCloudMsg.header.stamp.toSec()<<std::endl;

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ScanRegistration");
  ros::NodeHandle nodeHandler("~");

  ros::Subscriber customCloud;

  std::string config_file;
  nodeHandler.getParam("config_file", config_file);

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cout << "config_file error: cannot open " << config_file << std::endl;
    return false;
  }

  //读取confige里的参数
  Lidar_Type = static_cast<int>(fsSettings["Lidar_Type"]);
  N_SCANS = static_cast<int>(fsSettings["Used_Line"]);
  Feature_Mode = static_cast<int>(fsSettings["Feature_Mode"]);
  Use_seg = static_cast<int>(fsSettings["Use_seg"]);

  int NumCurvSize = static_cast<int>(fsSettings["NumCurvSize"]);
  float DistanceFaraway = static_cast<float>(fsSettings["DistanceFaraway"]);
  int NumFlat = static_cast<int>(fsSettings["NumFlat"]);
  int PartNum = static_cast<int>(fsSettings["PartNum"]);
  float FlatThreshold = static_cast<float>(fsSettings["FlatThreshold"]);
  float BreakCornerDis = static_cast<float>(fsSettings["BreakCornerDis"]);
  float LidarNearestDis = static_cast<float>(fsSettings["LidarNearestDis"]);
  float KdTreeCornerOutlierDis = static_cast<float>(fsSettings["KdTreeCornerOutlierDis"]);

  laserCloud.reset(new pcl::PointCloud<PointType>);
  laserConerCloud.reset(new pcl::PointCloud<PointType>);
  laserSurfCloud.reset(new pcl::PointCloud<PointType>);
  laserNonFeatureCloud.reset(new pcl::PointCloud<PointType>);

  customCloud = nodeHandler.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar_3WEDH5900101251", 100, &lidarCallBackHorizon);

  pubFullLaserCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_full_cloud", 10);

  //这三个topic并没有发出
  pubSharpCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_less_sharp_cloud", 10);
  pubFlatCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_less_flat_cloud", 10);
  pubNonFeature = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_nonfeature_cloud", 10);

  lidarFeatureExtractor = new LidarFeatureExtractor(N_SCANS,NumCurvSize,DistanceFaraway,NumFlat,PartNum,
                                                    FlatThreshold,BreakCornerDis,LidarNearestDis,KdTreeCornerOutlierDis);

  ros::spin();

  return 0;
}

