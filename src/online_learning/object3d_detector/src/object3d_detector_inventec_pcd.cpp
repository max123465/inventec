// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>
// SVM
#include "svm.h"
#include <glob.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>


pcl::PointCloud<pcl::PointXYZI>::Ptr background_cloud (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);

typedef struct feature {
  /*** for visualization ***/
  Eigen::Vector4f centroid;
  Eigen::Vector4f min;
  Eigen::Vector4f max;
  /*** for classification ***/
  int number_points;
  float min_distance;
  Eigen::Matrix3f covariance_3d;
  Eigen::Matrix3f moment_3d;
   float partial_covariance_2d[9];
   float histogram_main_2d[98];
   float histogram_second_2d[45];
  float slice[20];
  float intensity[27];
} Feature;

static const int FEATURE_SIZE = 61;

std::vector<std::string> pcd_path;
std::vector<std::string> gt_path;
std::vector<std::vector<float>> gt_xyz;
std::string pcd_filename;
float accumulate_acc = 0;
float frame_num = 0;

class Object3dDetector {
private:
  /*** Publishers and Subscribers ***/
  ros::NodeHandle node_handle_;
  ros::Subscriber point_cloud_sub_;
  ros::Publisher pose_array_pub_;
  ros::Publisher marker_array_pub_;
  ros::Publisher change_pub;
  ros::Publisher lidar_pub;
  
  bool print_fps_;
  std::string frame_id_;
  float z_limit_min_;
  float z_limit_max_;
  int cluster_size_min_;
  int cluster_size_max_;
  
  std::vector<Feature> features_;
  std::string model_file_name_;
  std::string range_file_name_;
  struct svm_node *svm_node_;
  struct svm_model *svm_model_;
  bool use_svm_model_;
  bool is_probability_model_;
  float svm_scale_range_[FEATURE_SIZE][2];
  float x_lower_;
  float x_upper_;
  float human_probability_;
  bool human_size_limit_;
  
public:
  Object3dDetector();
  ~Object3dDetector();
  
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2);
  void extractChangeParts(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_change, 
    float resolution);  
  void extractCluster(pcl::PointCloud<pcl::PointXYZI>::Ptr pc);
  void extractFeature(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, Feature &f,
		      Eigen::Vector4f &min, Eigen::Vector4f &max, Eigen::Vector4f &centroid);
  void saveFeature(Feature &f, struct svm_node *x);
  void classify();
  void acc_cal();
};

Object3dDetector::Object3dDetector() {
  point_cloud_sub_ = node_handle_.subscribe<sensor_msgs::PointCloud2>("velodyne_points", 1, &Object3dDetector::pointCloudCallback, this);
  
  ros::NodeHandle private_nh("~");
  marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  change_pub = private_nh.advertise<sensor_msgs::PointCloud2>("change_cloud", 100);
  lidar_pub = private_nh.advertise<sensor_msgs::PointCloud2>("cloud", 100);
  /*** Parameters ***/
  private_nh.param<bool>("print_fps", print_fps_, false);
  private_nh.param<std::string>("frame_id", frame_id_, "velodyne");
  private_nh.param<float>("z_limit_min", z_limit_min_, -5);
  private_nh.param<float>("z_limit_max", z_limit_max_, 5);
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 10);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 150);
  private_nh.param<float>("human_probability", human_probability_, 0.1);
  private_nh.param<bool>("human_size_limit", human_size_limit_, 1);
  
  /****** load a pre-trained svm model ******/
  private_nh.param<std::string>("model_file_name", model_file_name_, "");
  private_nh.param<std::string>("range_file_name", range_file_name_, "");


  glob_t glob_result_pcd;
  //glob("/home/aemass/Desktop/data/1212_1/tmp2/*" ,GLOB_TILDE,NULL,&glob_result_pcd);
  glob("/home/aemass/aemass_ws/data/inventec/1212_Inventec/4/pointcloud_4_delete/data/*" ,GLOB_TILDE,NULL,&glob_result_pcd);
  for(unsigned int i=0; i<glob_result_pcd.gl_pathc; ++i){
    std::string filename = glob_result_pcd.gl_pathv[i];
    pcd_path.push_back(filename);
  }

  glob_t glob_gt;
  glob("/home/aemass/aemass_ws/data/inventec/1212_Inventec/4/output_4/*" ,GLOB_TILDE,NULL,&glob_gt);
  for(unsigned int i=0; i<glob_gt.gl_pathc; ++i){
    std::string gt = glob_gt.gl_pathv[i];
    gt_path.push_back(gt);
  }

  pcl::io::loadPCDFile<pcl::PointXYZI> ("/home/aemass/aemass_ws/data/inventec/650.992488540.pcd", *background_cloud);
  for(auto &point : *background_cloud){
    //std::cout<<point<<std::endl;
    point.intensity = 1;
  }


  

  use_svm_model_ = 0;
  if((svm_model_ = svm_load_model(model_file_name_.c_str())) == NULL) {
    ROS_WARN("[object3d detector] can not load SVM model, use model-free detection.");
  } else {
    ROS_INFO("[object3d detector] load SVM model from '%s'.", model_file_name_.c_str());
    is_probability_model_ = svm_check_probability_model(svm_model_)?true:false;
    svm_node_ = (struct svm_node *)malloc((FEATURE_SIZE+1)*sizeof(struct svm_node)); // 1 more size for end index (-1)
    
    // load range file, for more details: https://github.com/cjlin1/libsvm/
    std::fstream range_file;
    range_file.open(range_file_name_.c_str(), std::fstream::in);
    if(!range_file.is_open()) {
      ROS_WARN("[object3d detector] can not load range file, use model-free detection.");
    } else {
      ROS_INFO("[object3d detector] load SVM range from '%s'.", range_file_name_.c_str());
      std::string line;
      std::vector<std::string> params;
      std::getline(range_file, line);
      std::getline(range_file, line);
      boost::split(params, line, boost::is_any_of(" "));
      x_lower_ = atof(params[0].c_str());
      x_upper_ = atof(params[1].c_str());
      int i = 0;
      while(std::getline(range_file, line)) {
	boost::split(params, line, boost::is_any_of(" "));
	svm_scale_range_[i][0] = atof(params[1].c_str());
	svm_scale_range_[i][1] = atof(params[2].c_str());
	i++;
	//std::cerr << i << " " <<  svm_scale_range_[i][0] << " " << svm_scale_range_[i][1] << std::endl;
      }
      use_svm_model_ = true;
    }
  }
}

Object3dDetector::~Object3dDetector() {
  if(use_svm_model_) {
    svm_free_and_destroy_model(&svm_model_);
    free(svm_node_);
  }
}

int pcd_counter = 0;
int frames; clock_t start_time; bool reset = true;//fps
void Object3dDetector::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2) {
  if(print_fps_)if(reset){frames=0;start_time=clock();reset=false;}//fps

  pcd_filename = pcd_path[pcd_counter];
  const size_t last_slash_idx = pcd_filename.find_last_of("\\/");
  if (std::string::npos != last_slash_idx){
    pcd_filename.erase(0, last_slash_idx + 1);
  }

  const size_t period_idx = pcd_filename.rfind('.');
  if (std::string::npos != period_idx){
    pcd_filename.erase(period_idx);
  }

  std::vector<std::vector<std::string>> gt_names;


  std::string txt_path = "/home/aemass/aemass_ws/data/inventec/1212_Inventec/4/output_4/" + pcd_filename + ".txt" ;
  std::ifstream ifs(txt_path, std::ios::in);

  if (!ifs.is_open()) {
      std::cout << "Failed to open file.\n";
  }

  std::string s;
  while (std::getline(ifs, s)) {
    std::vector<std::string> lineData;
    std::stringstream lineStream(s);
    std::string words;
    while(lineStream >> words){
      lineData.push_back(words);
    }    
    gt_names.push_back(lineData);
  }

  gt_xyz.clear();
  for(int i=0; i<gt_names.size(); i++){
    std::vector<float> line_float;
    for(int j=11; j<14; j++){
      line_float.push_back(std::stof(gt_names[i][j]));
    }
    gt_xyz.push_back(line_float);
  }
  
  ifs.close();


  pcl::io::loadPCDFile<pcl::PointXYZI> (pcd_path[pcd_counter], *cloud);
  pcd_counter+=1;
  //pcd_counter = pcd_counter % 100;

  for(auto &point2 : *cloud){
    point2.intensity = 1;
  }  

  pcl::IndicesPtr pc_indices(new std::vector<int>);
  pcl::PassThrough<pcl::PointXYZI> filter_cloud;
  filter_cloud.setInputCloud(cloud);
  filter_cloud.setFilterFieldName("z");
  filter_cloud.setFilterLimits(-2.9, -0.9);
  filter_cloud.filter(*pc_indices);

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_filter (new pcl::PointCloud<pcl::PointXYZI>);

  pcl_pc_filter->width = pc_indices->size();
  pcl_pc_filter->height = 1;
  pcl_pc_filter->points.resize (pcl_pc_filter->width * pcl_pc_filter->height);
  for (size_t i=0;  i<pc_indices->size(); ++i){
    //ROS_INFO("123123123");
    (*pcl_pc_filter)[i].x = cloud->points[(*pc_indices)[i]].x;
    (*pcl_pc_filter)[i].y = cloud->points[(*pc_indices)[i]].y;
    (*pcl_pc_filter)[i].z = cloud->points[(*pc_indices)[i]].z;
    (*pcl_pc_filter)[i].intensity = 1;
    //std::cout<<(*pcl_pc_change)[i].x<<"and"<<(*pcl_pc)[newPointIdxVector[i]].x<<std::endl;
  }



  sensor_msgs::PointCloud2 lidar_msg;
  pcl::toROSMsg(*pcl_pc_filter, lidar_msg);
  lidar_msg.header.frame_id = frame_id_;
  lidar_pub.publish(lidar_msg); 

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_change (new pcl::PointCloud<pcl::PointXYZI>);


  float resolution = 0.3f;


  extractChangeParts(cloud, pcl_pc_change, resolution);

  sensor_msgs::PointCloud2 change_msg;  
  pcl::toROSMsg(*pcl_pc_change, change_msg);
  change_msg.header.frame_id = frame_id_;
  change_pub.publish(change_msg); 

  
  extractCluster(pcl_pc_change);
  classify();
  acc_cal();
  
  if(print_fps_)if(++frames>10){std::cerr<<"[object3d_detector]: fps = "<<float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC)<<", timestamp = "<<clock()/CLOCKS_PER_SEC<<std::endl;reset = true;}//fps
}

const int nested_regions_ = 14;
int zone_[nested_regions_] = {2,3,3,3,3,3,3,2,3,3,3,3,3,3}; // for more details, see our IROS'17 paper.
void Object3dDetector::extractCluster(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) {
  features_.clear();
  
  pcl::IndicesPtr pc_indices(new std::vector<int>);
  pcl::PassThrough<pcl::PointXYZI> pass;
  pass.setInputCloud(pc);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(z_limit_min_, z_limit_max_);
  pass.filter(*pc_indices);
  
  boost::array<std::vector<int>, nested_regions_> indices_array;
  for(int i = 0; i < pc_indices->size(); i++) {
    float range = 0.0;
    for(int j = 0; j < nested_regions_; j++) {
      float d2 = pc->points[(*pc_indices)[i]].x * pc->points[(*pc_indices)[i]].x +
	pc->points[(*pc_indices)[i]].y * pc->points[(*pc_indices)[i]].y +
	pc->points[(*pc_indices)[i]].z * pc->points[(*pc_indices)[i]].z;
      if(d2 > range*range && d2 <= (range+zone_[j])*(range+zone_[j])) {
      	indices_array[j].push_back((*pc_indices)[i]);
      	break;
      }
      range += zone_[j];
    }
  }
  
  float tolerance = 0.2;
  for(int i = 0; i < nested_regions_; i++) {
    tolerance = 0.3;
    if(indices_array[i].size() > cluster_size_min_) {
      boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
      tree->setInputCloud(pc, indices_array_ptr);
      
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pc);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);
      
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
      	pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      	for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
      	  cluster->points.push_back(pc->points[*pit]);
      	cluster->width = cluster->size();
      	cluster->height = 1;
      	cluster->is_dense = true;
	
      	Eigen::Vector4f min, max, centroid;
      	pcl::getMinMax3D(*cluster, min, max);
      	pcl::compute3DCentroid(*cluster, centroid);
	
      	// Size limitation is not reasonable, but it can increase fps.
      	//0.2 1.0/ 0.2 1.0/ 0.5 2.0/
        /*
      	if(human_size_limit_ &&
	   (max[0]-min[0] < 0.3 || max[0]-min[0] > 0.7 ||
	    max[1]-min[1] < 0.3 || max[1]-min[1] > 0.7 ||
	    max[2]-min[2] < 0.4 || max[2]-min[2] > 1.8)) */
        if(human_size_limit_ &&
     (max[0]-min[0] > 1.0 ||
      //max[1]-min[1] > 0.7 ||
      max[2]-min[2] > 1.8)) 
	  continue;
	
	Feature f;
  f.centroid = centroid;
  f.min = min;
  f.max = max;

	features_.push_back(f);
      }
    }
  }
}

void Object3dDetector::extractChangeParts(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_change, float resolution){

  pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZI> octree (resolution);
  octree.setInputCloud (background_cloud);
  octree.addPointsFromInputCloud ();
  octree.switchBuffers ();
  octree.setInputCloud (pc);
  octree.addPointsFromInputCloud ();
  pcl::PointCloud<pcl::PointXYZI> differ_cloud;
  std::vector<int> newPointIdxVector;
  octree.getPointIndicesFromNewVoxels(newPointIdxVector);
  pcl_pc_change->width = newPointIdxVector.size();
  pcl_pc_change->height = 1;
  pcl_pc_change->points.resize (pcl_pc_change->width * pcl_pc_change->height);
  for (size_t i=0;  i<newPointIdxVector.size(); ++i){
    //ROS_INFO("123123123");
    (*pcl_pc_change)[i].x = (*pc)[newPointIdxVector[i]].x;
    (*pcl_pc_change)[i].y = (*pc)[newPointIdxVector[i]].y;
    (*pcl_pc_change)[i].z = (*pc)[newPointIdxVector[i]].z;
    (*pcl_pc_change)[i].intensity = 1;
    //std::cout<<(*pcl_pc_change)[i].x<<"and"<<(*pcl_pc)[newPointIdxVector[i]].x<<std::endl;
  }
}
/* *** Feature Extraction ***
 * f1 (1d): the number of points included in a cluster.
 * f2 (1d): the minimum distance of the cluster to the sensor.
 * => f1 and f2 should be used in pairs, since f1 varies with f2 changes.
 * f3 (6d): 3D covariance matrix of the cluster.
 * f4 (6d): the normalized moment of inertia tensor.
 * => Since both f3 and f4 are symmetric, we only use 6 elements from each as features.
 * ~f5 (9d): 2D covariance matrix in 3 zones, which are the upper half, and the left and right lower halves.
 * ~f6 (98d): The normalized 2D histogram for the main plane, 14 × 7 bins.
 * ~f7 (45d): The normalized 2D histogram for the secondary plane, 9 × 5 bins.
 * f8 (20d): Slice feature for the cluster.
 * f9 (27d): Intensity.
 */

void computeMomentOfInertiaTensorNormalized(pcl::PointCloud<pcl::PointXYZI> &pc, Eigen::Matrix3f &moment_3d) {
  moment_3d.setZero();
  for(size_t i = 0; i < pc.size(); i++) {
    moment_3d(0,0) += pc[i].y*pc[i].y+pc[i].z*pc[i].z;
    moment_3d(0,1) -= pc[i].x*pc[i].y;
    moment_3d(0,2) -= pc[i].x*pc[i].z;
    moment_3d(1,1) += pc[i].x*pc[i].x+pc[i].z*pc[i].z;
    moment_3d(1,2) -= pc[i].y*pc[i].z;
    moment_3d(2,2) += pc[i].x*pc[i].x+pc[i].y*pc[i].y;
  }
  moment_3d(1, 0) = moment_3d(0, 1);
  moment_3d(2, 0) = moment_3d(0, 2);
  moment_3d(2, 1) = moment_3d(1, 2);
}

/* Main plane is formed from the maximum and middle eigenvectors.
 * Secondary plane is formed from the middle and minimum eigenvectors.
 */
void computeProjectedPlane(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, Eigen::Matrix3f &eigenvectors, int axe, Eigen::Vector4f &centroid, pcl::PointCloud<pcl::PointXYZI>::Ptr plane) {
  Eigen::Vector4f coefficients;
  coefficients[0] = eigenvectors(0,axe);
  coefficients[1] = eigenvectors(1,axe);
  coefficients[2] = eigenvectors(2,axe);
  coefficients[3] = 0;
  coefficients[3] = -1 * coefficients.dot(centroid);
  for(size_t i = 0; i < pc->size(); i++) {
    float distance_to_plane =
      coefficients[0] * pc->points[i].x +
      coefficients[1] * pc->points[i].y +
      coefficients[2] * pc->points[i].z +
      coefficients[3];
    pcl::PointXYZI p;
    p.x = pc->points[i].x - distance_to_plane * coefficients[0];
    p.y = pc->points[i].y - distance_to_plane * coefficients[1];
    p.z = pc->points[i].z - distance_to_plane * coefficients[2];
    plane->points.push_back(p);
  }
}

/* Upper half, and the left and right lower halves of a pedestrian. */
void compute3ZoneCovarianceMatrix(pcl::PointCloud<pcl::PointXYZI>::Ptr plane, Eigen::Vector4f &mean, float *partial_covariance_2d) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr zone_decomposed[3];
  for(int i = 0; i < 3; i++)
    zone_decomposed[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
  for(size_t i = 0; i < plane->size(); i++) {
    if(plane->points[i].z >= mean(2)) { // upper half
      zone_decomposed[0]->points.push_back(plane->points[i]);
    } else {
      if(plane->points[i].y >= mean(1)) // left lower half
	zone_decomposed[1]->points.push_back(plane->points[i]);
      else // right lower half
	zone_decomposed[2]->points.push_back(plane->points[i]);
    }
  }
  
  Eigen::Matrix3f covariance;
  Eigen::Vector4f centroid;
  for(int i = 0; i < 3; i++) {
    pcl::compute3DCentroid(*zone_decomposed[i], centroid);
    pcl::computeCovarianceMatrix(*zone_decomposed[i], centroid, covariance);
    partial_covariance_2d[i*3+0] = covariance(0,0);
    partial_covariance_2d[i*3+1] = covariance(0,1);
    partial_covariance_2d[i*3+2] = covariance(1,1);
  }
}

void computeHistogramNormalized(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, int horiz_bins, int verti_bins, float *histogram) {
  Eigen::Vector4f min, max, min_box, max_box;
  pcl::getMinMax3D(*pc, min, max);
  float horiz_itv, verti_itv;
  horiz_itv = (max[0]-min[0]>max[1]-min[1]) ? (max[0]-min[0])/horiz_bins : (max[1]-min[1])/horiz_bins;
  verti_itv = (max[2] - min[2])/verti_bins;
  
  for(int i = 0; i < horiz_bins; i++) {
    for(int j = 0; j < verti_bins; j++) {
      if(max[0]-min[0] > max[1]-min[1]) {
	min_box << min[0]+horiz_itv*i, min[1], min[2]+verti_itv*j, 0;
	max_box << min[0]+horiz_itv*(i+1), max[1], min[2]+verti_itv*(j+1), 0;
      } else {
	min_box << min[0], min[1]+horiz_itv*i, min[2]+verti_itv*j, 0;
	max_box << max[0], min[1]+horiz_itv*(i+1), min[2]+verti_itv*(j+1), 0;
      }
      std::vector<int> indices;
      pcl::getPointsInBox(*pc, min_box, max_box, indices);
      histogram[i*verti_bins+j] = (float)indices.size() / (float)pc->size();
    }
  }
}

void computeSlice(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, int n, float *slice) {
  Eigen::Vector4f pc_min, pc_max;
  pcl::getMinMax3D(*pc, pc_min, pc_max);
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr blocks[n];
  float itv = (pc_max[2] - pc_min[2]) / n;
  if(itv > 0) {
    for(int i = 0; i < n; i++) {
      blocks[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
    for(unsigned int i = 0, j; i < pc->size(); i++) {
      j = std::min((n-1), (int)((pc->points[i].z - pc_min[2]) / itv));
      blocks[j]->points.push_back(pc->points[i]);
    }
    
    Eigen::Vector4f block_min, block_max;
    for(int i = 0; i < n; i++) {
      if(blocks[i]->size() > 0) {
	// pcl::PCA<pcl::PointXYZ> pca;
	// pcl::PointCloud<pcl::PointXYZ>::Ptr block_projected(new pcl::PointCloud<pcl::PointXYZ>);
	// pca.setInputCloud(blocks[i]);
	// pca.project(*blocks[i], *block_projected);
	pcl::getMinMax3D(*blocks[i], block_min, block_max);
      } else {
	block_min.setZero();
	block_max.setZero();
      }
      slice[i*2] = block_max[0] - block_min[0];
      slice[i*2+1] = block_max[1] - block_min[1];
    }
  } else {
    for(int i = 0; i < 20; i++)
      slice[i] = 0;
  }
}

void computeIntensity(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, int bins, float *intensity) {
  float sum = 0, mean = 0, sum_dev = 0;
  float min = FLT_MAX, max = -FLT_MAX;
  for(int i = 0; i < 27; i++)
    intensity[i] = 0;
  
  for(size_t i = 0; i < pc->size(); i++) {
    sum += pc->points[i].intensity;
    min = std::min(min, pc->points[i].intensity);
    max = std::max(max, pc->points[i].intensity);
  }
  mean = sum / pc->size();
  
  for(size_t i = 0; i < pc->size(); i++) {
    sum_dev += (pc->points[i].intensity-mean)*(pc->points[i].intensity-mean);
    int ii = std::min(float(bins-1), std::floor((pc->points[i].intensity-min)/((max-min)/bins)));
    intensity[ii]++;
  }
  intensity[25] = sqrt(sum_dev/pc->size());
  intensity[26] = mean;
}

void Object3dDetector::extractFeature(pcl::PointCloud<pcl::PointXYZI>::Ptr pc, Feature &f,
				      Eigen::Vector4f &min, Eigen::Vector4f &max, Eigen::Vector4f &centroid) {
  f.centroid = centroid;
  f.min = min;
  f.max = max;

  pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi;

  for(size_t i=0; i<pc->size(); i++){
    xyzi->points[i].x = pc->points[i].x;
    xyzi->points[i].y = pc->points[i].y;
    xyzi->points[i].z = pc->points[i].z;
    xyzi->points[i].intensity = 1;
  }

  if(use_svm_model_) {
    // f1: Number of points included the cluster.
    f.number_points = xyzi->size();
    // f2: The minimum distance to the cluster.
    f.min_distance = FLT_MAX;
    float d2; //squared Euclidean distance
    for(int i = 0; i < xyzi->size(); i++) {
      d2 = xyzi->points[i].x*xyzi->points[i].x + xyzi->points[i].y*xyzi->points[i].y + xyzi->points[i].z*xyzi->points[i].z;
      if(f.min_distance > d2)
	f.min_distance = d2;
    }
    //f.min_distance = sqrt(f.min_distance);
    
    pcl::PCA<pcl::PointXYZI> pca;
    pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi_projected(new pcl::PointCloud<pcl::PointXYZI>);
    pca.setInputCloud(xyzi);
    pca.project(*xyzi, *xyzi_projected);
    // f3: 3D covariance matrix of the cluster.
    pcl::computeCovarianceMatrixNormalized(*xyzi_projected, centroid, f.covariance_3d);
    // f4: The normalized moment of inertia tensor.
    computeMomentOfInertiaTensorNormalized(*xyzi_projected, f.moment_3d);
    // Navarro et al. assume that a pedestrian is in an upright position.
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr main_plane(new pcl::PointCloud<pcl::PointXYZI>), secondary_plane(new pcl::PointCloud<pcl::PointXYZI>);
    computeProjectedPlane(xyzi, pca.getEigenVectors(), 2, centroid, main_plane);
    computeProjectedPlane(xyzi, pca.getEigenVectors(), 1, centroid, secondary_plane);
    // f5: 2D covariance matrix in 3 zones, which are the upper half, and the left and right lower halves.
    //compute3ZoneCovarianceMatrix(main_plane, pca.getMean(), f.partial_covariance_2d);
    // f6 and f7
    //computeHistogramNormalized(main_plane, 7, 14, f.histogram_main_2d);
    //computeHistogramNormalized(secondary_plane, 5, 9, f.histogram_second_2d);
    // f8
    computeSlice(xyzi, 10, f.slice);
    // f9
    computeIntensity(xyzi, 25, f.intensity);
  }
}

void Object3dDetector::saveFeature(Feature &f, struct svm_node *x) {
  x[0].index  = 1;  x[0].value  = f.number_points; // libsvm indices start at 1
  x[1].index  = 2;  x[1].value  = f.min_distance;
  x[2].index  = 3;  x[2].value  = f.covariance_3d(0,0);
  x[3].index  = 4;  x[3].value  = f.covariance_3d(0,1);
  x[4].index  = 5;  x[4].value  = f.covariance_3d(0,2);
  x[5].index  = 6;  x[5].value  = f.covariance_3d(1,1);
  x[6].index  = 7;  x[6].value  = f.covariance_3d(1,2);
  x[7].index  = 8;  x[7].value  = f.covariance_3d(2,2);
  x[8].index  = 9;  x[8].value  = f.moment_3d(0,0);
  x[9].index  = 10; x[9].value  = f.moment_3d(0,1);
  x[10].index = 11; x[10].value = f.moment_3d(0,2);
  x[11].index = 12; x[11].value = f.moment_3d(1,1);
  x[12].index = 13; x[12].value = f.moment_3d(1,2);
  x[13].index = 14; x[13].value = f.moment_3d(2,2);
  // for(int i = 0; i < 9; i++) {
  //   x[i+14].index = i+15;
  //   x[i+14].value = f.partial_covariance_2d[i];
  // }
  // for(int i = 0; i < 98; i++) {
  // 	x[i+23].index = i+24;
  // 	x[i+23].value = f.histogram_main_2d[i];
  // }
  // for(int i = 0; i < 45; i++) {
  // 	x[i+121].index = i+122;
  // 	x[i+121].value = f.histogram_second_2d[i];
  // }
  for(int i = 0; i < 20; i++) {
    x[i+14].index = i+15;
    x[i+14].value = f.slice[i];
  }
  for(int i = 0; i < 27; i++) {
    x[i+34].index = i+35;
    x[i+34].value = f.intensity[i];
  }
  x[FEATURE_SIZE].index = -1;
  
  // for(int i = 0; i < FEATURE_SIZE; i++) {
  //   std::cerr << x[i].index << ":" << x[i].value << " ";
  //   std::cerr << std::endl;
  // }
}
  
void Object3dDetector::classify() {
  visualization_msgs::MarkerArray marker_array;
  geometry_msgs::PoseArray pose_array;
  for(std::vector<Feature>::iterator it = features_.begin(); it != features_.end(); ++it) {
    if(use_svm_model_) {
      saveFeature(*it, svm_node_);
      ROS_INFO("safe saveFeature");
      //std::cerr << "test_id = " << it->id << ", number_points = " << it->number_points << ", min_distance = " << it->min_distance << std::endl;
      
      // scale data
      for(int i = 0; i < FEATURE_SIZE; i++) {
      	if(svm_scale_range_[i][0] == svm_scale_range_[i][1]) // skip single-valued attribute
      	  continue;
      	if(svm_node_[i].value == svm_scale_range_[i][0])
      	  svm_node_[i].value = x_lower_;
      	else if(svm_node_[i].value == svm_scale_range_[i][1])
      	  svm_node_[i].value = x_upper_;
      	else
      	  svm_node_[i].value = x_lower_ + (x_upper_ - x_lower_) * (svm_node_[i].value - svm_scale_range_[i][0]) / (svm_scale_range_[i][1] - svm_scale_range_[i][0]);
      }

      ROS_INFO("safe scale data");
      // predict
      if(is_probability_model_) {
	double prob_estimates[svm_model_->nr_class];
      	svm_predict_probability(svm_model_, svm_node_, prob_estimates);
	if(prob_estimates[0] < human_probability_)
	  continue;
      } else {
      	if(svm_predict(svm_model_, svm_node_) != 1)
      	  continue;
      }
    }
    
    visualization_msgs::Marker marker;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = frame_id_;
    marker.ns = "object3d";
    marker.id = it-features_.begin();
    marker.type = visualization_msgs::Marker::LINE_LIST;
    geometry_msgs::Point p[24];
    p[0].x = it->max[0]; p[0].y = it->max[1]; p[0].z = it->max[2];
    p[1].x = it->min[0]; p[1].y = it->max[1]; p[1].z = it->max[2];
    p[2].x = it->max[0]; p[2].y = it->max[1]; p[2].z = it->max[2];
    p[3].x = it->max[0]; p[3].y = it->min[1]; p[3].z = it->max[2];
    p[4].x = it->max[0]; p[4].y = it->max[1]; p[4].z = it->max[2];
    p[5].x = it->max[0]; p[5].y = it->max[1]; p[5].z = it->min[2];
    p[6].x = it->min[0]; p[6].y = it->min[1]; p[6].z = it->min[2];
    p[7].x = it->max[0]; p[7].y = it->min[1]; p[7].z = it->min[2];
    p[8].x = it->min[0]; p[8].y = it->min[1]; p[8].z = it->min[2];
    p[9].x = it->min[0]; p[9].y = it->max[1]; p[9].z = it->min[2];
    p[10].x = it->min[0]; p[10].y = it->min[1]; p[10].z = it->min[2];
    p[11].x = it->min[0]; p[11].y = it->min[1]; p[11].z = it->max[2];
    p[12].x = it->min[0]; p[12].y = it->max[1]; p[12].z = it->max[2];
    p[13].x = it->min[0]; p[13].y = it->max[1]; p[13].z = it->min[2];
    p[14].x = it->min[0]; p[14].y = it->max[1]; p[14].z = it->max[2];
    p[15].x = it->min[0]; p[15].y = it->min[1]; p[15].z = it->max[2];
    p[16].x = it->max[0]; p[16].y = it->min[1]; p[16].z = it->max[2];
    p[17].x = it->max[0]; p[17].y = it->min[1]; p[17].z = it->min[2];
    p[18].x = it->max[0]; p[18].y = it->min[1]; p[18].z = it->max[2];
    p[19].x = it->min[0]; p[19].y = it->min[1]; p[19].z = it->max[2];
    p[20].x = it->max[0]; p[20].y = it->max[1]; p[20].z = it->min[2];
    p[21].x = it->min[0]; p[21].y = it->max[1]; p[21].z = it->min[2];
    p[22].x = it->max[0]; p[22].y = it->max[1]; p[22].z = it->min[2];
    p[23].x = it->max[0]; p[23].y = it->min[1]; p[23].z = it->min[2];
    for(int i = 0; i < 24; i++)
      marker.points.push_back(p[i]);
    marker.scale.x = 0.1;
    marker.color.a = 1.0;
    if(!use_svm_model_) {
      marker.color.r = 0.0;
      marker.color.g = 0.5;
      marker.color.b = 1.0;
    } else {
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.5;
    }
    
    marker.lifetime = ros::Duration(0.1);
    marker_array.markers.push_back(marker);
    
    geometry_msgs::Pose pose;
    pose.position.x = it->centroid[0];
    pose.position.y = it->centroid[1];
    pose.position.z = it->centroid[2];
    pose.orientation.w = 1;
    pose_array.poses.push_back(pose);


  }
  
  if(marker_array.markers.size()) {
    marker_array_pub_.publish(marker_array);
  }
  if(pose_array.poses.size()) {
    pose_array.header.stamp = ros::Time::now();
    pose_array.header.frame_id = frame_id_;
    pose_array_pub_.publish(pose_array);
  }
}

  void Object3dDetector::acc_cal(){

    

    std::vector<float> found_vec;
    for(std::vector<Feature>::iterator it = features_.begin(); it != features_.end(); ++it) {
      std::vector<float> dis_vec;

      for(int i=0; i<gt_xyz.size(); i++){
        float square_sum = pow((it->centroid[0] - gt_xyz[i][0]), 2) + pow((it->centroid[1] - gt_xyz[i][1]), 2) + pow((it->centroid[2] - gt_xyz[i][2]), 2);
        float dis = std::sqrt(square_sum);
        dis_vec.push_back(dis);       
      }
      sort(dis_vec.begin(), dis_vec.end());
      if(dis_vec[0]<=1) found_vec.push_back(dis_vec[0]);
    }

    float acc = (float(found_vec.size()) / float(gt_xyz.size()))*100 ;
    frame_num += 1;
    accumulate_acc = (accumulate_acc + acc);

    std::cout<<"Pedestrians: "<<found_vec.size()<<std::endl;
    std::cout<<"Pedestrians in gt: "<<gt_xyz.size()<<std::endl;
    
    std::cout<<"Acc is: "<< acc <<" %" << std::endl;
    std::cout<<"Avg Acc is: "<< accumulate_acc / frame_num<<" %" << std::endl;

    if(acc<=70) std::cout<<"該刪: "<< pcd_filename<<std::endl;
  }

int main(int argc, char **argv) {  
  ros::init(argc, argv, "object3d_detector");
  Object3dDetector d;
  ros::spin();
  return 0;
}
