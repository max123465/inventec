#ifndef EXTRACT_H
#define EXTRACT_H

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include <pcl/filters/crop_hull.h>
#include <pcl/surface/concave_hull.h>

void extract_left(pcl::PointCloud<pcl::PointXYZ>::Ptr left_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in);
void extract_btm(pcl::PointCloud<pcl::PointXYZ>::Ptr right_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in);
void extract_center(pcl::PointCloud<pcl::PointXYZ>::Ptr center_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in);
void extract_top(pcl::PointCloud<pcl::PointXYZ>::Ptr top_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in);

#endif
