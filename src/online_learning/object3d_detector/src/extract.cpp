#include "extract.h"



void extract_left(pcl::PointCloud<pcl::PointXYZ>::Ptr left_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in){

	pcl::PointCloud<pcl::PointXYZ>::Ptr left_bbox_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	left_bbox_ptr->push_back(pcl::PointXYZ(20, -8, -2.8));
	left_bbox_ptr->push_back(pcl::PointXYZ( 21.6, -4.5,  -2.8 ));
	left_bbox_ptr->push_back(pcl::PointXYZ(-21.9, 0.143, -2.5));
	left_bbox_ptr->push_back(pcl::PointXYZ(-20,   2.78,  -2.5 ));
	left_bbox_ptr->push_back(pcl::PointXYZ(20, -8, -0.8));
	left_bbox_ptr->push_back(pcl::PointXYZ( 21.6, -5.5,  -0.8 ));
	left_bbox_ptr->push_back(pcl::PointXYZ(-21.9, 0.143, -0.5 ));
	left_bbox_ptr->push_back(pcl::PointXYZ(-20,   2.78,  -0.5 ));


	pcl::ConvexHull<pcl::PointXYZ> left_bbox_hull;
	left_bbox_hull.setInputCloud(left_bbox_ptr);
	left_bbox_hull.setDimension(3);
	std::vector<pcl::Vertices> left_polygons;//保存凸包的容器
	pcl::PointCloud<pcl::PointXYZ>::Ptr left_surface_hull (new pcl::PointCloud<pcl::PointXYZ>);
	left_bbox_hull.reconstruct(*left_surface_hull, left_polygons);

	
	pcl::CropHull<pcl::PointXYZ> left_filter;
	left_filter.setDim(3);
	left_filter.setInputCloud(cloud_in);
	left_filter.setHullIndices(left_polygons);
	left_filter.setHullCloud(left_surface_hull); //输入凸包形状
	left_filter.filter(*left_objects);
}

void extract_btm(pcl::PointCloud<pcl::PointXYZ>::Ptr btm_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in){

	pcl::PointCloud<pcl::PointXYZ>::Ptr btm_bbox_ptr (new pcl::PointCloud<pcl::PointXYZ>); 
	btm_bbox_ptr->push_back(pcl::PointXYZ(25.0,  -2.60, -2.8));
	btm_bbox_ptr->push_back(pcl::PointXYZ(25.0,   2.70, -2.8));
	btm_bbox_ptr->push_back(pcl::PointXYZ(7.00,   1.00, -2.8));
	btm_bbox_ptr->push_back(pcl::PointXYZ(7.00,   4.30, -2.8));
	btm_bbox_ptr->push_back(pcl::PointXYZ(25.0,  -2.60, -0.8));
	btm_bbox_ptr->push_back(pcl::PointXYZ(25.0,   2.70, -0.8));
	btm_bbox_ptr->push_back(pcl::PointXYZ(7.00,   1.00, -0.8));
	btm_bbox_ptr->push_back(pcl::PointXYZ(7.00,   4.30, -0.8));	

	pcl::ConvexHull<pcl::PointXYZ> btm_bbox_hull;
	btm_bbox_hull.setInputCloud(btm_bbox_ptr);
	btm_bbox_hull.setDimension(3);
	std::vector<pcl::Vertices> btm_polygons;//保存凸包的容器
	pcl::PointCloud<pcl::PointXYZ>::Ptr btm_surface_hull (new pcl::PointCloud<pcl::PointXYZ>);
	btm_bbox_hull.reconstruct(*btm_surface_hull, btm_polygons);

	pcl::CropHull<pcl::PointXYZ> btm_filter;
	btm_filter.setDim(3);
	btm_filter.setInputCloud(cloud_in);
	btm_filter.setHullIndices(btm_polygons);
	btm_filter.setHullCloud(btm_surface_hull); //输入凸包形状
	btm_filter.filter(*btm_objects);
}

void extract_center(pcl::PointCloud<pcl::PointXYZ>::Ptr center_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in){

	pcl::PointCloud<pcl::PointXYZ>::Ptr center_bbox_ptr (new pcl::PointCloud<pcl::PointXYZ>); 
	center_bbox_ptr->push_back(pcl::PointXYZ(7.05,   4.30, -2.8));
	center_bbox_ptr->push_back(pcl::PointXYZ(7.05,   -0.5, -2.8));
	center_bbox_ptr->push_back(pcl::PointXYZ(-6.55,  1.00, -2.8));
	center_bbox_ptr->push_back(pcl::PointXYZ(-5.50,  7.00, -2.8));
	center_bbox_ptr->push_back(pcl::PointXYZ(7.05,   4.30, -0.8));
	center_bbox_ptr->push_back(pcl::PointXYZ(7.05,   -0.5, -0.8));
	center_bbox_ptr->push_back(pcl::PointXYZ(-6.55,  1.00, -0.8));
	center_bbox_ptr->push_back(pcl::PointXYZ(-5.50,  7.00, -0.8));

	pcl::ConvexHull<pcl::PointXYZ> center_bbox_hull;
	center_bbox_hull.setInputCloud(center_bbox_ptr);
	center_bbox_hull.setDimension(3);
	std::vector<pcl::Vertices> center_polygons;//保存凸包的容器
	pcl::PointCloud<pcl::PointXYZ>::Ptr center_surface_hull (new pcl::PointCloud<pcl::PointXYZ>);
	center_bbox_hull.reconstruct(*center_surface_hull, center_polygons);

	pcl::CropHull<pcl::PointXYZ> center_filter;
	center_filter.setDim(3);
	center_filter.setInputCloud(cloud_in);
	center_filter.setHullIndices(center_polygons);
	center_filter.setHullCloud(center_surface_hull); //输入凸包形状
	center_filter.filter(*center_objects);
}

void extract_top(pcl::PointCloud<pcl::PointXYZ>::Ptr top_objects, pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_in){

	pcl::PointCloud<pcl::PointXYZ>::Ptr top_bbox_ptr (new pcl::PointCloud<pcl::PointXYZ>); 
	top_bbox_ptr->push_back(pcl::PointXYZ(-32.8,  8.27, -2.8));
	top_bbox_ptr->push_back(pcl::PointXYZ(-32.8,  12.27,-2.8));
	top_bbox_ptr->push_back(pcl::PointXYZ(-11.8,  3.23, -2.8));
	top_bbox_ptr->push_back(pcl::PointXYZ(-11.8,  7.71, -2.8));
	top_bbox_ptr->push_back(pcl::PointXYZ(-32.8,  8.27, -0.8));
	top_bbox_ptr->push_back(pcl::PointXYZ(-32.8,  12.27,-0.8));
	top_bbox_ptr->push_back(pcl::PointXYZ(-11.8,  3.23, -0.8));
	top_bbox_ptr->push_back(pcl::PointXYZ(-11.8,  7.71, -0.8));

	pcl::ConvexHull<pcl::PointXYZ> top_bbox_hull;
	top_bbox_hull.setInputCloud(top_bbox_ptr);
	top_bbox_hull.setDimension(3);
	std::vector<pcl::Vertices> top_polygons;//保存凸包的容器
	pcl::PointCloud<pcl::PointXYZ>::Ptr top_surface_hull (new pcl::PointCloud<pcl::PointXYZ>);
	top_bbox_hull.reconstruct(*top_surface_hull, top_polygons);

	pcl::CropHull<pcl::PointXYZ> top_filter;
	top_filter.setDim(3);
	top_filter.setInputCloud(cloud_in);
	top_filter.setHullIndices(top_polygons);
	top_filter.setHullCloud(top_surface_hull); //输入凸包形状
	top_filter.filter(*top_objects);
}