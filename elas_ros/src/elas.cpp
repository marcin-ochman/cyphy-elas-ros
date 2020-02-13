/*
 Copywrite 2012. All rights reserved.
 Cyphy Lab, https://wiki.qut.edu.au/display/cyphy/Robotics,+Vision+and+Sensor+Networking+at+QUT
 Queensland University of Technology
 Brisbane, Australia

 Author: Patrick Ross
 Contact: patrick.ross@connect.qut.edu.au

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <ros/ros.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <image_transport/subscriber_filter.h>

#include <image_geometry/stereo_camera_model.h>

#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <elas_ros/ElasFrameData.h>

#include <libelas/elas.h>

#include <dynamic_reconfigure/server.h>
#include <nodelet/nodelet.h>

#include <elas_ros/ElasParametersConfig.h>

namespace elas_ros
{

class ElasProcNodelet : public nodelet::Nodelet
{
  public:
    void onInit() override
    {
        const std::string transport = "raw";

        ros::NodeHandle local_nh = getNodeHandle();
        local_nh.param("queue_size", queue_size_, 5);

        // Topics
        std::string stereo_ns = nh.resolveName("stereo");
        std::string left_topic = ros::names::clean(stereo_ns + "/left/" + nh.resolveName("image"));
        std::string right_topic = ros::names::clean(stereo_ns + "/right/" + nh.resolveName("image"));
        std::string left_info_topic = stereo_ns + "/left/camera_info";
        std::string right_info_topic = stereo_ns + "/right/camera_info";

        image_transport::ImageTransport it(nh);
        left_sub_.subscribe(it, left_topic, 1, transport);
        right_sub_.subscribe(it, right_topic, 1, transport);
        left_info_sub_.subscribe(nh, left_info_topic, 1);
        right_info_sub_.subscribe(nh, right_info_topic, 1);

        ROS_INFO("Subscribing to:\n%s\n%s\n%s\n%s", left_topic.c_str(), right_topic.c_str(), left_info_topic.c_str(),
                 right_info_topic.c_str());

        pub_disparity_ = local_nh.advertise<stereo_msgs::DisparityImage>("disparity", 1);

        // Synchronize input topics. Optionally do approximate synchronization.
        bool approx;
        local_nh.param("approximate_sync", approx, true);

        if(approx)
        {
            approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(queue_size_), left_sub_, right_sub_,
                                                        left_info_sub_, right_info_sub_));
            approximate_sync_->registerCallback(boost::bind(&ElasProcNodelet::process, this, _1, _2, _3, _4));
        }
        else
        {
            exact_sync_.reset(
                new ExactSync(ExactPolicy(queue_size_), left_sub_, right_sub_, left_info_sub_, right_info_sub_));
            exact_sync_->registerCallback(boost::bind(&ElasProcNodelet::process, this, _1, _2, _3, _4));
        }

        elas_.reset(new Elas());

        dynamicReconfigureServer =
            std::make_unique<DynamicReconfigureServer>(nh.resolveName("/elas_ros/dynamic_parameters"));
        dynamicReconfigureServer->setCallback(boost::bind(&ElasProcNodelet::updateParameters, this, _1, _2));
    }

    void updateParameters(ElasParametersConfig& config, uint32_t level)
    {
        ROS_INFO("Updating dynamic parameters");

        param.disp_min = config.disp_min;
        param.disp_max = config.disp_max;
        param.support_threshold = config.support_threshold;
        param.support_texture = config.support_texture;
        param.candidate_stepsize = config.candidate_stepsize;
        param.incon_window_size = config.incon_window_size;
        param.incon_threshold = config.incon_threshold;
        param.incon_min_support = config.incon_min_support;
        param.add_corners = config.add_corners;
        param.grid_size = config.grid_size;
        param.beta = config.beta;
        param.gamma = config.gamma;
        param.sigma = config.sigma;
        param.sradius = config.sradius;
        param.match_texture = config.match_texture;
        param.lr_threshold = config.lr_threshold;
        param.speckle_sim_threshold = config.speckle_sim_threshold;
        param.speckle_size = config.speckle_size;
        param.ipol_gap_width = config.ipol_gap_width;
        param.filter_median = config.filter_median;
        param.filter_adaptive_mean = config.filter_adaptive_mean;
        param.postprocess_only_left = config.postprocess_only_left;
        param.subsampling = config.subsampling;
    }

    using DynamicReconfigureServer = dynamic_reconfigure::Server<ElasParametersConfig>;
    using Subscriber = image_transport::SubscriberFilter;
    using InfoSubscriber = message_filters::Subscriber<sensor_msgs::CameraInfo>;
    using Publisher = image_transport::Publisher;
    using ExactPolicy = message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image,
                                                                  sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>;
    using ApproximatePolicy =
        message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
                                                        sensor_msgs::CameraInfo>;
    using ExactSync = message_filters::Synchronizer<ExactPolicy>;
    using ApproximateSync = message_filters::Synchronizer<ApproximatePolicy>;
    using PointCloud = pcl::PointCloud<pcl::PointXYZRGB>;

    stereo_msgs::DisparityImagePtr prepareNewDisparityMessage(sensor_msgs::ImageConstPtr l_image_msg,
                                                              sensor_msgs::CameraInfoConstPtr l_info_msg)
    {
        auto disp_msg = boost::make_shared<stereo_msgs::DisparityImage>();

        disp_msg->header = l_info_msg->header;
        disp_msg->image.header = l_info_msg->header;
        disp_msg->image.height = l_image_msg->height;
        disp_msg->image.width = l_image_msg->width;
        disp_msg->image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        disp_msg->image.step = disp_msg->image.width * sizeof(float);
        disp_msg->image.data.resize(disp_msg->image.height * disp_msg->image.step);
        disp_msg->min_disparity = param.disp_min;
        disp_msg->max_disparity = param.disp_max;
        disp_msg->delta_d = 1.0f;
        disp_msg->f = model_.right().fx();
        disp_msg->T = model_.baseline();

        return disp_msg;
    }

    void process(const sensor_msgs::ImageConstPtr& l_image_msg, const sensor_msgs::ImageConstPtr& r_image_msg,
                 const sensor_msgs::CameraInfoConstPtr& l_info_msg, const sensor_msgs::CameraInfoConstPtr& r_info_msg)
    {
        ROS_ASSERT(r_image_msg->encoding == sensor_msgs::image_encodings::MONO8)
        ROS_ASSERT(l_image_msg->encoding == sensor_msgs::image_encodings::MONO8)
        ROS_ASSERT(l_image_msg->step == r_image_msg->step);
        ROS_ASSERT(l_image_msg->width == r_image_msg->width);
        ROS_ASSERT(l_image_msg->height == r_image_msg->height);

        auto copyOfParameters = param;
        // Update the camera model
        model_.fromCameraInfo(l_info_msg, r_info_msg);

        // Allocate new disparity image message

        stereo_msgs::DisparityImagePtr disp_msg = prepareNewDisparityMessage(l_image_msg, l_info_msg->step);

        uint8_t *l_image_data, *r_image_data;

        l_image_data = const_cast<uint8_t*>(&(l_image_msg->data[0]));
        r_image_data = const_cast<uint8_t*>(&(r_image_msg->data[0]));

        const int32_t dims[3] = {l_image_msg->width, l_image_msg->height, l_image_msg};
        float* l_disp_data = reinterpret_cast<float*>(&disp_msg->image.data[0]);
        std::unique_ptr<float[]> r_disp_data{new float[r_image_msg->width * r_image_msg->height * sizeof(float)]};

        elas_->process(l_image_data, r_image_data, l_disp_data, r_disp_data.get(), dims, copyOfParameters);
        pub_disparity_.publish(disp_msg);
    }

  private:
    ros::NodeHandle nh;
    Subscriber left_sub_, right_sub_;
    InfoSubscriber left_info_sub_, right_info_sub_;
    boost::shared_ptr<ExactSync> exact_sync_;
    boost::shared_ptr<ApproximateSync> approximate_sync_;
    boost::shared_ptr<Elas> elas_;
    int queue_size_;

    image_geometry::StereoCameraModel model_;
    ros::Publisher pub_disparity_;
    Elas::Parameters param;
    std::unique_ptr<DynamicReconfigureServer> dynamicReconfigureServer;
};

} // namespace elas_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(elas_ros::ElasProcNodelet, nodelet::Nodelet)
