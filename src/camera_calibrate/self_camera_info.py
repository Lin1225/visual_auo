#!/usr/bin/env python3


# rosrun camera_calibration cameracalibrator.py --size 7x5 --square 0.03 image:=/rgb/image_raw

# rosrun visp_test self_camera_info.py
# ROS_NAMESPACE=my_camera rosrun image_proc image_proc
# rosrun image_view image_view image:=my_camera/image_rect_color

import rospy
from sensor_msgs.msg import CameraInfo,Image

pub_info = rospy.Publisher('/my_camera/camera_info', CameraInfo, queue_size=10)
pub_img = rospy.Publisher('/my_camera/image_raw', Image, queue_size=10)
camera_info = CameraInfo()

def callback(data):
    rospy.loginfo("heard image")
    camera_info.header=data.header
    pub_info.publish(camera_info)

    pub_img.publish(data)

def set_info_1080p():
    camera_info.header.frame_id = 'camera_color_optical_frame'
    camera_info.width = int(1920)
    camera_info.height = int(1080)
    camera_info.distortion_model = 'plumb_bob'
    
    camera_info.K = [1391.421392437757, 0.0, 984.8817867585748, 0.0, 1399.062332428129, 546.0107499189386, 0.0, 0.0, 1.0]
    camera_info.D = [0.10677420376999978, -0.22224956982603475, 0.00234550219175402, 0.0046939168627538, 0.0]
    camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    camera_info.P = [1390.0489501953125, 0.0, 998.0423795295574, 0.0, 0.0, 1421.0853271484375, 547.7746690318418, 0.0, 0.0, 0.0, 1.0, 0.0]

def set_info_720p():
    camera_info.header.frame_id = 'camera_color_optical_frame'
    camera_info.width = int(1280)
    camera_info.height = int(720)
    camera_info.distortion_model = 'plumb_bob'

    camera_info.K = [614.628928745069, 0.0, 642.827137787509, 0.0, 616.8066524222719, 365.5359193507853, 0.0, 0.0, 1.0]
    camera_info.D = [0.07543316280660027, -0.034454442054812774, 0.0007653942564255259, 0.0007844840947241387, 0.0]
    camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    camera_info.P = [639.791015625, 0.0, 643.8564601298422, 0.0, 0.0, 641.4843139648438, 365.94773203009754, 0.0, 0.0, 0.0, 1.0, 0.0]


def listener():
    set_info_720p()
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/rgb/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
        # talker()
    except rospy.ROSInterruptException:
        pass