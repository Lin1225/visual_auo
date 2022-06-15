#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import os

import cv2
from argparse import ArgumentParser

import timeit
import rospy
import ros_numpy
import tf2_ros
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from tm_msgs.msg import FeedbackState

import message_filters

import copy
import threading

from skimage import measure
from scipy.spatial.transform import Rotation as Ra

import torch
import torch.nn as nn
from cnn import SegmentationModel as net
from datasets.dataset_Class import PoseDataset as PoseDataset_semantics



lock = threading.Lock()
data_lock = threading.Lock()
os.environ['NUMEXPR_MAX_THREADS'] = '12'

class detector:
    def __init__(self,args) -> None:
        self.robot_base2camera = 0
        self.cv_image = 0
        self.depth_image = 0
        self.args = args
        self.rgb_time = 0
        self.depth_time = 0
        self.pc_time = 0
        self.o3d_pc = 0
        self.start = 0
        self.thread_id_list = []

        self.modelA = net.EESPNet_Seg(args.classes, s=args.s)
        if not os.path.isfile(args.pretrained):
            print('ESPnet Pre-trained model file does not exist. Please check ./pretrained_models folder')
            exit(-1)
        self.modelA = nn.DataParallel(self.modelA)
        self.modelA.load_state_dict(torch.load(args.pretrained))
        if args.gpu:
            self.modelA = self.modelA.cuda()

        # # set to evaluation mode
        self.modelA.eval()
        self.conter = 0
        
        self.br = tf2_ros.TransformBroadcaster()

        
        
        if args.mode == 'Pointcloud':
            # arm_sub = rospy.Subscriber('/feedback_states', FeedbackState, self.arm_callback,queue_size=1)
            # pc_sub = rospy.Subscriber('/points2', PointCloud2, self.pc_callback)
            arm_sub_mf = message_filters.Subscriber('/feedback_states', FeedbackState)
            pc_sub_mf = message_filters.Subscriber('/points2',PointCloud2)
            ts = message_filters.ApproximateTimeSynchronizer([pc_sub_mf,arm_sub_mf], 10, 1, allow_headerless=False)
            ts.registerCallback(self.all_callback_pc)

        else :
            # depth_sub = rospy.Subscriber("/depth_to_rgb/image_raw", Image, self.depth_callback,queue_size=1)
            # rgb_sub = rospy.Subscriber("/rgb/image_raw", Image, self.rgb_callback,queue_size=1)
            # arm_sub = rospy.Subscriber('/feedback_states', FeedbackState, self.arm_callback,queue_size=1)

            arm_sub_mf = message_filters.Subscriber('/feedback_states', FeedbackState)
            depth_sub_mf = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)
            # rgb_sub_mf = message_filters.Subscriber('/rgb/image_raw', Image)
            rgb_sub_mf = message_filters.Subscriber('/rgb/image_raw', Image)
            ts = message_filters.ApproximateTimeSynchronizer([arm_sub_mf, rgb_sub_mf, depth_sub_mf], 1, 0.1, allow_headerless=False)
            ts.registerCallback(self.all_callback_esp)
        
        self.pub_armbase2Obj = rospy.Publisher('/my_den_armbase2Obj', Odometry, queue_size=1)


        self.source = o3d.io.read_point_cloud("obj_01.ply")
        voxel_size=0.01
        self.source_down, self.source_fpfh = self.preprocess_point_cloud(self.source, voxel_size)

         
        end_2_camera_t = np.array([-0.00023549566695874437, 0.13491215388963854, 0.05961087242911679]) # end(tool0)2rgb_camera_link
        end_2_camera_r = Ra.from_quat([-0.4976347570176995, -0.5014797154560687, -0.5017517550955758, 0.4991221492305714])

        end_2_camera_r = end_2_camera_r.as_matrix()
        camera_2_rgb_t = np.array([-0.0038808,  -0.031956,  -0.000166]) # end(tool0)2rgb_camera_link
        camera_2_rgb_r = Ra.from_quat([0.50158, -0.50119, 0.49898, -0.49825])
        camera_2_rgb_r = camera_2_rgb_r.as_matrix()
        

        end_2_camera_TF = np.r_[np.c_[np.array(end_2_camera_r),end_2_camera_t.reshape((3,1))],np.array([0,0,0,1]).reshape((1,4))]
        camera_2_rgbb_TF = np.r_[np.c_[np.array(camera_2_rgb_r),camera_2_rgb_t.reshape((3,1))],np.array([0,0,0,1]).reshape((1,4))]
        self.end_2_camera_TF = np.matmul(end_2_camera_TF, camera_2_rgbb_TF)


        print("we have done it")
        
    def arm_callback(self,data):
        trans =[data.tool_pose[0],data.tool_pose[1],data.tool_pose[2]]
        rot = Ra.from_euler('xyz',[data.tool_pose[3],data.tool_pose[4],data.tool_pose[5]], degrees=False).as_matrix()
        
        ARM_2_end_TF = np.r_[np.c_[np.array(rot),np.array(trans).reshape(3,1)],np.array([0,0,0,1]).reshape(1,4)]

        self.robot_base2camera = np.matmul(ARM_2_end_TF, self.end_2_camera_TF)
  
    def depth_callback(self,data):
        self.depth_time = data.header.stamp
        # Convert image to OpenCV format
        try:
            cv_image = CvBridge().imgmsg_to_cv2(data,'16UC1')
        except CvBridgeError as e:
            print(e)
            
        self.depth_image = cv_image

    def rgb_callback(self,data):
        self.rgb_time = data.header.stamp
        try:
            self.img_ = CvBridge().imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def all_callback_esp(self,arm_dataX,rgb_dataX,depth_dataX):
        # print("get ",self.conter)
        # self.conter+=1
        if data_lock.acquire():
            depth_data = copy.deepcopy(depth_dataX)
            rgb_data = copy.deepcopy(rgb_dataX)
            arm_data = copy.deepcopy(arm_dataX)
            trans =[arm_data.tool_pose[0],arm_data.tool_pose[1],arm_data.tool_pose[2]]
            rot = Ra.from_euler('xyz',[arm_data.tool_pose[3],arm_data.tool_pose[4],arm_data.tool_pose[5]], degrees=False).as_matrix()
            ARM_2_end_TF = np.r_[np.c_[np.array(rot),np.array(trans).reshape(3,1)],np.array([0,0,0,1]).reshape(1,4)]
            self.robot_base2camera = np.matmul(ARM_2_end_TF, self.end_2_camera_TF)   
            # print("depth time s ",depth_data.header.stamp.secs)   
            # print("depth time ns ",depth_data.header.stamp.nsecs)
            # print("rgb time s ",rgb_data.header.stamp.secs)   
            # print("rgb time ns ",rgb_data.header.stamp.nsecs)
            # print("arm time s ",arm_data.header.stamp.secs)   
            # print("arm time ns ",arm_data.header.stamp.nsecs)
            data_lock.release()
            
        self.depth_time = depth_data.header.stamp
        self.rgb_time = rgb_data.header.stamp

         
        # Convert image to OpenCV format
        try:
            self.depth_image = CvBridge().imgmsg_to_cv2(depth_data,'16UC1')
            self.img_ = CvBridge().imgmsg_to_cv2(rgb_data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.start = 1
            
    def all_callback_pc(self,ros_cloudX,arm_dataX):
        # print("get once")
        if data_lock.acquire():
            ros_cloud = copy.deepcopy(ros_cloudX)
            arm_data = copy.deepcopy(arm_dataX)
            

            self.pc_time = ros_cloud.header.stamp
            
            trans =[arm_data.tool_pose[0],arm_data.tool_pose[1],arm_data.tool_pose[2]]
            rot = Ra.from_euler('xyz',[arm_data.tool_pose[3],arm_data.tool_pose[4],arm_data.tool_pose[5]], degrees=False).as_matrix()
            ARM_2_end_TF = np.r_[np.c_[np.array(rot),np.array(trans).reshape(3,1)],np.array([0,0,0,1]).reshape(1,4)]
            self.robot_base2camera = np.matmul(ARM_2_end_TF, self.end_2_camera_TF)

            self.o3d_pc=self.rospc_to_o3dpc(ros_cloud, remove_nans=True, use_rgb=False)
            data_lock.release()
        self.start =1

    def preprocess_point_cloud(self,pcd, voxel_size):
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 3
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
        return pcd_down, pcd_fpfh

    def rospc_to_o3dpc(self,rospc, remove_nans=False, use_rgb=False):
        """ covert ros point cloud to open3d point cloud
        Args: 
            rospc (sensor.msg.PointCloud2): ros point cloud message
            remove_nans (bool): if true, ignore the NaN points
        Returns: 
            o3dpc (open3d.geometry.PointCloud): open3d point cloud
        """
        field_names = [field.name for field in rospc.fields]
        is_rgb = 'rgb' in field_names
        cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(rospc).ravel()
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]
        if is_rgb and use_rgb:
            cloud_npy = np.zeros(cloud_array.shape + (4,), dtype=float)
        else: 
            cloud_npy = np.zeros(cloud_array.shape + (3,), dtype=float)
        
        cloud_npy[...,0] = cloud_array['x']
        cloud_npy[...,1] = cloud_array['y']
        cloud_npy[...,2] = cloud_array['z']
        o3dpc = o3d.geometry.PointCloud()

        if len(np.shape(cloud_npy)) == 3:
            cloud_npy = np.reshape(cloud_npy[:, :, :3], [-1, 3], 'F')
        o3dpc.points = o3d.utility.Vector3dVector(cloud_npy[:, :3])

        if is_rgb and use_rgb:
            rgb_npy = cloud_array['rgb']
            rgb_npy.dtype = np.uint32
            r = np.asarray((rgb_npy >> 16) & 255, dtype=np.uint8)
            g = np.asarray((rgb_npy >> 8) & 255, dtype=np.uint8)
            b = np.asarray(rgb_npy & 255, dtype=np.uint8)
            rgb_npy = np.asarray([r, g, b])
            rgb_npy = rgb_npy.astype(np.float)/255
            rgb_npy = np.swapaxes(rgb_npy, 0, 1)
            o3dpc.colors = o3d.utility.Vector3dVector(rgb_npy)
        return o3dpc

    def filter_points(self,pcd, z_lowest=0.01):
        pointcloud_as_array = np.asarray(pcd.points)
        X = 0
        Y = 0
        # Z = 0.27
        # Z = 0.36
        Z = 0.48
        d = 0.15
        final_pointcloud_array = []
        for point in pointcloud_as_array:
            if point[2] < Z and X - d < point[0] < X + d and Y - d < point[1] < Y + d:
                final_pointcloud_array.append(point)

        # Create Open3D point cloud object from array
        final_pointcloud = o3d.geometry.PointCloud()
        final_pointcloud.points = o3d.utility.Vector3dVector(final_pointcloud_array)

        # o3d.visualization.draw_geometries([final_pointcloud])

        plane_model, inliers = final_pointcloud.segment_plane(distance_threshold=0.005,
                                            ransac_n=5,
                                            num_iterations=200)
        # inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = final_pointcloud.select_by_index(inliers, invert=True)


        return outlier_cloud

    def pc_callback(self,ros_cloud):
        self.pc_time = ros_cloud.header.stamp
        self.o3d_pc=self.rospc_to_o3dpc(ros_cloud, remove_nans=True, use_rgb=False)

    def draw_registration_result(self,source, target , transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def evaluateModel(self,target,robot_base2camera,now_time_):
        if target.is_empty():
            print('no data')
            return 0
        
        thread_id = threading.get_ident()
        lock.acquire()
        self.thread_id_list.append(thread_id)
        lock.release() 
        
        # if len(target.points)<1000:
        #     print("pointcloud too small")
        #     return 0
        
        # trans_init = np.array([[1.0, 0.0, 0.0, 0.0],
        # [0.0, 1.0, 0.0, 0.0],
        # [0.0, 0.0, 1.0, 0.0],
        # [0.0, 0.0, 0.0, 1.0]])
        # draw_registration_result(source, target, trans_init)
        threshold = 0.5
        voxel_size=0.005

        
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        # reg_p2p = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
                
        # reg_p2p_1 = o3d.pipelines.registration.registration_fast_based_on_feature_matching( # v0.12.0
        reg_p2p_1 = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(  # v0.14.1 later
            self.source_down, target_down, self.source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=0.5,iteration_number=20))
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
                    self.source, target, threshold, reg_p2p_1.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())

        
        final_translation = reg_p2p.transformation
        
        # r_f = Ra.from_matrix([[final_translation[0][0], final_translation[0][1], final_translation[0][2]],
        #                     [final_translation[1][0], final_translation[1][1], final_translation[1][2]],
        #                     [final_translation[2][0], final_translation[2][1], final_translation[2][2]]])
        
        r_f = Ra.from_matrix(final_translation[0:3,0:3])
        r_f_q = r_f.as_quat()

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = now_time_
        t.header.frame_id = "rgb_camera_link"
        t.transform.translation.x = final_translation[0][3]+0.02
        t.transform.translation.y = final_translation[1][3]+0.02
        t.transform.translation.z = final_translation[2][3]+0.02

        t.transform.rotation.x = r_f_q[0]
        t.transform.rotation.y = r_f_q[1]
        t.transform.rotation.z = r_f_q[2]
        t.transform.rotation.w = r_f_q[3]
        t.child_frame_id = "aaaaaaaaa"
        self.br.sendTransform(t)
        
        camera_2_Obj_r = final_translation[0:3,0:3]
        camera_2_Obj_t = final_translation[0:3,3]+0.02 # model axis to center
        camera_2_Obj_TF = np.r_[np.c_[np.array(camera_2_Obj_r),np.array(camera_2_Obj_t).reshape(3,1)],np.array([0,0,0,1]).reshape(1,4)]
        rabot_base2Obj = np.matmul(robot_base2camera, camera_2_Obj_TF)
        
        
        r_f = Ra.from_matrix(rabot_base2Obj[0:3,0:3])
        r_f_q = r_f.as_quat()
        
        
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = now_time_
        # t.header.frame_id = "depth_camera_link"
        t.header.frame_id = "base"
        t.transform.translation.x = rabot_base2Obj[0][3]
        t.transform.translation.y = rabot_base2Obj[1][3]
        t.transform.translation.z = rabot_base2Obj[2][3]

        t.transform.rotation.x = r_f_q[0]
        t.transform.rotation.y = r_f_q[1]
        t.transform.rotation.z = r_f_q[2]
        t.transform.rotation.w = r_f_q[3]
        t.child_frame_id = "lllllllllllllllllllllll"
        self.br.sendTransform(t)

        
        pub_data = Odometry()
        pub_data.header.stamp=now_time_
        pub_data.header.frame_id = "base"
        pub_data.child_frame_id = "lllllllllllllllllllllll"
        pub_data.pose.pose.position.x=rabot_base2Obj[0][3]
        pub_data.pose.pose.position.y=rabot_base2Obj[1][3]
        pub_data.pose.pose.position.z=rabot_base2Obj[2][3]
        pub_data.pose.pose.orientation.x=r_f_q[0]
        pub_data.pose.pose.orientation.y=r_f_q[1]
        pub_data.pose.pose.orientation.z=r_f_q[2]
        pub_data.pose.pose.orientation.w=r_f_q[3]

         
        while thread_id != self.thread_id_list[0]:
            pass

        lock.acquire()
        self.thread_id_list.pop(0)
        lock.release()  
        self.pub_armbase2Obj.publish(pub_data)   
        
    def go_espnet(self):
        while(not rospy.is_shutdown()):
            
            # start_all = timeit.default_timer()
               
            if data_lock.acquire():
                if self.start !=1:
                    rospy.sleep(0.001)
                    data_lock.release()
                    continue
                self.start = 0
                now_time = copy.copy(self.rgb_time)
                imgC = copy.deepcopy(self.img_)
                imgD = copy.deepcopy(self.depth_image)
                copy_robot_base2camera = np.array(self.robot_base2camera)
                data_lock.release()
            
            img_d = copy.deepcopy(imgD)    
            img_c = imgC
            img = imgC.astype(np.float32)
            
            mean = [112.985016, 93.449036, 106.11367]
            std = [57.782948, 51.31353, 54.971416]

            args = self.args
            model = self.modelA
            
            num_points = args.num_points

            
            for j in range(3):
                img[:, :, j] -= mean[j]
            for j in range(3):
                img[:, :, j] /= std[j]

            img /= 255
            img = img.transpose((2, 0, 1))
            img_tensor = torch.from_numpy(img)
            img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
            img_tensor = img_tensor.cuda()

            img_out = model(img_tensor)
            classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
            
            classMap_numpy = cv2.resize(classMap_numpy, (1280, 720), interpolation=cv2.INTER_NEAREST)
            label_img = measure.label(classMap_numpy)

            # classMap_numpy_3 = classMap_numpy.copy()
            # classMap_numpy_3[classMap_numpy_3==0] = 0
            # classMap_numpy_3[classMap_numpy_3==1] = 150
            # cv2.imwrite('./output1.png', classMap_numpy_3)
            
            
            obj_box = [None, None, None, None]
            obj_number = -1
            for region in measure.regionprops(label_img):
                # print("region.area, ",region.area)
                if region.area < 1000: #5000 #8000
                    # print("too small")
                    continue
                else:                    
                    ## preprocessing
                    if obj_number == -1:
                        obj_box = [region.bbox[0],region.bbox[1],region.bbox[2],region.bbox[3]]
                        obj_number = classMap_numpy[region.coords[0,0],region.coords[0,1]]
                        classMap_numpy[label_img==(label_img[region.coords[0,0],region.coords[0,1]])] = 255      
                    else:
                        classMap_numpy[label_img==(label_img[region.coords[0,0],region.coords[0,1]])] = 255 # suppose only one object is detected
                        if obj_box[0] > region.bbox[0]:
                            obj_box[0] = region.bbox[0]
                        if obj_box[1] > region.bbox[1]:
                            obj_box[1] = region.bbox[1]
                        if obj_box[2] < region.bbox[2]:
                            obj_box[2] = region.bbox[2]
                        if obj_box[3] < region.bbox[3]:
                            obj_box[3] = region.bbox[3]
            
            if obj_number==-1:
                continue
            else:
                testdataset = PoseDataset_semantics('eval', num_points, False, img_c, img_d, classMap_numpy, obj_box, 0.0, True)   
                points= testdataset.__getitem__(obj_number)
            
            if len(points)!=self.args.num_points:
                continue
            targetxx = o3d.geometry.PointCloud()
            targetxx.points = o3d.utility.Vector3dVector(points)

            # print('Time esp: ', timeit.default_timer()-start_all)

            # target.paint_uniform_color([1, 0.706, 0])
            # targetxx.paint_uniform_color([0, 0.651, 0.929])
            # o3d.visualization.draw_geometries([target, targetxx])
            # o3d.visualization.draw_geometries([targetxx])
            
            # self.evaluateModel(targetxx,copy_robot_base2camera,now_time)
            processThread = threading.Thread(target=self.evaluateModel, args=(targetxx,copy_robot_base2camera,now_time,))
            
            
            processThread.start()
            
    def go_icp(self):
        while(not rospy.is_shutdown()):
            if self.start !=1:
                # print("fiail")
                rospy.sleep(0.1)
                continue                
            if data_lock.acquire():
                now_time = copy.copy(self.pc_time)
                # print("self.o3d_pc empty",self.o3d_pc.is_empty())
                Temp = copy.deepcopy(self.o3d_pc)
                copy_robot_base2camera = np.array(self.robot_base2camera)  
                data_lock.release()
                
            temp = Temp.voxel_down_sample(voxel_size=0.005)
            # o3d.visualization.draw_geometries([temp])
            target = self.filter_points(temp)
            # o3d.visualization.draw_geometries([target])

            processThread = threading.Thread(target=self.evaluateModel, args=(target,copy_robot_base2camera,now_time,))
            processThread.start()


def thread_job():
    while(not rospy.is_shutdown()):
        rospy.spin()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNetv2", help='Model name')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--pretrained', default='./model_results/model_best_auo.pth', help='Pretrained weights directory.')
    parser.add_argument('--s', default=1, type=float, help='scale')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset. 20 for Cityscapes')
    parser.add_argument('--num_points', type=int, default = '10000',  help='resume PoseRefineNet model')
    
    parser.add_argument('--mode', type=str, default = 'espnet',  help='Pointcloud or espnet')

    args = parser.parse_args()
    rospy.init_node('semantics', anonymous=True)

    add_thread = threading.Thread(target = thread_job)
    add_thread.start()

    K = detector(args)
    
    
    
    if args.mode == 'Pointcloud':        
        K.go_icp()
    elif args.mode == 'espnet':
        K.go_espnet()
        
