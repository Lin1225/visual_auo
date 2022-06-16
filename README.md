# vs_grasp
# AUO User guide

## install environment
- Ubuntu 20.04, ROS noetic, python 3.8

### python package intall
```
    pip3 install pytorch scipy skimage open3d numpy
```

### ros package git clone
```
git clone https://github.com/Lin1225/visual_auo.git
```

### camera calibration
- use 5*7 chess board  
執行步驟
```
roscore
roslaunch azure_kinect_ros_driver driver.launch
rosrun camera_calibration cameracalibrator.py --size 7x5 --square 0.03 image:=/rgb/image_raw
```
- 用手拿著棋盤格紙在相機可見範圍內移動
- 改變棋盤格的位置、和距離距離、傾斜角度，讓gui介面上的x y size skew四個都變成綠色後，按下calibrate開始計算參數
- 最後輸出 K, D, R, P vector
#### modify camera intrinsic parameter in estimation python code
- 將cx, cy, fx, fy輸入至`4.with_ICP/datasets/dataset_Class.py`的`cam_cx, cam_cy, cam_fx, cam_fy`  
- 完成相機校正

### hand eye calibration
- 準備 april_tag 350 size :  0.076* 0.076 或是修改`cali_0329.launch`內的參數值
#### 注意 
1. 因為手眼校正需要機械手臂末端點的位置，因此需要有一個從手臂基座標到末端點的tf
2. modify `robot_base_frame` & `robot_effector_frame` in hiwin_realsense_calibration.launch
3. modify `self_camera_info.py` in camera_calibrate folder
4. modify `camera_info.K`, `camera_info.D`, `camera_info.R`, `camera_info.P`

執行步驟
```
roscore
roslaunch azure_kinect_ros_driver driver.launch
python src/camera_calibrate/self_camera_info.py
ROS_NAMESPACE=my_camera rosrun image_proc image_proc
roslaunch apriltag_ros cali_0329.launch
roslaunch easy_handeye hiwin_realsense_calibration.launch  
```
#### 校正方式
- 將相機固定在機械手臂上
- 固定apriltag
- 移動機械手臂，讓apriltag盡量保持在畫面中心，並且手臂各關節要盡量移動
- 每次移動完手臂都按下take sample
- 多筆之後可以按下caculate看目前的結果，若結果開始收斂則結束。一般取20筆左右
- 將最後的GUI顯示的結果`(translation, rotaion)`紀錄下來, 將剛剛得到的值輸入並執行
```
rosrun tf static_transform_publisher x y z rx ry rz w /flange_link /camera_base 100
```

#### modify hand eye transform in estimation python code
##### Use rviz to check the transform
- 在rviz內,將fixed frame改成手臂法蘭面的tf
- 點開camera_base的tf 
- 將position和orientation輸入至`4.with_ICP/pcicp_class.py`的第97行`(end_2_camera_t)`和第98行`(end_2_camera_r)`
- 完成手眼校正

### 開始執行程式
```
roscore
roslaunch azure_kinect_ros_driver driver.launch
開啟robot driver
rosrun tf static_transform_publisher x y z rx ry rz w /flange_link /camera_base 100
python src/4.with_ICP/pcicp_class.py
rosrun vs_demo control_final.cpp
```
#### 注意 
- 目前訂閱的topic內容包含
    1. 手臂末端點的位置 ArmPosCallback
    2. 各軸移動的角速度 ArmPosCallback
    3. 夾爪位置       gripperCallback
- 輸出包含
    1. 夾爪移動位置    set_gripper
    2. 手臂移動速度（卡式座標）arm_move_vel
    3. 手臂移動位置（卡式座標）和時間 arm_move_pos
- control_final 第34-41行為起點、放置點、以及其他參數, 345-367行為夾取姿態, 77-130行為夾爪控制,必須修改
- 第45-47行為可調整的參數



