#include <cstdlib>
#include "ros/ros.h"
#include <ros/init.h>


#include "std_msgs/Float32MultiArray.h"
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Int8.h"
#include "std_msgs/Float32.h"

#include "std_msgs/Header.h"

#include <math.h>
#include <time.h>
#include <algorithm>
#include <stdio.h> 
#include <termios.h> 
#include <unistd.h> 
#include <fcntl.h>
#include <vector>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include "robotiq_controller/Robotiq2FGripper_robot_output.h"
#include "robotiq_controller/Robotiq2FGripper_robot_input.h"
#include "tm_msgs/FeedbackState.h"


#define to_arc 0.01745329
#define pi 3.14159265

float home[6]={0.1,-0.5,0.51,3.1416,0,0};  // initial robot arm pose
float place[6]={0.3862,-0.007142,0.249196,3.1416,0,1.5708}; // place object arm pose

int vel_average_time = 5;           // the amount of object position is used when caculating object velocity
float lowest_Z = 0.2;               // flange z height when picking
float picking_Roll = 3.14;          // flange roll when picking
float picking_Yaw = 0.0;            // flange yaw when picking
float tracking_Z = 0.48967 ;        // flange z height when camera is 28cm away from object
float flange2camera_y = 0.14;       // the y axis distance from flange to camera
float move_second_open_loop = 1.8;  // robot arm motion time to grasp position
float grasp_close_time = 0.4257;    // operation time of gripper from open to grasp
float tracking_time = 1.3 ;         // time to get to the best view position when tracking object


int start_cal_vel = 0;              // object speed calculate times
int can_grasp = 0;                  // 1: already tracked the object and can grasp it 
int grasp_po;                       // gripper position, get it from gripperCallback
float memory_go[6]={0};             // object position, get from obj_callback
float object_velocity[3]={0};       // object velocity, get from obj_callback
float arm_now_vel[6]={0};           // robot arm velocity, get from ArmPosCallback
float arm_speed_sum=0;              // sum of robot arm joint speed
tf::Transform transform;            // robot arm position, get from ArmPosCallback
tf::Transform rabot_end2obj;        // object 6d position, get from obj_callback
ros::Time now_time,last_time,start_move,move_time,grasp_time;


robotiq_controller::Robotiq2FGripper_robot_output gripper_command;
ros::Subscriber robot_status_sub;
ros::Subscriber gripper_sub;
ros::Subscriber transform_sub;
ros::Publisher  arm_pos_cmd_pub;
ros::Publisher  arm_vel_cmd_pub;
ros::Publisher  gripper_ctrl;

typedef struct vector_data
{
  float obj_pos_memory[3];
  ros::Time obj_time;
}vector_data;

vector_data est_obj_info;
std::vector<vector_data> est_obj_vec;
std::vector<vector_data>::iterator obj_it;


void set_gripper(int mode)
{   
  /**
  *  gripper close or open control function
  */
  grasp_time = ros::Time::now();
  if(mode == 1)   //Open
  {
      gripper_command.rACT = 1;
      gripper_command.rGTO = 1;
      gripper_command.rSP  = 200;
      gripper_command.rFR  = 0;
      gripper_command.rPR = 0;
      gripper_ctrl.publish(gripper_command);
  }
  else if(mode == 0)   //Close
  {
      gripper_command.rACT = 1;
      gripper_command.rGTO = 1;
      gripper_command.rSP  = 200;
      gripper_command.rFR  = 0;
      gripper_command.rPR = 120;
      gripper_ctrl.publish(gripper_command);
  }
  // printf("wait grasp\n");
  while (ros::ok() && (abs(grasp_po-gripper_command.rPR)>5)){
    // printf("pos:%d\n",abs(grasp_po-gripper_command.rPR));
  }
  // sleep(0.5);
  printf("grasp move time pass :%f\n",ros::Time::now().toSec() - grasp_time.toSec());
  
}
void init_gripper()
{
  /*
    gripper initialization function
  */
  gripper_command.rACT = 1;
  gripper_command.rGTO = 1;
  gripper_command.rSP  = 200;
  gripper_command.rFR  = 0;
  gripper_ctrl.publish(gripper_command);
}
void reset_gripper()
{
  /*
    gripper reset function
  */
  gripper_command.rACT = 0;
  gripper_command.rGTO = 0;
  gripper_command.rSP  = 0;
  gripper_command.rFR  = 0;  
  gripper_ctrl.publish(gripper_command); 
}
int kbhit(void) 
{ 
  /*
    wait until keyboard press function
    reference : https://blog.csdn.net/lanmanck/article/details/5823562
  */
  struct termios oldt, newt; 
  int ch; 
  int oldf; 
  tcgetattr(STDIN_FILENO, &oldt); 
  newt = oldt; 
  newt.c_lflag &= ~(ICANON | ECHO); 
  tcsetattr(STDIN_FILENO, TCSANOW, &newt); 
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0); 
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK); 
  ch = getchar(); 
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt); 
  fcntl(STDIN_FILENO, F_SETFL, oldf); 
  if(ch != EOF) { 
      ungetc(ch, stdin); 
      return 1; 
  } 
  return 0; 
} 

/*
  read arm now info
  var :  
    arm_now_vel : save the speed of the robot flange in the cartesian space
    arm_speed_sum : accumulate joint speed
    transform : save arm pose in the cartesian space
*/
void ArmPosCallback(const tm_msgs::FeedbackState::ConstPtr& msg)
{
  transform.setOrigin( tf::Vector3(msg->tool_pose[0], msg->tool_pose[1], msg->tool_pose[2]) );
  tf::Quaternion q;
  q.setRPY(msg->tool_pose[3]*to_arc, msg->tool_pose[4]*to_arc, msg->tool_pose[5]*to_arc);
  transform.setRotation(q);
  arm_speed_sum = 0;
  for(int i=0;i<6;i++){
    arm_now_vel[i] = msg->tcp_speed[i];
    arm_speed_sum += abs(msg->joint_vel[i]);
  }
  
}

//check gripper status
void gripperCallback(const robotiq_controller::Robotiq2FGripper_robot_input::ConstPtr& msg){
  grasp_po = msg->gPO;
}


/*
  caluate the velocity of the object
*/
void transformCallback_modi_vel_v2(const nav_msgs::Odometry &msg){
  if(start_cal_vel!=0){

    rabot_end2obj.setOrigin(tf::Vector3( msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z));
    rabot_end2obj.setRotation(tf::Quaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w));

    //memory_go is the newest object pose
    memory_go[0] =  rabot_end2obj.getOrigin().x() ;
    memory_go[1] =  rabot_end2obj.getOrigin().y() ;
    memory_go[2] =  rabot_end2obj.getOrigin().z() ;

    tf::Matrix3x3 m(rabot_end2obj.getRotation());
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    memory_go[3] =  roll;
    memory_go[4] =  pitch;
    memory_go[5] =  yaw;
    last_time = now_time;
    now_time = msg.header.stamp;
  
    if((now_time.toSec()-last_time.toSec())<=0){
        printf("BBBBBBBBBBBBBe careful\n");
        return ;
    }
    if(start_cal_vel==-1){
      return;
    }
    else if(start_cal_vel == 1){
      ROS_INFO("start_cal_vel == 1");
      est_obj_info.obj_time = now_time;
      for (int i = 0; i < 3; i++)
      {
        est_obj_info.obj_pos_memory[i]=memory_go[i];
      }      
      est_obj_vec.push_back(est_obj_info);
      start_cal_vel = 2;
    }
    else if(start_cal_vel <= vel_average_time){
      est_obj_info.obj_time = now_time;
      for(int i=0;i<3;i++){        
        est_obj_info.obj_pos_memory[i]=memory_go[i];      
      }
      est_obj_vec.push_back(est_obj_info);

      int cal_time = 0;
      float temp_vel[3]={0.0};
      std::vector<vector_data> temp_vec(est_obj_vec.begin(),est_obj_vec.end());
      temp_vec.erase(temp_vec.begin());
      std::vector<vector_data>::iterator temp_it;

      for (obj_it=est_obj_vec.begin();obj_it!=est_obj_vec.end();obj_it++){
        if(temp_vec.empty()){
          break;
        }
        for(temp_it=temp_vec.begin();temp_it!=temp_vec.end();temp_it++){
          for (int i = 0; i < 3; i++)
          {
            temp_vel[i] += ((*temp_it).obj_pos_memory[i]-(*obj_it).obj_pos_memory[i])
                                /((*temp_it).obj_time.toSec()-(*obj_it).obj_time.toSec());
          }
          cal_time++;
        }
        temp_vec.erase(temp_vec.begin());        
      }
      for (int i = 0; i < 3; i++)
      {
        object_velocity[i]=temp_vel[i]/cal_time;
      }
      start_cal_vel += 1;
    }
    else if(start_cal_vel > vel_average_time){
      start_cal_vel++;
      est_obj_info.obj_time = now_time;
      for(int i=0;i<3;i++){        
        est_obj_info.obj_pos_memory[i]=memory_go[i];      
      }
      est_obj_vec.push_back(est_obj_info);

      est_obj_vec.erase(est_obj_vec.begin());
      
      int cal_time = 0;
      float temp_vel[3]={0.0};
      std::vector<vector_data> temp_vec(est_obj_vec.begin(),est_obj_vec.end());
      temp_vec.erase(temp_vec.begin());
      std::vector<vector_data>::iterator temp_it;

      for (obj_it=est_obj_vec.begin();obj_it!=est_obj_vec.end();obj_it++){
        if(temp_vec.empty()){
          break;
        }
        for(temp_it=temp_vec.begin();temp_it!=temp_vec.end();temp_it++){
          for (int i = 0; i < 3; i++)
          {
            temp_vel[i] += ((*temp_it).obj_pos_memory[i]-(*obj_it).obj_pos_memory[i])
                                /((*temp_it).obj_time.toSec()-(*obj_it).obj_time.toSec());
          }
          cal_time++;
        }
        temp_vec.erase(temp_vec.begin());        
      }
      for (int i = 0; i < 3; i++)
      {
        object_velocity[i]=temp_vel[i]/cal_time;
      }
    }
    printf("callback velocity x:%2f ; y:%2f ; z:%2f\n",object_velocity[0],object_velocity[1],object_velocity[2]);
    
  } 
}

/*
  PTP publisher 
  input :
    x, y, z, roll, pitch, yaw : end flange 6D pose in cartesian space
    time : the time robot arm reached the  6D pose goal
*/ 
void arm_move_pos(float x,float y, float z,float roll,float pitch,float yaw,float time){
    std_msgs::Float32MultiArray arm_new_pos;
    float value[7];
    arm_new_pos.data.push_back(x);
    arm_new_pos.data.push_back(y);
    arm_new_pos.data.push_back(z);
    arm_new_pos.data.push_back(roll);
    arm_new_pos.data.push_back(pitch);
    arm_new_pos.data.push_back(yaw);
    arm_new_pos.data.push_back(time);
    arm_pos_cmd_pub.publish(arm_new_pos);
    ROS_INFO("ALL arm move time pass :%f",ros::Time::now().toSec() - start_move.toSec());
    move_time =  ros::Time::now();
    while (ros::ok() && ((arm_speed_sum)>5*1e-5 || (abs(transform.getOrigin().x() - x)>0.01) || (abs(transform.getOrigin().y() - y)>0.01) || (abs(transform.getOrigin().z() - z)>0.01)));
    arm_new_pos.data.clear();
}

/*
  cartesian space speed publisher
  input :
    x, y, z, roll, pitch, yaw : end flange linear speed & angular velocity in cartesian space
*/
void arm_move_vel(float x,float y, float z,float roll,float pitch,float yaw){
  geometry_msgs::Twist arm_new_vel;
  arm_new_vel.linear.x = x;
  arm_new_vel.linear.y = y;
  arm_new_vel.linear.z = z;
  arm_new_vel.angular.x = roll;
  arm_new_vel.angular.y = pitch;
  arm_new_vel.angular.z = yaw;
  if(start_cal_vel>0)
    arm_vel_cmd_pub.publish(arm_new_vel);
}

/*
  open loop grasp 
  input :
    now object pose in cartesian space
  output :
    arm moving pose in cartesian space
*/
int choose_waypoint_pos(float memory[6]){ //open loop to grasp
    //target_postion
    
    float image_deal_time = ros::Time::now().toSec() - now_time.toSec();
    memory[0] = memory_go[0] + (move_second_open_loop + image_deal_time + grasp_close_time) * (object_velocity[0]);
    printf("pass time = %f\n",image_deal_time);
    // memory[1] = memory_go[1] ;
    memory[1] = memory_go[1] + (move_second_open_loop + image_deal_time + grasp_close_time) * (object_velocity[1]);
    memory[2] = lowest_Z; 
    memory[3] = picking_Roll;
    memory[4] = picking_Yaw;
    
    // keep in range -3.14~3.14
    if(memory_go[5] < -pi)
      memory[5] = memory_go[5] + 2*pi;
    else if(memory_go[5] > pi)
      memory[5] = memory_go[5] - 2*pi;

    // keep in range -0.785~-1.785  , 0.785~1.785 
    if(memory_go[5] <-2.35)           //-180 ~ -135 degree
      memory[5] = memory_go[5] + pi/2;
    else if(memory_go[5] <-1.57)      //-135 ~  -90 degree
      memory[5] = memory_go[5] + pi;
    else if(memory_go[5] < 0.785)     // -90 ~  -45 degree
      memory[5] = memory_go[5];
    else if(memory_go[5] < 0)         // -45 ~    0 degree
      memory[5] = memory_go[5] + pi/2;
    else if(memory_go[5] <-0.785)     //   0 ~   45 degree
      memory[5] = memory_go[5] - pi/2;
    else if(memory_go[5] <1.57)       //  45 ~   90 degree
      memory[5] = memory_go[5];
    else if(memory_go[5] <2.35)       //  90 ~  135 degree
       memory[5] = memory_go[5] -pi;
    else                              // 135 ~  180 degree
       memory[5] = memory_go[5] -pi/2;

    
    printf("go to target\n");
    printf("x:%f ,y:%f, z:%f\n",memory[0],memory[1],memory[2]);
    printf("rx:%f ,ry:%f, rz:%f\n",memory[3],memory[4],memory[5]);
    printf("velocity x:%2f ; y:%2f ; z:%2f\n\n",object_velocity[0],object_velocity[1],object_velocity[2]);
    start_move = ros::Time::now();
    return 1;
}

/*
  close loop visual servoing
  input :
    now object pose in cartesian space
  output :
    arm moving speed in cartesian space
*/
int choose_waypoint_vel(float memory[6]){ //vision sevoing close loop    
  if(can_grasp == 0){
    float temp[6] = {0};

    //now arm x, y, z
    temp[0]=transform.getOrigin().x();
    temp[1]=transform.getOrigin().y();
    temp[2]=transform.getOrigin().z();
    
    float image_deal_time = ros::Time::now().toSec() - now_time.toSec();
    
    float p_error_x = memory[0] - temp[0];
    float p_error_y = memory[1] - (temp[1]-flange2camera_y); //grasp to camera y 
    float p_error_z = tracking_Z - temp[2];

    float distance_xy = sqrt(p_error_x*p_error_x+p_error_y*p_error_y);
    printf("distance_xy = %f\n",distance_xy);
    if(distance_xy<(0.05+object_velocity[0]*0.2) && start_cal_vel>15){
      printf("reach xy, grasp!!!\n ");
      can_grasp = 1;
      return 1;
    }
    
    memory[0] = (p_error_x + image_deal_time*object_velocity[0])/tracking_time + object_velocity[0];
    memory[1] = (p_error_y + image_deal_time*object_velocity[1])/tracking_time + object_velocity[1];
    memory[2] = (p_error_z + image_deal_time*object_velocity[2])/tracking_time + object_velocity[2];
    
    
    for(int i=0;i<3;i++){
      if(abs(memory[i])<0.01){
        memory[i]=0;
      }
      else if(abs(memory[i])>0.1){
        if (memory[i]<0){
          memory[i] = -0.1;
        }else{
          memory[i] = 0.1;
        }
      }
    }
    // printf("x:%f ,y:%f, z:%f\n",memory[0],memory[1],memory[2]);
    printf("velocity x:%2f ; y:%2f ; z:%2f\n\n",object_velocity[0],object_velocity[1],object_velocity[2]);
    
    memory[3] = 0.0;
    memory[4] = 0.0;
    memory[5] = 0.0;
    
    start_move = ros::Time::now();
    return 0;
  }
  else {
    return 0;
  }
}

/*
  smooth the arm moving speed in cartesian space
  input :
    arm moving speed in cartesian space
  output :
    arm moving speed in cartesian space
*/
void velocity_smooth(float memory[6]){
  float p_error[6] = {0};
  for (int i=0;i<3;i++){
    p_error[i] = memory[i] - arm_now_vel[i];
    memory[i] = arm_now_vel[i] + p_error[i]*0.3;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "visual_tracking_grasp");
  ros::NodeHandle n;
  ros::AsyncSpinner spinner(3);
  robot_status_sub = n.subscribe("feedback_states", 1, ArmPosCallback);
  gripper_sub = n.subscribe("Robotiq2FGripperRobotInput", 10,gripperCallback);
  transform_sub = n.subscribe("my_den_armbase2Obj", 1, transformCallback_modi_vel_v2);
  arm_pos_cmd_pub = n.advertise<std_msgs::Float32MultiArray>("arm_pos_cmd",10);
  arm_vel_cmd_pub = n.advertise<geometry_msgs::Twist>("arm_vel_cmd",10);
  gripper_ctrl = n.advertise<robotiq_controller::Robotiq2FGripper_robot_output>("Robotiq2FGripperRobotOutput",1);
 

  start_cal_vel=0;
  spinner.start();
  printf("sleep 1 s\n");
  sleep(1);

  printf("set_gripper\n");
  reset_gripper();
  sleep(1);

  init_gripper();
  sleep(1);

  printf("open\n");
  set_gripper(1);
  
  printf("arm_move\n");
  arm_move_pos(home[0],home[1],home[2],home[3],home[4],home[5],5);
  
  for (int i = 0; i < 10 ; i++){ // grasp 10 times 
    
    // initialization
    float memory[6]={};
    can_grasp = 0;

    for (int i=0;i<3;i++){
      object_velocity[i] = 0;
    }
    est_obj_vec.clear();
    printf("--------------------------------------------------------------------\n");
    printf("enter 2 to cal\n");

    while (1){
      while(!kbhit());
      if ((getchar()) == '2'){
        start_cal_vel = 1 ;
        printf("start\n");
        break;
      }
    }

    int k = 0;
    ros::Rate r(90);
    while (k==0){ // close_loop
      for (int i=0;i<6;i++){
        memory[i] = memory_go[i];
      }      
      k = choose_waypoint_vel(memory);
      if(k!=0){
        start_cal_vel = 0 ;
        ROS_INFO("end close loop");
      }
      else{
        velocity_smooth(memory);
        arm_move_vel(memory[0],memory[1],memory[2],memory[3],memory[4],memory[5]);
        r.sleep();
      }      
    }
    
    // sleep(1.5);
    ROS_INFO("getting down");
    start_cal_vel = 0 ;
    
    choose_waypoint_pos(memory);
    arm_move_pos(memory[0],memory[1],memory[2],memory[3],memory[4],memory[5],move_second_open_loop);
        
    while(ros::Time::now().toSec() - start_move.toSec() < (move_second_open_loop));

    ROS_INFO("grasp close");
    set_gripper(0);

    ROS_INFO("go to home");    
    arm_move_pos(home[0],home[1],home[2],home[3],home[4],home[5],5);

    // ROS_INFO("to place");
    // arm_move_pos(place[0],place[1],place[2],place[3],place[4],place[5],8);
     
    // ROS_INFO("grasp open");
    // set_gripper(1);

    // ROS_INFO("go to home");
    // arm_move_pos(home[0],home[1],home[2],home[3],home[4],home[5],8);
    
    ROS_INFO("grasp open");
    sleep(1);
    set_gripper(1);
    
    
  } // end grasp loop

  return 0;
}

