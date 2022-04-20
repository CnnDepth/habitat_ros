import rospy
import sys
import tf
import numpy as np
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry

rospy.init_node('cmd_recorder')
path_to_save_cmd = sys.argv[1]
path_to_save_odom = sys.argv[2]
tf_listener = tf.TransformListener()
commands = []
odometry = []


def get_robot_pose(tf_listener, odom_pose):
    try:
        pos, quat = tf_listener.lookupTransform(
                            'map', 'odom',
                            tf_listener.getLatestCommonTime('map',
                            'odom'))
    except:
        print('NO TRANSFORM FROM ODOM TO MAP!!!!')
        return None, None, None
    _, __, tf_angle = tf.transformations.euler_from_quaternion(quat)
    _, __, odom_angle = tf.transformations.euler_from_quaternion([odom_pose.orientation.x, 
                                                                  odom_pose.orientation.y, 
                                                                  odom_pose.orientation.z, 
                                                                  odom_pose.orientation.w])
    current_x, current_y = odom_pose.position.x, odom_pose.position.y
    current_x_new = current_x * np.cos(-tf_angle) + current_y * np.sin(-tf_angle)
    current_y_new = -current_x * np.sin(-tf_angle) + current_y * np.cos(-tf_angle)
    current_x_new += pos[0]
    current_y_new += pos[1]
    return current_x_new, current_y_new, odom_angle + tf_angle


def cmd_callback(msg):
    cur_time = rospy.Time.now().to_sec()
    commands.append([msg.data, cur_time])


def odom_callback(msg):
    odom_pose = msg.pose.pose
    stamp = msg.header.stamp.to_sec()
    x, y, angle = get_robot_pose(tf_listener, odom_pose)
    if x is not None:
        odometry.append([x, y, angle, stamp])


cmd_subscriber = rospy.Subscriber('habitat_action', Int32, cmd_callback)
odom_subscriber = rospy.Subscriber('odom', Odometry, odom_callback)

rospy.spin()
np.savetxt(path_to_save_cmd, np.array(commands))
np.savetxt(path_to_save_odom, np.array(odometry))