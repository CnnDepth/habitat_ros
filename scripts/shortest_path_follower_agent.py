import rospy
import numpy as np
import habitat
import tf
import cv2
import math
import keyboard
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from habitat.utils.visualizations import maps
from skimage.io import imsave


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))
    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


class ShortestPathFollowerAgent(habitat.Agent):

    def __init__(self, env, goal_radius):
        self.follower = ShortestPathFollower(env.sim, goal_radius, False)

        # initialize ROS publishers and subscribers
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_subscriber = rospy.Subscriber('/exploration_goal', PoseStamped, self.goal_callback)
        self.tf_listener = tf.TransformListener()
        self.habitat_goal_publisher = rospy.Publisher('/habitat_goal', PoseStamped, queue_size=100, latch=True)

        # initialize poses
        self.robot_pose_in_slam_coords = None
        self.robot_pose_in_habitat_coords = None
        self.goal_pose_in_slam_coords = None
        self.goal_pose_in_habitat_coords = None
        self.env = env
        env.reset()
        self.update_time = rospy.Time.now()
        self.topdown_saved = False


    def normalize(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


    def odom_callback(self, msg):
        self.robot_pose_in_slam_coords = msg.pose.pose


    def get_robot_pose(self):
        cur_pose = self.robot_pose_in_slam_coords
        try:
            pos, quat = self.tf_listener.lookupTransform(
                                'map', 'odom',
                                self.tf_listener.getLatestCommonTime('map',
                                'odom'))
        except:
            print('NO TRANSFORM FROM ODOM TO MAP!!!!')
            return None, None, None
        if self.robot_pose_in_slam_coords is None:
            print('NO ODOMETRY!!!')
            return None, None, None
        _, __, tf_angle = tf.transformations.euler_from_quaternion(quat)
        _, __, odom_angle = tf.transformations.euler_from_quaternion([cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w])
        print('Euler angles:', _, __, odom_angle)
        current_x, current_y = cur_pose.position.x, cur_pose.position.y
        current_x_new = current_x * math.cos(-tf_angle) + current_y * math.sin(-tf_angle)
        current_y_new = -current_x * math.sin(-tf_angle) + current_y * math.cos(-tf_angle)
        current_x_new += pos[0]
        current_y_new += pos[1]
        return current_x_new, current_y_new, odom_angle + tf_angle


    def goal_callback(self, msg):
        # Receive goal pose in SLAM coords
        print('Received goal with coords: {}, {}'.format(msg.pose.position.x, msg.pose.position.y))
        self.goal_pose_in_slam_coords = msg.pose
        goal_x, goal_y = msg.pose.position.x, msg.pose.position.y

        # Find robot's position and orientation in SLAM and Habitat coords
        slam_x, slam_y, slam_angle = self.get_robot_pose()
        habitat_position, habitat_orientation = self.robot_pose_in_habitat_coords
        #print('Robot pose in habitat coords:', habitat_position, habitat_orientation)
        habitat_y, habitat_z, habitat_x = habitat_position
        _, __, habitat_angle = tf.transformations.euler_from_quaternion([habitat_orientation.x, habitat_orientation.z, habitat_orientation.y, habitat_orientation.w])

        # Calculate transform between SLAM and Habitat coordinate systems
        d_angle = self.normalize(habitat_angle - slam_angle + np.pi)
        #print('D_ANGLE:', d_angle)
        dx = habitat_x - (slam_x * math.cos(d_angle) + slam_y * math.sin(d_angle))
        dy = habitat_y - (-slam_x * math.sin(d_angle) + slam_y * math.cos(d_angle))

        # Compute goal position in Habitat coords
        goal_x_rotated = goal_x * math.cos(d_angle) + goal_y * math.sin(d_angle)
        goal_y_rotated = -goal_x * math.sin(d_angle) + goal_y * math.cos(d_angle)
        self.goal_pose_in_habitat_coords = np.array([goal_y_rotated + dy, habitat_z, goal_x_rotated + dx])
        #print('GOAL COORDS IN HABITAT SYSTEM:', self.goal_pose_in_habitat_coords)

        # publish goal position in Habitat coords
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.position.x = self.goal_pose_in_habitat_coords[2]
        msg.pose.position.y = self.goal_pose_in_habitat_coords[0]
        msg.pose.position.z = self.goal_pose_in_habitat_coords[1]
        self.habitat_goal_publisher.publish(msg)


    def reset(self):
        pass


    def act(self, observations, env):
        info = env.get_metrics()
        if info['top_down_map'] is not None and not self.topdown_saved and (rospy.Time.now() - self.update_time).to_sec() > 1:
            topdown_map = draw_top_down_map(info, observations['heading'][0], observations['rgb'][0].shape[0])
            imsave('/home/kirill/topdown_map.png', topdown_map)
            self.update_time = rospy.Time.now()

            print('TOPDOWN_MAP:', topdown_map)
        self.robot_pose_in_habitat_coords = observations['agent_position']
        if keyboard.is_pressed('left'):
            return HabitatSimActions.TURN_LEFT
        elif keyboard.is_pressed('right'):
            return HabitatSimActions.TURN_RIGHT
        elif keyboard.is_pressed('up'):
            return HabitatSimActions.MOVE_FORWARD
        elif self.goal_pose_in_habitat_coords is None:
            return HabitatSimActions.STOP
        else:
            next_action = self.follower.get_next_action(self.goal_pose_in_habitat_coords)
            print(next_action)
            if next_action is None:
                print('CANNOT MOVE TO GOAL!!!')
                return HabitatSimActions.STOP
            return next_action