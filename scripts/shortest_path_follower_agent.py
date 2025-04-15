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
from std_msgs.msg import Bool
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

    def __init__(self, env, goal_radius, goal_positions=None):
        self.follower = ShortestPathFollower(env.sim, goal_radius, False)
        self.goal_radius = goal_radius

        # initialize ROS publishers and subscribers
        self.goal_subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.freeze_subscriber = rospy.Subscriber('/freeze', Bool, self.freeze_callback)
        self.robot_pose_publisher = rospy.Publisher('/robot_pose_in_habitat_coords', PoseStamped, latch=True, queue_size=100)
        self.goal_received = False
        self.traveled_distance = 0

        # initialize poses
        self.robot_pose_in_slam_coords = None
        self.robot_pose_in_habitat_coords = None
        self.goal_pose_in_slam_coords = None
        self.goal_pose_in_habitat_coords = None
        self.env = env
        env.reset()
        self.update_time = rospy.Time.now()
        self.topdown_saved = False
        self.freeze = False

        if goal_positions is not None:
            self.goal_positions = goal_positions
        else:
            self.goal_positions = []
        self.goal_position_id = 0


    def normalize(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


    def get_robot_pose(self, observations):
        current_x, current_y = observations['gps']
        current_y = -current_y
        robot_angle = observations['compass'][0]
        current_x_new = current_x * math.cos(-robot_angle) + current_y * math.sin(-robot_angle)
        current_y_new = -current_x * math.sin(-robot_angle) + current_y * math.cos(-robot_angle)
        return current_x, current_y, robot_angle


    def goal_callback(self, msg):
        self.goal_received = True 
        # Receive goal pose in SLAM coords
        print('Received goal with coords: {}, {}'.format(msg.pose.position.x, msg.pose.position.y))
        self.goal_pose_in_slam_coords = msg.pose
        goal_x, goal_y = msg.pose.position.x, msg.pose.position.y

        # Find robot's position and orientation in SLAM and Habitat coords
        habitat_position, habitat_orientation = self.robot_pose_in_habitat_coords
        print('Robot pose in habitat coords:', habitat_position, habitat_orientation)
        habitat_y, habitat_z, habitat_x = habitat_position
        _, __, habitat_angle = tf.transformations.euler_from_quaternion([habitat_orientation.x, habitat_orientation.z, habitat_orientation.y, habitat_orientation.w])

        # Calculate transform between SLAM and Habitat coordinate systems
        d_angle = self.normalize(habitat_angle - self.slam_angle + np.pi)
        #print('D_ANGLE:', d_angle)
        dx = habitat_x - (self.slam_x * math.cos(d_angle) + self.slam_y * math.sin(d_angle))
        dy = habitat_y - (-self.slam_x * math.sin(d_angle) + self.slam_y * math.cos(d_angle))

        # Compute goal position in Habitat coords
        goal_x_rotated = goal_x * math.cos(d_angle) + goal_y * math.sin(d_angle)
        goal_y_rotated = -goal_x * math.sin(d_angle) + goal_y * math.cos(d_angle)
        self.goal_pose_in_habitat_coords = np.array([goal_y_rotated + dy, habitat_z, goal_x_rotated + dx])
        #print('GOAL COORDS IN HABITAT SYSTEM:', self.goal_pose_in_habitat_coords)


    def freeze_callback(self, msg):
        self.freeze = msg.data


    def reset(self):
        pass


    def goal_reached(self):
        robot_position, robot_rotation = self.robot_pose_in_habitat_coords
        # print(robot_position, robot_rotation)
        dst_robot_to_goal = np.sqrt(np.sum((robot_position - self.goal_pose_in_habitat_coords) ** 2))
        if dst_robot_to_goal < self.goal_radius * 1.2:
            print('Goal reached!')
        return (dst_robot_to_goal < self.goal_radius * 1.2)


    def act(self, observations, env):
        self.slam_x, self.slam_y, self.slam_angle = self.get_robot_pose(observations)
        self.robot_pose_in_habitat_coords = observations['agent_position']
        robot_position, robot_rotation = self.robot_pose_in_habitat_coords
        robot_pose_msg = PoseStamped()
        robot_pose_msg.header.stamp = rospy.Time.now()
        robot_pose_msg.header.frame_id = 'habitat'
        robot_pose_msg.pose.position.x = robot_position[0]
        robot_pose_msg.pose.position.y = robot_position[1]
        robot_pose_msg.pose.position.z = robot_position[2]
        robot_pose_msg.pose.orientation.w = robot_rotation.w
        robot_pose_msg.pose.orientation.x = robot_rotation.x
        robot_pose_msg.pose.orientation.y = robot_rotation.y
        robot_pose_msg.pose.orientation.z = robot_rotation.z
        self.robot_pose_publisher.publish(robot_pose_msg)
        print('Robot pose in habitat coords:', self.robot_pose_in_habitat_coords[0])
        if self.goal_pose_in_habitat_coords is None or self.goal_reached():
            if self.goal_position_id < len(self.goal_positions):
                self.goal_pose_in_habitat_coords = self.goal_positions[self.goal_position_id]
            else:
                self.goal_pose_in_habitat_coords = None
            print('Switch to next goal:', self.goal_pose_in_habitat_coords)
            self.goal_position_id += 1
        # print('Freeze:', self.freeze)
        if keyboard.is_pressed('left'):
            return HabitatSimActions.TURN_LEFT
        elif keyboard.is_pressed('right'):
            return HabitatSimActions.TURN_RIGHT
        elif keyboard.is_pressed('up'):
            return HabitatSimActions.MOVE_FORWARD
        elif self.goal_pose_in_habitat_coords is None:
            #if not self.goal_received:
                #print('Random action')
                #return np.random.choice([HabitatSimActions.MOVE_FORWARD, HabitatSimActions.TURN_LEFT])
            print('Total traveled distance:', self.traveled_distance)
            return HabitatSimActions.STOP
        #elif self.freeze:
        #    return HabitatSimActions.STOP
        else:
            next_action = self.follower.get_next_action(self.goal_pose_in_habitat_coords)
            if next_action == HabitatSimActions.MOVE_FORWARD:
                self.traveled_distance += 0.2
            # print('Next action:', next_action)
            if next_action is None:
                print('CANNOT MOVE TO GOAL!!!')
                return HabitatSimActions.STOP
            return next_action