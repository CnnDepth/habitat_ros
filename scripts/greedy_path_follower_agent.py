import rospy
import numpy as np
import habitat
import tf
import keyboard
from nav_msgs.msg import Path, Odometry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from threading import Lock


def normalize(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle


class GreedyPathFollowerAgent(habitat.Agent):

    def __init__(self, goal_radius=0.25, max_d_angle=0.1):
        self.path_subscriber = rospy.Subscriber('/path', Path, self.path_callback)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.tf_listener = tf.TransformListener()
        self.path = []
        self.odom_pose = None
        self.goal_radius = goal_radius
        self.max_d_angle = max_d_angle
        self.mutex = Lock()


    def get_robot_pose(self):
        cur_pose = self.odom_pose
        try:
            pos, quat = self.tf_listener.lookupTransform(
                                'map', 'odom',
                                self.tf_listener.getLatestCommonTime('map',
                                'odom'))
        except:
            print('NO TRANSFORM FROM ODOM TO MAP!!!!')
            return None, None, None
        if cur_pose is None:
            print('NO ODOMETRY!!!')
            return None, None, None
        _, __, tf_angle = tf.transformations.euler_from_quaternion(quat)
        _, __, odom_angle = tf.transformations.euler_from_quaternion([cur_pose.orientation.x, 
                                                                      cur_pose.orientation.y, 
                                                                      cur_pose.orientation.z, 
                                                                      cur_pose.orientation.w])
        current_x, current_y = cur_pose.position.x, cur_pose.position.y
        current_x_new = current_x * np.cos(-tf_angle) + current_y * np.sin(-tf_angle)
        current_y_new = -current_x * np.sin(-tf_angle) + current_y * np.cos(-tf_angle)
        current_x_new += pos[0]
        current_y_new += pos[1]
        return current_x_new, current_y_new, odom_angle + tf_angle


    def path_callback(self, msg):
        print('Path received')
        self.mutex.acquire()
        self.path = []
        for pose in msg.poses:
            self.path.append([pose.pose.position.x, pose.pose.position.y])
        self.mutex.release()


    def odom_callback(self, msg):
        self.odom_pose = msg.pose.pose


    def act(self, observations, env):
        if keyboard.is_pressed('left'):
            return HabitatSimActions.TURN_LEFT
        if keyboard.is_pressed('right'):
            return HabitatSimActions.TURN_RIGHT
        if keyboard.is_pressed('up'):
            return HabitatSimActions.MOVE_FORWARD
        robot_x, robot_y, robot_angle = self.get_robot_pose()
        self.mutex.acquire()
        if len(self.path) < 2 or robot_x is None:
            self.mutex.release()
            return HabitatSimActions.STOP
        nearest_id = 0
        best_distance = np.inf
        for i in range(len(self.path)):
            x, y = self.path[i]
            dst = np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
            if dst < best_distance:
                best_distance = dst
                nearest_id = i
        goal_x, goal_y = self.path[min(nearest_id + 1, len(self.path) - 1)]
        self.mutex.release()
        angle_to_goal = np.arctan2(goal_y - robot_y, goal_x - robot_x)
        turn_angle = normalize(angle_to_goal - robot_angle)
        if abs(turn_angle) < self.max_d_angle:
            return HabitatSimActions.MOVE_FORWARD
        if turn_angle < 0:
            return HabitatSimActions.TURN_RIGHT
        return HabitatSimActions.TURN_LEFT