import rospy
import numpy as np
import habitat
import tf
import keyboard
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from threading import Lock
from collections import deque


DEFAULT_TIMEOUT_VALUE = 2


def normalize(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle


class GreedyPathFollowerAgent(habitat.Agent):

    def __init__(self, goal_radius=0.25, max_d_angle=0.1):
        self.path_subscriber = rospy.Subscriber('/path', Path, self.path_callback)
        self.odom_subscriber = rospy.Subscriber('/true_pose', PoseStamped, self.odom_callback)
        self.map_subscriber = rospy.Subscriber('/habitat/map', OccupancyGrid, self.map_callback)
        self.stuck_subscriber = rospy.Subscriber('/stuck_alarm', Bool, self.stuck_callback)
        self.tf_listener = tf.TransformListener()
        self.path = None
        self.odom_pose = None
        self.goal_radius = goal_radius
        self.max_d_angle = max_d_angle
        self.mutex = Lock()
        self.map_update_time = 0
        self.update_timeout = DEFAULT_TIMEOUT_VALUE
        self.start_time = 0
        self.slam_lost = False
        self.slam_lost_time = 0
        self.path_update_time = 0
        self.odom_track = []
        self.stuck = False


    def reset(self):
        self.mutex.acquire()
        self.odom_track = []
        self.stuck = False
        self.start_time = 0
        self.map_update_time = 0
        self.slam_lost = False
        self.slam_lost_time = 0
        self.path_update_time = 0
        self.mutex.release()


    def stuck_callback(self, msg):
        self.stuck = msg.data


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
        self.mutex.acquire()
        #print('PATH RECEIVED')
        self.path = []
        for pose in msg.poses:
            self.path.append([pose.pose.position.x, pose.pose.position.y])
        self.mutex.release()


    def odom_callback(self, msg):
        self.odom_pose = msg.pose
        #print(self.odom_pose.position)
        self.odom_track.append([self.odom_pose.position.x, self.odom_pose.position.y])


    def map_callback(self, msg):
        self.map_update_time = rospy.Time.now().to_sec()
        if self.start_time == 0:
            self.start_time = rospy.Time.now().to_sec()


    def act(self, observations, env):
        # if arrow keys pressed, give control to keyboard
        if keyboard.is_pressed('left'):
            return HabitatSimActions.TURN_LEFT
        if keyboard.is_pressed('right'):
            return HabitatSimActions.TURN_RIGHT
        if keyboard.is_pressed('up'):
            return HabitatSimActions.MOVE_FORWARD
        #if keyboard.is_pressed('w'):
        #    return HabitatSimActions.LOOK_UP
        #if keyboard.is_pressed('s'):
        #    return HabitatSimActions.LOOK_DOWN

        cur_time = rospy.Time.now().to_sec()
        # at start, let's rotate 360 degrees to look around
        #print(cur_time - self.start_time)
        if cur_time - self.start_time < 3:
            return HabitatSimActions.TURN_LEFT

        # if SLAM is lost, let's try to restore it - rotate 180, move forward, and rotate 180 again
        """
        if self.odom_pose is not None:
            odom_x, odom_y = self.odom_pose.position.x, self.odom_pose.position.y
            eps = 0.1
            std_threshold = 0.5
            if np.std(np.array(list(self.odom_track)), axis=0).max() > std_threshold or \
               (np.abs(odom_x) < eps and np.abs(odom_y) < eps and cur_time - self.map_update_time > self.update_timeout):
                rospy.logwarn('SLAM lost!')
                if not self.slam_lost:
                    self.slam_lost_time = cur_time
                self.slam_lost = True
            elif cur_time - self.slam_lost_time >= 6:
                self.slam_lost = False
            if self.slam_lost:
                if cur_time - self.slam_lost_time < 6:
                    return HabitatSimActions.TURN_LEFT
                if cur_time - self.slam_lost_time < 7.5:
                    return HabitatSimActions.MOVE_FORWARD
                if cur_time - self.slam_lost_time < 13.5:
                    return HabitatSimActions.TURN_LEFT
                if cur_time - self.slam_lost_time < 15:
                    return HabitatSimActions.MOVE_FORWARD
                if cur_time - self.slam_lost_time > 15:
                    self.slam_lost_time = cur_time
        """

        # if we receive empty paths for long time, let's try to move somewhere
        """
        if self.path == []:
            if cur_time - self.path_update_time > self.update_timeout:
                rospy.logwarn('No path received!')
                n_seconds_from_lost = int(cur_time - self.path_update_time)
                if n_seconds_from_lost % 2 == 0:
                    return HabitatSimActions.TURN_LEFT
                return HabitatSimActions.MOVE_FORWARD
        else:
            self.path_update_time = cur_time
        """

        # find nearest point of path to robot position
        robot_x, robot_y, robot_angle = self.get_robot_pose()
        self.mutex.acquire()
        if self.path is None or len(self.path) < 2 or robot_x is None:
            self.mutex.release()
            #print('PATH OR ROBOT_X IS NONE')
            #print('Path and robot_x:', self.path, robot_x)
            #return HabitatSimActions.STOP
            return np.random.choice([HabitatSimActions.TURN_LEFT, HabitatSimActions.MOVE_FORWARD])
        nearest_id = 0
        best_distance = np.inf
        for i in range(len(self.path)):
            x, y = self.path[i]
            dst = np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
            if dst < best_distance:
                best_distance = dst
                nearest_id = i
        x, y = self.path[min(nearest_id + 1, len(self.path) - 1)]
        x_prev, y_prev = self.path[min(nearest_id, len(self.path) - 1)]
        segment = np.sqrt((x_prev - x) ** 2 + (y_prev - y) ** 2)
        dst = np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
        #if segment < 0.2 and nearest_id + 2 < len(self.path):
        #    nearest_id += 1
        #    x, y = self.path[min(nearest_id + 1, len(self.path) - 1)]
        self.mutex.release()
        dst = np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
        angle_to_goal = np.arctan2(y - robot_y, x - robot_x)
        turn_angle = normalize(angle_to_goal - robot_angle)

        cur_time = rospy.Time.now().to_sec()
        # if SLAM isn't updated, let's wait
        #print(cur_time - self.map_update_time)
        #if cur_time - self.map_update_time > self.update_timeout:
        #    return HabitatSimActions.STOP
        print(dst, turn_angle)

        # if we reached goal, stop
        if dst < self.goal_radius and nearest_id + 1 >= len(self.path) - 1:
            print('GOAL REACHED. STAY ON PLACE')
            #return HabitatSimActions.STOP
            return np.random.choice([HabitatSimActions.TURN_LEFT, HabitatSimActions.MOVE_FORWARD])

        # if next path point is too close to us, move forward to avoid "swing"
        if dst < 0.3 and abs(turn_angle) < np.pi / 2 and not self.stuck:
            return HabitatSimActions.MOVE_FORWARD

        # if our direction is close to direction to goal, move forward
        if abs(turn_angle) < self.max_d_angle and not self.stuck:
            return HabitatSimActions.MOVE_FORWARD

        if self.stuck:
            return np.random.choice([HabitatSimActions.TURN_RIGHT, HabitatSimActions.TURN_LEFT])

        # if direction isn't close, turn left or right
        if turn_angle < 0:
            return HabitatSimActions.TURN_RIGHT
        return HabitatSimActions.TURN_LEFT