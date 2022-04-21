#! /usr/bin/env python

import rospy
import habitat
from habitat_map.env_orb import Env
from semantic_predictor import SemanticPredictor
from std_msgs.msg import Int32
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from keyboard_agent import KeyboardAgent
from shortest_path_follower_agent import ShortestPathFollowerAgent
from greedy_path_follower_agent import GreedyPathFollowerAgent
from random_movement_agent import RandomMovementAgent
from custom_sensors import AgentPositionSensor
from publishers import HabitatObservationPublisher
from habitat_map.mapper import Mapper
from habitat_baselines.config.default import get_config
from habitat_map.utils import draw_top_down_map
import numpy as np
from cv_bridge import CvBridge
from PIL import Image
import cv2
import os
import roslaunch
import gc
import subprocess

DEFAULT_RATE = 30
DEFAULT_AGENT_TYPE = 'keyboard'
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1


class HabitatRunner():
    def __init__(self):
        # Initialize ROS node and take arguments
        task_config = rospy.get_param('~task_config')
        rate_value = rospy.get_param('~rate', DEFAULT_RATE)
        agent_type = rospy.get_param('~agent_type', DEFAULT_AGENT_TYPE)
        self.goal_radius = rospy.get_param('~goal_radius', DEFAULT_GOAL_RADIUS)
        self.max_d_angle = rospy.get_param('~max_d_angle', DEFAULT_MAX_ANGLE)
        rgb_topic = rospy.get_param('~rgb_topic', None)
        depth_topic = rospy.get_param('~depth_topic', None)
        camera_info_topic = rospy.get_param('~camera_info_topic', None)
        true_pose_topic = rospy.get_param('~true_pose_topic', None)
        camera_info_file = rospy.get_param('~camera_calib', None)
        self.rate = rospy.Rate(rate_value)
        self.publisher = HabitatObservationPublisher(rgb_topic, 
                                                    depth_topic, 
                                                    camera_info_topic, 
                                                    true_pose_topic,
                                                    camera_info_file)
        # Now define the config for the sensor
        self.action_publisher = rospy.Publisher('habitat_action', Int32, latch=True, queue_size=100)
        self.map_publisher = rospy.Publisher('habitat/map', OccupancyGrid, latch=True, queue_size=100)
        self.semantic_map_publisher = rospy.Publisher('habitat/semantic_map', OccupancyGrid, latch=True, queue_size=100)
        self.reset_publisher = rospy.Publisher('/reset_exploration', String, latch=True, queue_size=100)

        # Now define the config for the sensor
        habitat_path = '/home/kirill/habitat-lab/data'
        config = habitat.get_config(task_config)
        config.defrost()
        config.DATASET.DATA_PATH = os.path.join(habitat_path, 'datasets/objectnav_hm3d_v1/val/val.json.gz')
        #config.DATASET.CONTENT_SCENES = ['mv2HUxq3B53']
        config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
        config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
        config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
        config.SIMULATOR.AGENT_0.SENSORS.append("SEMANTIC_SENSOR")
        config.SIMULATOR.SCENE_DATASET = os.path.join(habitat_path, "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json")
        #config.SIMULATOR.SCENE = 'mv2HUxq3B53'
        print('PATH:', config.SIMULATOR.SCENE_DATASET)
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK.SENSORS.append("HEADING_SENSOR")
        config.freeze()
        self.config = config

        # Initialize the agent and environment
        self.env = Env(config=config)
        #env = env_orb.env(config=config)
        print('Environment created')

        self.eval_episodes = [3, 4, 5, 7, 99, 100, 101, 102, 103, 104, 106, 107, 198, 199, 200, 202, 203, 204, 205, 206, 297, 299, 300, 301, 303, 304, 305, 396, 397, 398, 399, 401, 402, 403, 495, 496, 497, 498, 499, 501, 503, 594, 595, 596, 597, 598, 599, 600, 602, 693, 694, 696, 697, 698, 700, 701, 792, 793, 794, 795, 797, 798, 799, 800, 891, 894, 895, 896, 899, 990, 991, 992, 994, 996, 997, 998, 1089, 1093, 1094, 1095, 1096, 1097, 1188, 1189, 1191, 1192, 1193, 1194, 1195, 1196, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1386, 1387, 1390, 1391, 1392, 1393, 1394, 1485, 1486, 1487, 1489, 1493, 1584, 1585, 1587, 1588, 1589, 1590, 1591, 1592, 1684, 1685, 1686, 1687, 1688, 1690, 1691, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1881, 1885, 1886, 1888, 1889]
        self.successes = []
        self.spls = []
        self.softspls = []
        self.fake_finishes = 0

        self.mapper = Mapper(self.env)
        self.semantic_predictor = SemanticPredictor(threshold=0.35)
        if agent_type == 'keyboard':
           self.agent = KeyboardAgent()
        elif agent_type == 'shortest_path_follower':
            self.agent = ShortestPathFollowerAgent(self.env, self.goal_radius)
        elif agent_type == 'greedy_path_follower':
            self.agent = GreedyPathFollowerAgent(self.goal_radius, self.max_d_angle)
        elif agent_type == 'random_movement':
            self.agent = RandomMovementAgent()
        else:
            print('AGENT TYPE {} IS NOT DEFINED!!!'.format(agent_type))
            return


    def run_episode(self, ii):
        observations = self.env.reset(i=ii)
        print('Episode number', ii)
        objectgoal = observations['objectgoal'][0]
        objectgoal_name = {v: k for k, v in self.env.task._dataset.category_to_task_category_id.items()}[objectgoal]
        print('Objectgoal:', objectgoal_name)

        self.mapper.reset()
        self.agent.reset()
        reset_msg = String()
        reset_msg.data = 'reset'
        self.reset_publisher.publish(reset_msg)

        # Run the simulator with agent
        #observations = env.reset()
        #print("Observations received")
        #print([x for x in observations])
        step_start_time = rospy.Time.now()
        self.env.step(HabitatSimActions.MOVE_FORWARD)
        finished = False
        while not rospy.is_shutdown() and not self.env.episode_over:
            step_start_time = rospy.Time.now()
            self.publisher.publish(observations)
            action = self.agent.act(observations, self.env)
            action_msg = Int32()
            action_msg.data = action
            self.action_publisher.publish(action_msg)
            time_after_publish = rospy.Time.now()
            observations = self.env.step(action)
            time_after_step = rospy.Time.now()
            semantic_mask = self.semantic_predictor(observations['rgb'], observations['objectgoal'][0])
            self.mapper.step(observations, semantic_mask)
            time_after_mapping = rospy.Time.now()

            x, y = observations['gps']
            y *= -1
            x_cell = (x + 12) * 20
            y_cell = (y + 12) * 20
            semantic_map = self.mapper.semantic_map[0].copy()
            semantic_map[semantic_map > 0] = 1
            kernel = np.ones((3, 3), dtype=np.uint8)
            semantic_map = cv2.erode(semantic_map, kernel)
            i, j = (semantic_map > 0).nonzero()
            if len(i) > 0:
                dst = np.sqrt((y_cell - i) ** 2 + (x_cell - j) ** 2)
                print('Min distance:', dst.min())
                if dst.min() <= 1.0 / 0.05:#self.config.TASK.SUCCESS.SUCCESS_DISTANCE / 0.05:
                    print('FINISH EPISODE!')
                    finished = True
                    self.env.step(HabitatSimActions.STOP)
                    break

            map_msg = OccupancyGrid()
            map_msg.header.stamp = rospy.Time.now()
            map_msg.header.frame_id = 'map'
            map_msg.info.resolution = 0.05
            map_msg.info.width = 480
            map_msg.info.height = 480
            map_msg.info.origin.position.x = -12
            map_msg.info.origin.position.y = -12
            map_data = np.ones((480, 480), dtype=np.int8) * (-1)
            map_data[self.mapper.mapper.map[:, :, 0] > 0] = 0
            map_data[self.mapper.mapper.map[:, :, 1] > 0] = 100
            map_msg.data = list(map_data.ravel())
            self.map_publisher.publish(map_msg)

            map_msg.header.stamp = rospy.Time.now()
            map_msg.header.frame_id = 'map'
            map_msg.info.resolution = 0.05
            map_msg.info.width = 480
            map_msg.info.height = 480
            map_msg.info.origin.position.x = -12
            map_msg.info.origin.position.y = -12
            map_data = np.zeros((480, 480), dtype=np.int8)
            map_data[self.mapper.mapper.map[:, :, 1] > 0] = 100
            semantic_map = self.mapper.semantic_map[0].copy()
            semantic_map[semantic_map > 0] = 1
            kernel = np.ones((3, 3), dtype=np.uint8)
            map_data[semantic_map > 0] = 0
            semantic_map = cv2.erode(semantic_map, kernel)
            i, j = (semantic_map > 0).nonzero()
            for di in range(-5, 6):
                for dj in range(-5, 6):
                    semantic_map[np.clip(i + di, 0, 479), np.clip(j + dj, 0, 479)] = 1
            map_data[semantic_map > 0] = -1
            #map_data[self.mapper.mapper.semantic_map[0] > 0] = -1
            map_msg.data = list(map_data.ravel())
            self.semantic_map_publisher.publish(map_msg)

            time_after_map_publish = rospy.Time.now()

            iteration_time = (time_after_mapping - step_start_time).to_sec()
            if iteration_time > 0.5:
                print('Whole iteration time:', iteration_time)
                print('Env step time:', (time_after_step - time_after_publish).to_sec())
                print('Publishing time:', (time_after_publish - step_start_time).to_sec())
                print('Mapper update time:', (time_after_mapping - time_after_step).to_sec())
                print('Map publishing time:', (time_after_map_publish - time_after_mapping).to_sec())
                print()

            self.rate.sleep()

        metrics = self.env.task.measurements.get_metrics()
        if finished and metrics['success'] == 0:
            self.fake_finishes += 1
        print('Success:', metrics['success'])
        print('SPL:', metrics['spl'])
        print('SoftSPL:', metrics['softspl'])
        self.successes.append(metrics['success'])
        self.spls.append(metrics['spl'])
        self.softspls.append(metrics['softspl'])
        print('Average success:', np.mean(self.successes))
        print('Average SPL:', np.mean(self.spls))
        print('Average softSPL:', np.mean(self.softspls))
        print('Number of false goal detections:', self.fake_finishes)
        top_down_map = draw_top_down_map(metrics,
                                         observations['heading'][0],
                                         observations['rgb'][0].shape[0])
        im = Image.fromarray(top_down_map)
        im.save('/home/kirill/catkin_ws/src/habitat_ros/top_down_maps/episode_{}.png'.format(ii))
        return metrics


def main():
    rospy.init_node('habitat_ros_node', anonymous=True)
    #launch = roslaunch.scriptapi.ROSLaunch()
    #launch.start()
    #uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    #roslaunch.configure_logging(uuid)
    runner = HabitatRunner()
    for i in runner.eval_episodes:
        #launch_exploration = roslaunch.parent.ROSLaunchParent(uuid, ["/home/kirill/catkin_ws/src/m-explore/explore/launch/explore_with_planner.launch"])
        #launch_exploration.start()
        runner.run_episode(i)
        #launch_exploration.shutdown()
        #process.kill()
        #rospy.sleep(10)
        #gc.collect()


if __name__ == '__main__':
    main()