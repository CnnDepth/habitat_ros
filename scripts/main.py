#! /usr/bin/env python

import rospy
import habitat
#from habitat_map.env_orb import Env
#from semantic_predictor import SemanticPredictor
#from semantic_predictor_segformer import SemanticPredictor
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
from skimage.io import imsave
from tqdm import tqdm
from habitat_map import env_orb
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
        semantic_topic = rospy.get_param('~semantic_topic', None)
        camera_info_topic = rospy.get_param('~camera_info_topic', None)
        #semantic_mask_topic = rospy.get_param('~semantic_mask_topic', None)
        true_pose_topic = rospy.get_param('~true_pose_topic', None)
        camera_info_file = rospy.get_param('~camera_calib', None)
        scene_name = rospy.get_param('~scene_name', None)
        print('SCENE NAME:', scene_name)
        print('TASK CONFIG:', task_config)
        self.scene_name = scene_name
        self.rate = rospy.Rate(rate_value)
        self.publisher = HabitatObservationPublisher(rgb_topic, 
                                                    depth_topic, 
                                                    #semantic_topic,
                                                    camera_info_topic, 
                                                    true_pose_topic,
                                                    camera_info_file)
        # Now define the config for the sensor
        self.action_publisher = rospy.Publisher('habitat_action', Int32, latch=True, queue_size=100)
        self.map_publisher = rospy.Publisher('habitat/map', OccupancyGrid, latch=True, queue_size=100)
        #self.semantic_map_publisher = rospy.Publisher('habitat/semantic_map', OccupancyGrid, latch=True, queue_size=100)
        self.reset_publisher = rospy.Publisher('/reset_exploration', String, latch=True, queue_size=100)

        # Now define the config for the sensor
        habitat_path = '/home/kirill/habitat-lab/data'
        config = habitat.get_config(task_config)
        config.defrost()
        #config.DATASET.DATA_PATH = os.path.join(habitat_path, 'datasets/objectnav_hm3d_v1/val/val.json.gz')
        #config.DATASET.CONTENT_SCENES = ['mv2HUxq3B53']
        config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
        config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
        config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
        config.SIMULATOR.AGENT_0.SENSORS.append("SEMANTIC_SENSOR")
        print(config.DATASET)
        config.SIMULATOR.SCENE_DATASET = os.path.join(habitat_path, "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json")
        #config.SIMULATOR.SCENE = 'mv2HUxq3B53'
        print('PATH:', config.SIMULATOR.SCENE_DATASET)
        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK.SENSORS.append("HEADING_SENSOR")
        config.freeze()
        self.config = config

        # Initialize the agent and environment
        self.env = habitat.Env(config=config)
        #self.env = env_orb.Env(config=config)
        print('Environment created')

        self.mapper = Mapper()
        #self.semantic_predictor = SemanticPredictor(threshold=0.35)
        goal_positions = np.loadtxt('/home/kirill/catkin_ws/src/habitat_ros/goal_positions/mp3d/{}.txt'.format(scene_name))
        #goal_positions = np.loadtxt('/home/kirill/catkin_ws/src/habitat_ros/goal_positions_300m.txt')
        if agent_type == 'keyboard':
           self.agent = KeyboardAgent()
        elif agent_type == 'shortest_path_follower':
            self.agent = ShortestPathFollowerAgent(self.env, self.goal_radius, goal_positions)
        elif agent_type == 'greedy_path_follower':
            self.agent = GreedyPathFollowerAgent(self.goal_radius, self.max_d_angle)
        elif agent_type == 'random_movement':
            self.agent = RandomMovementAgent()
        else:
            print('AGENT TYPE {} IS NOT DEFINED!!!'.format(agent_type))
            return
        self.dataset_save_path = '/data/datasets/opr_training_data/gibson'


    def publish_map(self):
        occupancy_map = self.mapper.mapper.map
        map_msg = OccupancyGrid()
        map_msg.header.stamp = rospy.Time.now()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = self.mapper.mapper.resolution / 100.
        map_msg.info.width = occupancy_map.shape[1]
        map_msg.info.height = occupancy_map.shape[0]
        map_msg.info.origin.position.x = -occupancy_map.shape[1] * self.mapper.mapper.resolution / 200.
        map_msg.info.origin.position.y = -occupancy_map.shape[0] * self.mapper.mapper.resolution / 200.
        map_data = np.ones((map_msg.info.height, map_msg.info.width), dtype=np.int8) * (-1)
        map_data[occupancy_map[:, :, 0] > 0] = 0
        map_data[occupancy_map[:, :, 1] > 0] = 100
        map_msg.data = list(map_data.ravel())
        self.map_publisher.publish(map_msg)


    def run_episode(self):
        observations = self.env.reset()
        self.env.step(HabitatSimActions.MOVE_FORWARD)

        self.mapper.reset()
        self.agent.reset()
        reset_msg = String()
        reset_msg.data = 'reset'
        self.reset_publisher.publish(reset_msg)
        
        """
        points = self.env.get_navigable_points()
        orientations = [[0, 0, 0, 1], [0, 0.7071, 0, 0.7071], [0, 1, 0, 0], [0, -0.7071, 0, 0.7071]]
        poses = []
        i = 0
        if not os.path.exists(os.path.join(self.dataset_save_path, self.scene_name)):
            os.mkdir(os.path.join(self.dataset_save_path, self.scene_name))
        for pt in tqdm(points):
            for ori in orientations:
                i += 1
                observations = self.env.reset(start_position=pt, start_orientation=ori)
                #self.env.step(HabitatSimActions.TURN_LEFT)
                #print(observations.keys())
                rgb = observations['rgb']
                depth = observations['depth']
                depth = (depth * 255).astype(np.uint8)
                poses.append(pt + ori)
                imsave(os.path.join(self.dataset_save_path, self.scene_name, '{}_rgb.png'.format(i)), rgb)
                imsave(os.path.join(self.dataset_save_path, self.scene_name, '{}_depth.png'.format(i)), depth)
        np.savetxt(os.path.join(self.dataset_save_path, self.scene_name, 'poses.txt'), np.array(poses))
        return
        """

        step_start_time = rospy.Time.now()
        trajectory = []
        step = 0
        while not rospy.is_shutdown() and not self.env.episode_over:
            step_start_time = rospy.Time.now()
            t0 = rospy.Time.now().to_sec()
            self.publisher.publish(observations, step_start_time)
            t1 = rospy.Time.now().to_sec()
            #print('Publish time:', t1 - t0)
            action = self.agent.act(observations, self.env)
            t2 = rospy.Time.now().to_sec()
            #print('Action time:', t2 - t1)
            #if self.agent.goal_pose_in_habitat_coords is None:
            #    print('NO GOAL TO MOVE. FINISH')
            #    break
            #if step % 3 == 1:
            #    self.mapper.step(observations, observations['semantic'])
            #action_msg = Int32()
            #action_msg.data = action
            before_publish = rospy.Time.now().to_sec()
            #self.action_publisher.publish(action_msg)
            observations = self.env.step(action)
            robot_x, robot_y = observations['gps']
            robot_y = -robot_y
            robot_angle = observations['compass']
            after_publish = rospy.Time.now().to_sec()
            trajectory.append((robot_x, robot_y, robot_angle, step_start_time.to_sec()))
            t3 = rospy.Time.now().to_sec()
            #print('Step time:', t3 - t2)
            #if step % 10 == 1:
            #    self.publish_map()
            self.rate.sleep()
            step += 1
        np.savetxt('/home/kirill/catkin_ws/src/habitat_ros/trajectory.txt', trajectory)


def main():
    rospy.init_node('habitat_ros_node', anonymous=True)
    runner = HabitatRunner()
    runner.run_episode()


if __name__ == '__main__':
    main()