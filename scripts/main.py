#! /usr/bin/env python

import rospy
import habitat
from std_msgs.msg import Int32
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from keyboard_agent import KeyboardAgent
from shortest_path_follower_agent import ShortestPathFollowerAgent
from greedy_path_follower_agent import GreedyPathFollowerAgent
from random_movement_agent import RandomMovementAgent
from custom_sensors import AgentPositionSensor
from publishers import HabitatObservationPublisher

DEFAULT_RATE = 30
DEFAULT_AGENT_TYPE = 'keyboard'
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1


def main():
    # Initialize ROS node and take arguments
    rospy.init_node('habitat_ros_node')
    node_start_time = rospy.Time.now().to_sec()
    task_config = rospy.get_param('~task_config')
    rate_value = rospy.get_param('~rate', DEFAULT_RATE)
    agent_type = rospy.get_param('~agent_type', DEFAULT_AGENT_TYPE)
    goal_radius = rospy.get_param('~goal_radius', DEFAULT_GOAL_RADIUS)
    max_d_angle = rospy.get_param('~max_d_angle', DEFAULT_MAX_ANGLE)
    rgb_topic = rospy.get_param('~rgb_topic', None)
    depth_topic = rospy.get_param('~depth_topic', None)
    camera_info_topic = rospy.get_param('~camera_info_topic', None)
    true_pose_topic = rospy.get_param('~true_pose_topic', None)
    camera_info_file = rospy.get_param('~camera_calib', None)
    rate = rospy.Rate(rate_value)
    publisher = HabitatObservationPublisher(rgb_topic, 
                                            depth_topic, 
                                            camera_info_topic, 
                                            true_pose_topic,
                                            camera_info_file)
    action_publisher = rospy.Publisher('habitat_action', Int32, latch=True, queue_size=100)

    # Now define the config for the sensor
    config = habitat.get_config(task_config)
    config.defrost()
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    #config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    #config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()

    # Initialize the agent and environment
    env = habitat.Env(config=config)
    env.reset()
    if agent_type == 'keyboard':
       agent = KeyboardAgent()
    elif agent_type == 'shortest_path_follower':
        agent = ShortestPathFollowerAgent(env, goal_radius)
    elif agent_type == 'greedy_path_follower':
        agent = GreedyPathFollowerAgent(goal_radius, max_d_angle)
    elif agent_type == 'random_movement':
        agent = RandomMovementAgent()
    else:
        print('AGENT TYPE {} IS NOT DEFINED!!!'.format(agent_type))
        return

    # Run the simulator with agent
    observations = env.reset()
    print(env.current_episode)
    env.step(HabitatSimActions.MOVE_FORWARD)
    robot_start_time = rospy.Time.now().to_sec()
    print('TIME TO LAUNCH HABITAT:', robot_start_time - node_start_time)
    while not rospy.is_shutdown():
        publisher.publish(observations)
        action = agent.act(observations, env)
        action_msg = Int32()
        action_msg.data = action
        action_publisher.publish(action_msg)
        observations = env.step(action)
        rate.sleep()


if __name__ == '__main__':
    main()