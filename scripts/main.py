#! /usr/bin/env python

import rospy
import habitat
from keyboard_agent import KeyboardAgent
from typing import Any
from gym import spaces
import numpy as np

DEFAULT_RATE = 30
DEFAULT_AGENT_TYPE = 'keyboard'


# Define the sensor and register it with habitat
# For the sensor, we will register it with a custom name
@habitat.registry.register_sensor(name="position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        sensor_states = self._sim.get_agent_state().sensor_states
        return (sensor_states['rgb'].position, sensor_states['rgb'].rotation)


def main():
	# Initialize ROS node and take arguments
    rospy.init_node('habitat_ros_node')
    task_config = rospy.get_param('~task_config')
    rate_value = rospy.get_param('~rate', DEFAULT_RATE)
    agent_type = rospy.get_param('~agent_type', DEFAULT_AGENT_TYPE)
    rate = rospy.Rate(rate_value)

    # Now define the config for the sensor
    config = habitat.get_config(task_config)
    config.defrost()
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.freeze()

    # Initialize the agent and environment
    if agent_type == 'keyboard':
    	agent = KeyboardAgent()
    else:
    	print('AGENT TYPE {} IS NOT DEFINED!!!'.format(agent_type))
    	return
    env = habitat.Env(config=config)

    # Run the simulator with agent
    observations = env.reset()
    while not rospy.is_shutdown():
    	action = agent.act(observations, env)
    	observations = env.step(action)
    	rate.sleep()


if __name__ == '__main__':
	main()