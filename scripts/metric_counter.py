#! /usr/bin/env python

import rospy
import sys
import numpy as np
from nav_msgs.msg import OccupancyGrid

rospy.init_node('metric_counter')

map_topic = rospy.get_param('~map_topic', 'map')
time_checkpoints = [15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240]
path_to_save_results = rospy.get_param('~path_to_save_results')

def map_callback(msg):
	print('Map message received')
	if checkpoint_reached[-1]:
		return
	map_data = np.array(msg.data)
	explored_area = (map_data >= 0).sum() * msg.info.resolution ** 2
	timedelta = rospy.Time.now().to_sec() - start_time
	i = 0
	while i < len(time_checkpoints) and timedelta > time_checkpoints[i]:
		i += 1
	i -= 1
	if i >= 0 and not checkpoint_reached[i]:
		rospy.logwarn('CHECKPOINT {} REACHED. EXPLORED AREA IS {}'.format(i, explored_area))
		checkpoint_reached[i] = True
		areas[i] = explored_area

map_subscriber = rospy.Subscriber(map_topic, OccupancyGrid, map_callback)
start_time = rospy.Time.now().to_sec()
checkpoint_reached = [False] * len(time_checkpoints)
areas = [0] * len(time_checkpoints)

rospy.spin()
fout = open(path_to_save_results, 'w')
fout.write(' '.join([str(x) for x in time_checkpoints]) + '\n')
fout.write(' '.join([str(x) for x in areas]))
fout.close()