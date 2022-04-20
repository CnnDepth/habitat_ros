import numpy as np
# from numba import njit
import sys
sys.path.append('/home/kirill/catkin_ws/src/habitat_ros/scripts/habitat_map/utils_f')
import depth_utils as du
import time
import skimage.measure

from habitat.core.utils import try_cv2_import
from collections import deque

cv2 = try_cv2_import()

class MapBuilder(object):

    def __init__(self, params):
        self.params = params

        frame_width = params['frame_width']
        frame_height = params['frame_height']

        self.frame_semantic_width = params['frame_semantic_width']
        self.frame_semantic_height = params['frame_semantic_height']

        fov = params['fov']

        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)

        self.semantic_camera_matrix = du.get_camera_matrix(
            self.frame_semantic_width,
            self.frame_semantic_height,
            fov)

        self.vision_range = params['vision_range']
        self.vision_range_semantic = params['vision_range']# // n
        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution']
        self.resolution_semantic = params['resolution']# * n

        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']

        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']
        self.classes_number = params['classes_number']
        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)
        self.semantic_map = np.zeros((self.classes_number,
                                      self.map_size_cm // self.resolution_semantic,
                                      self.map_size_cm // self.resolution_semantic,
                                      ), dtype=np.float32)
        self.agent_height = params['agent_height']
        self.agent_view_angle = params['agent_view_angle']
        self.goal_threshold = params['goal_threshold']
        
        return
    
    def set_class_number(self, class_number):
        print("SET CLASS NUMBER", class_number)
        self.classes_number = class_number
        
    def fill_by_bfs(self, geocentric_map, current_pose):

        def normalize_angle(angle):
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            return angle

        def check_borders(x, y, current_pose, visited):
            HFOV = 79
            max_deviation = HFOV / (180 / np.pi) / 2
            if x < 0 or x >= 480 or y < 0 or y >= 480:
                return False
            if visited[x, y] == 1:
                return False
            x_start, y_start, angle = current_pose
            x_cell = round(x_start) / 5
            y_cell = round(y_start) / 5
            if abs(x_cell - x) > 32 or abs(y_cell - y) > 32:
                return False
            direction_to_point = np.arctan2(y - y_cell, x - x_cell)
            if (abs(x_cell - x) > 1 or abs(y_cell - y) > 1) and abs(normalize_angle(direction_to_point - angle)) > max_deviation:
                return False
            return True


        x_start, y_start, angle = current_pose
        x_cell = round(x_start / 5)
        y_cell = round(y_start / 5)
        dq = deque()
        dq.append((x_cell, y_cell))
        visited = np.zeros((geocentric_map.shape[0], geocentric_map.shape[1]))
        while len(dq) > 0:
            x, y = dq.popleft()
            #print(x, y)
            if not visited[x, y]:
                if geocentric_map[y, x].sum() == 0:
                    if check_borders(x - 1, y, current_pose, visited):
                        dq.append((x - 1, y))
                    if check_borders(x, y - 1, current_pose, visited):
                        dq.append((x, y - 1))
                    if check_borders(x, y + 1, current_pose, visited):
                        dq.append((x, y + 1))
                    if check_borders(x + 1, y, current_pose, visited):
                        dq.append((x + 1, y))
                    if check_borders(x - 1, y - 1, current_pose, visited):
                        dq.append((x - 1, y - 1))
                    if check_borders(x + 1, y - 1, current_pose, visited):
                        dq.append((x + 1, y - 1))
                    if check_borders(x + 1, y + 1, current_pose, visited):
                        dq.append((x + 1, y + 1))
                    if check_borders(x - 1, y + 1, current_pose, visited):
                        dq.append((x - 1, y + 1))
                    geocentric_map[y, x, 0] = 1
                else:
                    return geocentric_map
            visited[x, y] = 1
        return geocentric_map
        
        
    def get_geocentric_flat(self, depth, camera_matrix, vision_range, resolution, map_shape, current_pose):
        t1 = time.time()
        point_cloud = du.get_point_cloud_from_z(depth, camera_matrix, scale=self.du_scale)
        t2 = time.time()
        #print('Time to get point cloud:', t2 - t1)

        
        shift_loc = [vision_range * resolution // 2, 0, np.pi / 2.0]
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)
        agent_view_centered = du.transform_pose(agent_view, shift_loc)
        agent_view_flat = du.bin_points(
            agent_view_centered,
            vision_range,
            self.z_bins,
            resolution)
        agent_view_cropped = agent_view_flat[:, :, 1]
        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0
        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0
        t3 = time.time()
        #print('Time to get agent view:', t3 - t2)

        geocentric_pc = du.transform_pose(agent_view, current_pose)
        geocentric_flat = du.bin_points(
            geocentric_pc,
            map_shape,
            self.z_bins,
            resolution)
        t4 = time.time()
        #print('Time to bin points:', t4 - t3)

        geocentric_flat = self.fill_by_bfs(geocentric_flat, current_pose)
        t5 = time.time()
        #print('Time to bfs:', t5 - t4)
        
        return geocentric_flat, agent_view_cropped, agent_view_explored, geocentric_pc

    def update_map(self, depth, current_pose, semantic, goal_category_id):

        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN

        t0 = time.time()
        self.geocentric_flat, agent_view_cropped, agent_view_explored, _ = self.get_geocentric_flat(depth, self.camera_matrix, \
                                                                                                 self.vision_range, self.resolution, self.map.shape[0], current_pose)
        self.geocentric_view = self.geocentric_flat.sum(2)
        t1 = time.time()
        #print('Time to get geocentric flat:', t1 - t0)

        #print(self.map.shape)
        self.map = self.map + self.geocentric_flat
        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0
        t2 = time.time()
        #print('Time to count map:', t2 - t1)

        """
        ##############################
        depth_semantic = np.zeros([self.classes_number, *semantic.shape[:2]])
        depth_semantic[goal_category_id] = depth
        nan_map = (semantic == goal_category_id).astype(np.float)
        nan_map[nan_map == 0] = np.NaN
        depth_semantic[goal_category_id] = depth_semantic[goal_category_id] * nan_map
        agent_view_semantic = []
        geocentric_pc_semantic = []
        semantic_geocentric_flat, agent_view_semantic_cur, _, geocentric_pc = self.get_geocentric_flat(depth_semantic[goal_category_id], self.semantic_camera_matrix, self.vision_range_semantic, self.resolution_semantic, self.semantic_map.shape[1], current_pose)
        geocentric_pc_semantic.append(geocentric_pc)
        agent_view_semantic.append(agent_view_semantic_cur)
        
        self.semantic_map[goal_category_id] = self.semantic_map[goal_category_id] + semantic_geocentric_flat[:, :, 1]
            
        agent_view_semantic = np.array(agent_view_semantic)

        self.last_object_depth = depth_semantic[goal_category_id]
        self.last_object_point_cloud = geocentric_pc_semantic[0]
        ##############################
        """

        depth_semantic = np.zeros([self.classes_number, *semantic.shape[:2]])

        """
        for i in range(self.classes_number):
            depth_semantic[i] = depth
            nan_map = (semantic == i).astype(np.float)
            nan_map[nan_map == 0] = np.NaN
            depth_semantic[i] = depth_semantic[i] * nan_map
        """
        nan_map = (semantic == 1).astype(np.float)
        nan_map[nan_map == 0] = np.NaN
        depth_semantic[0] = depth * nan_map

        self.depth_semantic = depth_semantic.copy()
        t3 = time.time()
        #print('Time to take depth semantic:', t3 - t2)
        
        agent_view_semantic = []
        geocentric_pc_semantic = []
        for index, depth in enumerate(depth_semantic[:1]):
            #print(index, depth.shape)
            semantic_geocentric_flat, agent_view_semantic_cur, _, geocentric_pc = self.get_geocentric_flat(depth, 
                                                                                                           self.semantic_camera_matrix,  
                                                                                                           self.vision_range_semantic, 
                                                                                                           self.resolution_semantic, 
                                                                                                           self.semantic_map.shape[1], 
                                                                                                           current_pose)
            
            geocentric_pc_semantic.append(geocentric_pc)
            agent_view_semantic.append(agent_view_semantic_cur)
            #semantic = semantic_geocentric_flat[:, :, 1]
            #semantic[semantic > 0] = 1
            #kernel = np.ones((2, 2), dtype=np.uint8)
            #semantic = cv2.erode(semantic, kernel)
            semantic_geocentric_flat[semantic_geocentric_flat > 0] = 1
            self.semantic_map[index][self.geocentric_view > 0] *= 0.95
            self.semantic_map[index] = self.semantic_map[index] + semantic_geocentric_flat[:, :, 1]
            
        agent_view_semantic = np.array(agent_view_semantic)
        #print('Goal category id:', goal_category_id)

        self.last_object_depth = depth_semantic[goal_category_id]
        self.last_object_point_cloud = geocentric_pc_semantic[goal_category_id]
        t4 = time.time()
        #print('Time to get agent_view_semantic:', t4 - t3)
        
        #"""
        ################################

        map_semantic_gt = self.semantic_map.copy()
        map_semantic_gt[map_semantic_gt < self.goal_threshold] = 0.0
        map_semantic_gt[map_semantic_gt >= self.goal_threshold] = 1.0

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, \
               agent_view_semantic, map_semantic_gt

    def get_info_for_global_reward(self):
        return self.last_object_point_cloud, self.last_object_depth

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]

        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.semantic_map = np.zeros((self.classes_number,
                                      self.map_size_cm // self.resolution_semantic,
                                      self.map_size_cm // self.resolution_semantic,
                                      ), dtype=np.float32)

        self.last_object_point_cloud = None
        self.last_object_depth = None


    def get_map(self):
        return self.map