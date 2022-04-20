import sys
sys.path.append('/home/kirill/catkin_ws/src/habitat_ros/scripts/habitat_map')
from .arguments import get_args as get_args_env
from .utils_f.map_builder_objnav import MapBuilder
import skimage
import numpy as np
import utils_f.pose as pu
from time import time
import pandas as pd

class Mapper:
    def __init__(self, env):
        arguments = "--split train \
            --auto_gpu_config 0 \
            -n 1 \
            --num_processes_on_first_gpu 5 \
                --num_processes_per_gpu 16 \
            --train_global 0 --train_local 0 --train_slam 0 \
            --slam_memory_size 150000 \
                --exp_name zero-noise \
            --num_mini_batch 9 \
            --total_num_scenes 1 \
            --split train \
            --task_config my_challenge_mp3d_objectnav2020.local.rgbd.yaml \
            --load_global pretrained_models/model_best.global \
            --load_local pretrained_models/model_best.local \
            --load_slam pretrained_models/model_best.slam \
            --max_episode_length 500".split()
        args_env = get_args_env(arguments)
        args_env.hfov = 79
        args_env.env_frame_height = 480
        args_env.env_frame_width = 640
        args_env.env_frame_semantic_height = 480
        args_env.env_frame_semantic_width = 640
        args_env.camera_height = 0.88
        args_env.du_scale = 1
        args_env.map_size_cm = 2400
        self.args = args_env 

        self.mapper = self.build_mapper()
        full_map_size = args_env.map_size_cm//args_env.map_resolution
        self.mapper.set_class_number(7)
        self.map_size_cm = args_env.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        self.env = env
        
    def reset(self):
        self.mapper.reset_map(self.map_size_cm)
        
        self.collision_map = np.zeros((480,480))
        self.visited = np.zeros((480,480))
        self.curr_loc = [12.,12.]
        self.last_loc = [12.,12.]
        self.curr_loc_map = [12.,12.]
        self.last_loc_map = [12.,12.,0.]
        self.col_width = 1
        self.fat_map = 6
        self.goal_x,self.goal_y = 10,10
        self.poses = [[479,479],[480,480]]
        self.after_reset = 0
        self.last_sim_location = [0.,0.,0.]#self.get_sim_location()
        self.curr_loc_gt = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]

        """
        data = pd.read_csv('/home/kirill/catkin_ws/src/habitat_ros/matterport_name_mappings.tsv', encoding='utf-8', sep="    ")
        data = data.set_index('raw_category')
        self.category_mapping = data['mpcat40']
        self.index_to_title_map = {obj.category.index(): self.category_mapping.get(obj.category.name(), 'unknown') for obj in self.env.sim.semantic_annotations().objects}
        self.instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in self.env.sim.semantic_annotations().objects}
        self.instance_id_to_label_id[0]=41
        """

    
    def step(self, observations, semantic_mask):
        
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change(list(observations['gps']*np.array([1,-1]))+list(observations['compass']))
        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,(dx_gt, dy_gt, do_gt))
        self.curr_loc_map = [int((observations['gps'][0]+24) * 100.0/5),
                int((-observations['gps'][1]+24) * 100.0/5)]

        """
        objectgoal_name = {v: k for k, v in self.env.task._dataset.category_to_task_category_id.items()}[observations['objectgoal'][0]]
        objectgoal_ids = [i for i in self.index_to_title_map if self.index_to_title_map[i] == objectgoal_name]
        self.sem_to_model = np.vectorize(self.instance_id_to_label_id.get)(observations['semantic'])
        self.sem_to_model = np.copy(self.sem_to_model)
        self.sem_to_model[self.sem_to_model==255] = 41
        
        sem_hab22 = np.zeros((480,640))
        for i in objectgoal_ids:
            sem_hab22[self.sem_to_model == i] = 1
        if sem_hab22.max() > 0:
            print('Goal is observed')
        """
        #semantic_mask[observations['depth'][:, :, 0] < 1] = 0
        if semantic_mask.max() > 0:
            print('Goal is observed')
        
        depth = self._preprocess_depth(observations['depth'])
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,self.curr_loc_gt[1]*100.0,np.deg2rad(self.curr_loc_gt[2]))
        start = time()
        fp_proj, self.map, fp_explored, self.explored_map, fp_semantic, self.semantic_map = \
            self.mapper.update_map(depth, mapper_gt_pose, semantic_mask, 0) #observations['semantic'][:,:,0]
        np.savez('/home/kirill/catkin_ws/src/habitat_ros/fp_proj.npz', fp_explored)
        end = time()
        #print('Time to update map:', end - start)

        self.poses.append(self.curr_loc_map)
        
        self.visited[self.curr_loc_map[1]-1:self.curr_loc_map[1], self.curr_loc_map[0] - 0:self.curr_loc_map[0] + 1] = 1
        
        
    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['frame_semantic_width'] = self.args.env_frame_semantic_width
        params['frame_semantic_height'] = self.args.env_frame_semantic_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        params['classes_number'] = 42
        params['goal_threshold'] = 2

        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper 
    
    def get_gt_pose_change(self,loc):
        curr_sim_pose = loc
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do   
    
    def _preprocess_depth(self,depth):
        depth = depth[:, :, 0]*1
        mask2 = depth > 0.99
        depth[mask2] = 0.

        for i in range(depth.shape[1]):
            depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth*450. + 50.
        return depth 