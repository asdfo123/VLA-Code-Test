import sapien
import torch
import numpy as np
from typing import Dict, Any, Union, List
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig

import wzj.utils as utils
@register_env('StowItem-v1', max_episode_steps=100)
class StowItemEnv(BaseEnv):
    """
    Task: Stow an item from the tabletop to a specified drawer of the cabinet.
    """
    SUPPORTED_ROBOTS = ['panda']
    agent: Panda

    # set some commonly used cpnstants
    half_size = 0.02
    radius = 0.02
    bound_box: Dict[str, torch.Tensor] # bounding box of each drawer
    drawer_positions: Dict[str, torch.Tensor] # position of each drawer

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, num_envs=1,
                reconfiguration_freq=None, n_items=6, **kwargs):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        self.n_items = n_items
        self.robot_init_qpos_noise = robot_init_qpos_noise

        super().__init__(*args, robot_uids=robot_uids,
                        reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)
    
    def _load_agent(self, options):
        # set a reasonable initial pose for the agent
        super()._load_agent(options, sapien.Pose(p=[0.5, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        loader = self.scene.create_urdf_loader()
        # change the scale of the loaded articulation geometries
        loader.scale = 0.4
        urdf_file = f"./wzj/assets/45290/mobility.urdf"
        articulation_builders = loader.parse(urdf_file)["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[-0.2, -0.4, 0.4], q=[0.7071, 0, 0, -0.7072])
        builder.build(name="cabinet")

        # Randomly generate n_items of random size, color, and shape       
        self.items = utils.create_random_items(self.scene, self.n_items,
                                               self.half_size, self.radius)
        
        # Below for evaluation
        cabinet_art = self.scene.get_all_articulations()[1]
        bound_box_, drawer_position_ = {}, {}

        # link_2 : The top drawer; link_1 : The bottom drawer; link_0 : The bottom drawer
        # Use the Axis-Aligned Bounding Box to indicate the size of each drawer
        for link in cabinet_art.get_links():
            if not link.get_collision_shapes():
                continue
            aabb = link.get_global_aabb_fast()
            size = aabb[1] - aabb[0]
            pos = link.pose.p

            # calculate the bounding box of the drawer
            drawer_border_min = torch.tensor(pos) - torch.tensor(size) / 2
            drawer_border_max = torch.tensor(pos) + torch.tensor(size) / 2
            bound_box_[link.name] = (drawer_border_min, drawer_border_max)
            drawer_position_[link.name] = torch.tensor(pos)
            # print(f"{link.name}: length {size[0]:.3f}, wide {size[1]:.3f}, height {size[2]:.3f}, Pos {pos}")

        self.bound_box = bound_box_
        self.drawer_positions = drawer_position_

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        # set robot pose
        robot_pose = Pose.create_from_pq(
                p=[0.3, 0.3, 0], q=[0.7071, 0, 0, -0.7072]
            )
        self.agent.robot.set_pose(robot_pose)

        with torch.device(self.device):
            b = len(env_idx)
            # set item poses
            for i, item in enumerate(self.items):
                item_pos = torch.zeros((b, 3))
                item_pos[..., 0] = torch.rand((b,)) * 0.4 - 0.2
                item_pos[..., 1] = torch.rand((b,)) * 0.8
                # Use bounding box to make sure objects upright and don’t intersect z=0
                collision_mesh = item.get_first_collision_mesh()
                item_pos[..., 2] = -collision_mesh.bounding_box.bounds[0, 2]
                
                item_pose = Pose.create_from_pq(p=item_pos, q=[1, 0, 0, 0])
                item.set_pose(item_pose)
                
                item.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                item.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        """ TODO: Consider n_envs > 1 """
        
        with torch.device(self.device):
            box_positions_list = [item.pose.p for item in self.items if "box" in item.name]
            n_boxes = len(box_positions_list)
            sphere_positions_list = [item.pose.p for item in self.items if "sphere" in item.name]
            n_spheres = len(sphere_positions_list)

            # check the number of items in the right place
            sphere_in_drawer_2 = [
                sphere for sphere in sphere_positions_list
                if torch.all(torch.logical_and(self.bound_box["scene-0-cabinet_link_3"][0] < torch.tensor(sphere),
                            torch.tensor(sphere) < self.bound_box["scene-0-cabinet_link_3"][1]))
            ]

            box_in_drawer_1 = [
                box for box in box_positions_list
                if torch.all(torch.logical_and(self.bound_box["scene-0-cabinet_link_2"][0] < torch.tensor(box),
                            torch.tensor(box) < self.bound_box["scene-0-cabinet_link_2"][1]))
            ]
        
        # print(f"n_spheres: {n_spheres}, n_boxes: {n_boxes}")
        # print(f"sphere_in_drawer_2: {len(sphere_in_drawer_2)}, box_in_drawer_1: {len(box_in_drawer_1)}")

        all_placed = len(sphere_in_drawer_2) == n_spheres and len(box_in_drawer_1) == n_boxes
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool) + all_placed,
            "box_positions": box_positions_list,
            "sphere_positions": sphere_positions_list,
        }

    def _get_obs_extra(self, info: Dict):     
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            for i, item in enumerate(self.items):
                obs[f"item_{i}_pose"] = item.pose.raw_pose
                obs[f"item_{i}_vel"] = item.linear_velocity
        
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)
            
            # Success reward
            success = info["success"]
            reward = torch.where(success, reward + 10.0, reward)

            # Reward for having the box closer to the drawer 1
            box_positions = info["box_positions"]
            shpere_positions = info["sphere_positions"]
            # The numbers of boxes and spheres could be None. If any bug occurs, please check this part.
            if len(box_positions) > 0:
                box_positions = torch.stack(box_positions)
            else:
                box_positions = torch.zeros(self.num_envs, 3, device=self.device)
            if len(shpere_positions) > 0:               
                shpere_positions = torch.stack(shpere_positions)
            else:
                shpere_positions = torch.zeros(self.num_envs, 3, device=self.device)
            
            tcp_pose = self.agent.tcp.pose.p
            box_goal_distances = torch.linalg.norm(
                box_positions[..., :2] - self.drawer_positions["scene-0-cabinet_link_2"][:2], dim=-1)
            tcp_to_boxes = torch.linalg.norm(
                box_positions - tcp_pose.unsqueeze(0), dim=-1)

            for i in range(len(box_positions)):
                reward += torch.exp(-5.0 * box_goal_distances[i]) * 0.2

            shpere_goal_distances = torch.linalg.norm(
                shpere_positions[..., :2] - self.drawer_positions["scene-0-cabinet_link_3"][:2], dim=-1)
            tcp_to_spheres = torch.linalg.norm(
                shpere_positions - tcp_pose.unsqueeze(0), dim=-1)

            for i in range(len(shpere_positions)):
                reward += torch.exp(-5.0 * shpere_goal_distances[i]) * 0.2
            
            # Reward for getting the TCP close to any item (to encourage picking)
            closest_item_to_tcp = torch.min(torch.min(tcp_to_boxes, dim=0)[0], torch.min(tcp_to_spheres, dim=0)[0])
            reward += torch.exp(-10.0 * closest_item_to_tcp) * 0.1  
            return reward     

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """ TODO: Verify the max_reward """
        max_reward = 16.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[-0.1, 0.9, 0.3], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([-0.6, 1.3, 0.8], [0.0, 0.13, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

