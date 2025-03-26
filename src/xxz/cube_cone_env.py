from typing import Any, Dict, Union
import numpy as np
import torch
from transforms3d.euler import euler2quat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.sapien_utils import look_at


@register_env("CubeConEnv-v1", max_episode_steps=100)
class CubeConEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", num_envs=1, reconfiguration_freq=None, **kwargs):
        self.cube_half_size = 0.02
        self.bin_radius = 0.08
        self.bin_half_height = 0.03
        self.cube_bin_dis = 0.3
        
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        # Cube (dynamic)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.8, 0.2, 0.2, 1],
            name="cube",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, 0], q=[1, 0, 0, 0])
        )
        
        self.bin = actors.build_cylinder(
            self.scene,
            radius=self.bin_radius,
            half_length=self.bin_half_height,
            color=np.array([0.2, 0.8, 0.2, 0.6]),
            name="bin",
            body_type="static",
            add_collision=True,
            initial_pose=Pose.create_from_pq(p=[0, 0, 0], q=[1, 0, 0, 0])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            cube_pos = torch.zeros((b, 3))
            cube_pos[..., :2] = torch.rand((b, 2)) * 0.3 - 0.2
            cube_pos[..., 2] = self.cube_half_size
            self.cube.set_pose(Pose.create_from_pq(p=cube_pos))
            
            angle = torch.rand((b,)) * np.pi - np.pi / 2
            bin_pos = cube_pos.clone()
            bin_pos[..., 0] += self.cube_bin_dis * torch.cos(angle)
            bin_pos[..., 1] += self.cube_bin_dis * torch.sin(angle)
            bin_pos[..., 2] = self.bin_half_height
            self.bin.set_pose(Pose.create_from_pq(
                p=bin_pos,
                q=euler2quat(0, np.pi / 2, 0)
            ))

    def evaluate(self):
        cube_pos = self.cube.pose.p
        bin_pos = self.bin.pose.p
        dist_xy = torch.linalg.norm(cube_pos[..., :2] - bin_pos[..., :2], dim=1)
        is_inside = (dist_xy < self.bin_radius) & (cube_pos[..., 2] < self.bin_half_height * 2)
        return {"success": is_inside}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_cube_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.cube.pose.p, dim=1
        )
        cube_to_bin_dist = torch.linalg.norm(
            self.cube.pose.p[..., :2] - self.bin.pose.p[..., :2], dim=1
        )
        reach_reward = 1 - torch.tanh(5 * tcp_to_cube_dist)
        place_reward = 1 - torch.tanh(5 * cube_to_bin_dist)
        reward = reach_reward + 2 * place_reward
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 5.0
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[0.6, 0.6, 0.6], target=[0, 0, 0.1]),
            width=512, height=512, fov=np.pi / 3, near=0.01, far=100
        )