from typing import Any, Dict, Union
import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.sapien_utils import look_at


@register_env("CollectCapsulesEnv-v1", max_episode_steps=100)
class CollectCapsulesEnv(BaseEnv):
    """
    Task: Several capsules are scattered on the table. Collect ONLY blue ones into the bottle on the table.
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", num_envs=1, reconfiguration_freq=None, **kwargs):
        
        # internal width, internal depth, wall height, wall thickness
        self.bin_size = (0.1, 0.1, 0.3, 0.005)
        self.bin_color = [0.6, 0.4, 0.2, 0.9]

        # the bottom center of the bin
        self.bin_center = [0.3, 0, 0]

        # radius, half_length
        self.capsule_size = (0.01, 0.02)
        self.capsule_colors = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
        
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
        
        # Four vertical walls combined to form a bin (with table surface as the bottom)
        bin_width, bin_depth, bin_height, wall_thickness = self.bin_size
        self.bin = []

        # Left wall
        left_wall = actors.build_box(
            scene=self.scene,
            half_sizes=[wall_thickness / 2, bin_depth / 2, bin_height / 2],
            color=self.bin_color,
            name="left_wall",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[-(bin_width / 2 + wall_thickness / 2), 0, bin_height / 2]
            ),
        )
        self.bin.append(left_wall)

        # Right wall
        right_wall = actors.build_box(
            scene=self.scene,
            half_sizes=[wall_thickness / 2, bin_depth / 2, bin_height / 2],
            color=self.bin_color,
            name="right_wall",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[bin_width / 2 + wall_thickness / 2, 0, bin_height / 2]
            ),
        )
        self.bin.append(right_wall)

        # Front wall
        front_wall = actors.build_box(
            scene=self.scene,
            half_sizes=[bin_width / 2, wall_thickness / 2, bin_height / 2],
            color=self.bin_color,
            name="front_wall",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[0, -(bin_depth / 2 + wall_thickness / 2), bin_height / 2]
            ),
        )
        self.bin.append(front_wall)

        # Back wall
        back_wall = actors.build_box(
            scene=self.scene,
            half_sizes=[bin_width / 2, wall_thickness / 2, bin_height / 2],
            color=self.bin_color,
            name="back_wall",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[0, bin_depth / 2 + wall_thickness / 2, bin_height / 2]
            ),
        )
        self.bin.append(back_wall)

        self.capsules = []
        capsule_radius, capsule_half_length = self.capsule_size

        for i in range(3):
            builder = self.scene.create_actor_builder()
            builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))

            builder.add_capsule_collision(
                radius=capsule_radius,
                half_length=capsule_half_length,
                density=500, 
            )
            
            builder.add_capsule_visual(
                radius=capsule_radius,
                half_length=capsule_half_length,
                material=sapien.render.RenderMaterial(base_color=self.capsule_colors[i]),
            )
            
            capsule = builder.build(name=f"capsule_{i}")
            self.capsules.append(capsule)

        # WIP: How to use articulations?

        # builder = self.scene.create_articulation_builder()

        # base = builder.create_link_builder()
        # base.set_name("bin_base")
        # base.add_box_collision(half_size=[bin_width/2, bin_depth/2, 0.01])
        # base.add_box_visual(half_size=[bin_width/2, bin_depth/2, 0.01], 
        #                 material=[0.3, 0.3, 0.3, 1.0])
        
        # lid = builder.create_link_builder(base)  # Child of base
        # lid.set_name("lid")
        # lid.add_box_collision(half_size=[bin_width/2, bin_depth/2, 0.01])
        # lid.add_box_visual(half_size=[bin_width/2, bin_depth/2, 0.01],
        #                 material=[0.7, 0.7, 0.7, 0.8])  # Semi-transparent
        
        # lid.set_joint_properties(
        #     type="prismatic",
        #     limits=[[-bin_width, 0]],  # Slide from fully open (-width) to closed (0)
        #     pose_in_parent=sapien.Pose(
        #         p=[0, 0, bin_height + 0.01],  # Above bin walls
        #         q=[1, 0, 0, 0]
        #     ),
        #     pose_in_child=sapien.Pose(
        #         p=[bin_width/2, 0, 0],  # Joint at lid's left edge
        #         q=[1, 0, 0, 0]
        #     ),
        #     friction=0.1,  # Prevents infinite sliding
        #     damping=1.0    # Adds smooth deceleration
        # )
        
        # self.bin_articulation = builder.build(name="bin_lid")
        # self.lid = self.bin_articulation.get_links()[1]  # Get lid link

        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Move the bin to the front of the robot by 0.3 meters
            for bin_part in self.bin:
                bin_part_pos = bin_part.pose.p + torch.tensor(self.bin_center)
                bin_part.set_pose(Pose.create_from_pq(p=bin_part_pos))

            bin_width, bin_depth, bin_height, wall_thickness = self.bin_size
            bin_center_x = self.bin_center[0] * torch.ones(b,)
            capsule_radius, capsule_half_length = self.capsule_size
            
            # compute a safe distance from the bin center
            min_distance = 2 * capsule_half_length + 0.02
            max_distance = bin_center_x - max(bin_width, bin_depth) / 2  \
                                        - 2 * capsule_half_length  \
                                        - wall_thickness \
                                        - 0.02
            
            # 60°, 180°, 300° - avoid spawning inside the bin
            # Consider different envs: need to do a separate shuffle for every env
            angles = torch.stack([torch.tensor([np.pi/3, np.pi, 5*np.pi/3])[
                torch.randperm(3)] for _ in range(b)])
            
            distances = min_distance + (max_distance.unsqueeze(-1) - min_distance) * torch.rand((b, 3))
            
            for i, capsule in enumerate(self.capsules):
                capsule_pos = torch.zeros((b, 3))
                capsule_pos[..., 0] = distances[:, i] * torch.cos(angles[:, i])
                capsule_pos[..., 1] = distances[:, i] * torch.sin(angles[:, i])
                capsule_pos[..., 2] = capsule_radius
                
                # introduce noise on location
                capsule_pos[..., :2] += (torch.rand((b, 2))) * 0.04 - 0.02

                # introduce noise on rotation                
                random_yaw = torch.rand((b,)) * 2 * np.pi
                random_pitch = (torch.rand((b,))) * 0.2 - 0.1
                
                capsule.set_pose(Pose.create_from_pq(
                    p=capsule_pos,
                    q=euler2quat(random_pitch, torch.zeros(b,), random_yaw)
                ))
                
                capsule.set_linear_velocity(torch.zeros((b, 3)))
                capsule.set_angular_velocity(torch.zeros((b, 3)))

                                

    def evaluate(self):
        with torch.device(self.device):
            bin_width, bin_depth, bin_height, wall_thickness = self.bin_size

            pos_x, pos_y, pos_z = self.bin_center
            bin_min = torch.tensor([
                pos_x - bin_width / 2 - wall_thickness,
                pos_y - bin_depth / 2 - wall_thickness,
                pos_z
            ])
            bin_max = torch.tensor([
                pos_x + bin_width / 2 + wall_thickness,
                pos_y + bin_depth / 2 + wall_thickness,
                pos_z + bin_height
            ])
            
            blue_status = []
            red_status = None
            
            for i, capsule in enumerate(self.capsules):
                pos = capsule.pose.p  # [b, 3]
                in_bin = torch.all(
                    (pos >= bin_min) & (pos <= bin_max),
                    dim=1
                )
                if i == 0:  # Red capsule
                    red_status = in_bin
                else:  # blue capsules
                    blue_status.append(in_bin)
            
            # Success <=> All blue capsules in bin (AND) Red capsule NOT in bin
            all_blue_in = torch.all(torch.stack(blue_status), dim=0)
            success = all_blue_in & (~red_status)
            
            return {
                "success": success,
                "blue_in_bin": torch.stack(blue_status),    # [2, b]
                "red_in_bin": red_status.unsqueeze(0)       # [1, b]
            }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs)
            bin_center = torch.tensor(self.bin_center)
            
            # Movement rewards for blue capsules
            for i in [1, 2]:  # blue capsules
                dist = torch.linalg.norm(
                    self.capsules[i].pose.p[..., :2] - bin_center[:2],
                    dim=1
                )
                reward += (1 - torch.tanh(5 * dist)) * 0.3
                
            # Collection bonus for blue capsules in bin
            blue_collected = torch.sum(info["blue_in_bin"].float(), dim=0)
            reward += blue_collected * 2.0
            
            # Penalty for red capsule in bin
            reward -= torch.sum(info["red_in_bin"].float(), dim=0) * 2.0 
            
            # Success bonus (override previous rewards)
            reward[info["success"]] = 10.0
            
            return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 10.0
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[0.6, 0.6, 0.6], target=[0, 0, 0.1]),
            width=512, height=512, fov=np.pi / 3, near=0.01, far=100
        )