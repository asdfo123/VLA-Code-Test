import os
import gymnasium as gym
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
import matplotlib.pyplot as plt
from robot_patrol import PatrolRoomEnv

def run_patrol_episodes(num_episodes=5, save_dir="results", max_steps_per_episode=100):
    """
    Run multiple episodes of the patrol task with improved Fetch robot movement.
    """
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make(
        "PatrolRoom-v1", 
        robot_uids="fetch",
        obs_mode="state_dict", 
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
    )
    env = RecordEpisode(
        env,
        output_dir=save_dir,
        save_video=True,
        save_trajectory=True,
        max_steps_per_video=max_steps_per_episode
    )
    
    for episode in range(num_episodes):
        print(f"Running episode {episode+1}/{num_episodes}")
        obs, info = env.reset(seed=episode, options=dict(reconfigure=True))
        
        # Print the initial target area
        target_area_id = info["target_area_id"].item()
        debris_area_id = info["debris_area_id"].item()
        print(f"Episode {episode+1}: Target area {target_area_id}, Debris area {debris_area_id}")
        
        episode_data = {
            "target_area_id": target_area_id,
            "debris_area_id": debris_area_id,
            "rewards": [],
            "at_target_area": [],
            "inspection_complete": False,
            "steps": 0,
        }
        
        done = False
        truncated = False
        while not (done or truncated) and episode_data["steps"] < max_steps_per_episode:
            # Get the current robot position and target area position
            robot_pos = obs["extra"]["tcp_pose"][0][:3]  # TCP pose (tool center point)
            target_area_pos = obs["extra"]["target_area_pos"][0]
            
            # Calculate direction and distance to target
            direction = target_area_pos[:2] - robot_pos[:2]
            distance = np.linalg.norm(direction)
            if distance > 0.01:
                direction = direction / distance
            action = np.zeros(env.action_space.shape[0])
            
            # Set base movement commands
            speed = 0.5
            if distance > 0.1:  # Only move if not close to target
                # For ManiSkill Fetch under "pd_ee_delta_pose" control, 
                # action[10] is delta x (linear movement) and action[11] is delta theta (angular movement)
                action[10] = direction[0] * speed  # Linear movement in x direction
                action[11] = np.arctan2(direction[1], direction[0])  # Angular movement (steering angle)
            
            # OR use random action as dummy input
            # action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            episode_data["rewards"].append(reward)
            episode_data["at_target_area"].append(info["at_target_area"].item())
            episode_data["inspection_complete"] = info["inspection_complete"].item()
            episode_data["steps"] += 1
            
            if episode_data["steps"] % 100 == 0:
                print(f"Step {episode_data['steps']}, Distance to target: {distance:.2f}, Reward: {reward.item():.4f}")
            
            # Log when the robot reaches the target area
            if info["at_target_area"].item() and not episode_data["inspection_complete"]:
                print(f"Robot reached target area {target_area_id}")
                debris_found = info["debris_found"].item()
                print(f"Debris found: {debris_found}")
                inspection_report = {
                    "target_area_id": target_area_id,
                    "debris_found": debris_found,
                    "correct_assessment": (debris_found == (target_area_id == debris_area_id))
                }
                inspection_report_path = os.path.join(save_dir, f"inspection_report_episode_{episode+1}.txt")
                with open(inspection_report_path, "w") as f:
                    for key, value in inspection_report.items():
                        f.write(f"{key}: {value}\n")
                print(f"Inspection report saved to {inspection_report_path}")
        
        print(f"Episode {episode+1} completed in {episode_data['steps']} steps")
        print(f"Average reward: {np.mean(episode_data['rewards']):.4f}")
        print(f"Inspection complete: {episode_data['inspection_complete']}")

if __name__ == "__main__":
    run_patrol_episodes(num_episodes=5, save_dir="results", max_steps_per_episode=1000)