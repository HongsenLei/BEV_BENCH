import os
import json
import numpy as np
import time
import random
import mediapy
from VLABench.envs import load_env

from VLABench.evaluation.utils import *
from VLABench.evaluation.evaluator.base import Evaluator
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

import warnings

warnings.filterwarnings("ignore")
from colorama import Fore, Back, Style, init

init(autoreset=True)


class MultiViewVLAEvaluator(Evaluator):
    def __init__(
        self,
        tasks,
        n_episodes,
        episode_config=None,
        max_substeps=1,
        tolerance=1e-2,
        metrics=["success_rate"],
        save_dir=None,
        visulization=False,
        eval_unseen=False,
        unnorm_key="primitive",
        observation_images=None,
        **kwargs,
    ):
        """
        MultiViewVLA evaluator
        """
        super().__init__(
            tasks,
            n_episodes,
            episode_config,
            max_substeps,
            tolerance,
            metrics,
            save_dir,
            visulization,
            eval_unseen,
            unnorm_key,
            **kwargs,
        )
        
        self.observation_images = observation_images
    
    def evaluate_single_episode(self, agent, task_name, episode_id, episode_config, seed=42, max_episode_length=300, **kwargs):
        """
        If episode_config is given, the task and scene will load deterministically.
        params:
            agent: policy to evaluate
            task_name: name of the task
            episode_id: id of the episode
            episode_config: configuration of the task
            seed: seed for the random number generator, if episode_config is None
            max_episode_length: maximum length of the episode
        """
        if episode_config is None: # use random seed to ditermine the task
            np.random.seed(seed)
            random.seed(seed)
        if episode_config is not None:
            env = load_env(task_name, episode_config=episode_config, random_init=False, eval=self.eval_unseen, run_mode="eval")
        else:
            env = load_env(task_name, random_init=True, eval=self.eval_unseen, run_mode="eval")
        env.reset()
        success = False
        info = {}
        frames_to_save = []
        last_action = None
        i = 0
        robot_frame = env.get_robot_frame_position()
        while i < max_episode_length:
            observation = env.get_observation(require_pcd=False)
            observation["instruction"] = env.task.get_instruction()
            ee_state = observation["ee_state"]
            observation['robot_frame'] = robot_frame
            if last_action is None:
                last_action = np.concatenate([ee_state[:3], quaternion_to_euler(ee_state[3:7])])
            observation["last_action"] = last_action
            if self.save_dir is not None and self.visulization:
                frame_width=observation["rgb"][0].shape[1]
                frame_height=observation["rgb"][0].shape[0]
                # 将所有帧堆叠到一个大图像中
                num_cols = 2
                num_rows = 3
                stacked_image = np.zeros((frame_height * num_rows, frame_width * num_cols, 3), dtype=np.uint8)
                for frame_i, frame in enumerate(observation["rgb"]):
                    row = frame_i // num_cols
                    col = frame_i % num_cols
                    y1 = row * frame_height
                    y2 = y1 + frame_height
                    x1 = col * frame_width
                    x2 = x1 + frame_width
                    stacked_image[y1:y2, x1:x2] = frame               
                # 写入堆叠后的帧
                frames_to_save.append(stacked_image)
            if agent.control_mode == "ee":
                pos, euler, gripper_state = agent.predict(observation, **kwargs)
                last_action = np.concatenate([pos, euler])
                quat = euler_to_quaternion(*euler)
                _, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)
                action = np.concatenate([action, gripper_state])
            elif agent.control_mode == "joint":
                qpos, gripper_state = agent.predict(observation, **kwargs)
                action = np.concatenate([qpos, gripper_state])
            else:
                raise NotImplementedError(f"Control mode {agent.control_mode} is not implemented")    
            for _ in range(self.max_substeps):
                timestep = env.step(action)
                if timestep.last():
                    success=True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos-np.array(action)[:7]) < self.tolerance \
                    and np.min(current_qpos - np.array(action)[:7]) > -self.tolerance:
                    break
            if success:
                break
            i += 1
        # intention_score =  env.get_intention_score(threshold=self.intention_score_threshold)
        # progress_score = env.get_task_progress()
        info["task"] = task_name
        info["success"] = success
        info["consumed_step"] = i
        # info["intention_score"] = intention_score
        # info["progress_score"] = progress_score
        
        env.close()
        if self.save_dir is not None and self.visulization:
            os.makedirs(os.path.join(self.save_dir, task_name, "videos"), exist_ok=True)
            self.save_video(frames_to_save, os.path.join(self.save_dir, task_name, "videos", f"{episode_id}_{str(success)}.mp4"))
        return info

    def save_video(self, frames, save_dir):
        if len(frames) == 0:
            return
        mediapy.write_video(save_dir, frames, fps=10)