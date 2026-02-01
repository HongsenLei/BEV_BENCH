import os
os.environ["MUJOCO_GL"] = "egl"
import json
import numpy as np
from VLABench.envs.dm_env import LM4ManipDMEnv
from VLABench.envs import load_env
from VLABench.tasks import * # 不能删
from VLABench.robots import * # 不能删

def get_robot_bottom(env: LM4ManipDMEnv):
    model = env._physics.model
    data = env._physics.data
    for body_id in range(model.nbody):
        body_name = model.body(body_id).name
        if body_name == "franka/link0":
            return data.xpos[body_id]
    return np.array([0,0,0])

ENV2DATASET_OBSERVATION={
    "observation.image_0":0,
    "observation.image_1":1,
    "observation.image_2":2,
    "observation.image_3":3,
    "observation.image_4":4,
    "observation.image_wrist":5,
} # 转换为lerobot的数据

print("初始化环境...")

env: LM4ManipDMEnv = load_env("get_coffee")
env.reset()
print("初始化环境完成...")
print("获取相机参数...")
obs = env.get_observation()
camera2world = obs["extrinsic"] # 相机坐标系到世界坐标系的变换矩阵 (N, 4, 4)
# print(camera2world.shape) # (6, 4, 4)
instrinsic = obs["instrinsic"] # 注意：原代码变量名可能有拼写错误(instrinsic)，此处沿用
# print(instrinsic.shape) # (6, 3, 3)

robot_frame = get_robot_bottom(env) # 机械臂基座在世界坐标系的位置
print("机器人基座位置:", robot_frame)

# TODO 从相机到机械臂基座坐标系的变换矩阵

# 1. 构建 机械臂基座 -> 世界坐标系 的变换矩阵 (T_robot_to_world)
# 假设基座没有旋转，只有平移 (通常 benchmark 中 base 是对齐世界坐标系的，或者仅有平移)
T_robot_to_world = np.eye(4)
T_robot_to_world[:3, 3] = robot_frame

# 2. 计算 世界坐标系 -> 机械臂基座 的变换矩阵 (T_world_to_robot)
# T_world_to_robot = (T_robot_to_world)^-1
T_world_to_robot = np.linalg.inv(T_robot_to_world)

camera_para = {}

for dataset_key, env_idx in ENV2DATASET_OBSERVATION.items():
    # --- 处理内参 (Intrinsic) ---
    K = instrinsic[env_idx] # (3, 3)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # 推断图像宽高 (通常光心在图像中心: width = cx * 2)
    width = int(cx * 2)
    height = int(cy * 2)
    
    # 计算垂直视场角 (FovY)
    # formula: fovy = 2 * arctan(h / (2 * fy)) * (180 / pi)
    fovy_rad = 2 * np.arctan(height / (2 * fy))
    fovy_deg = np.degrees(fovy_rad)

    intrinsic_dict = {
        "width": width,
        "height": height,
        "fovy_degrees": float(fovy_deg),
        "focal_length": float(fx),
        "matrix": K.tolist()
    }

    # --- 处理外参 (Extrinsic) ---
    # 获取 相机 -> 世界 的变换矩阵 (T_cam_to_world)
    T_cam_to_world = camera2world[env_idx] # (4, 4)
    
    # 提取位置和旋转矩阵 (直接用于 extrinsic 下的 position 和 rotation_matrix)
    pos_in_world = T_cam_to_world[:3, 3]
    rot_mat_in_world = T_cam_to_world[:3, :3]
    
    # 计算 相机 -> 机械臂基座 的变换矩阵 (T_cam_to_robot)
    # T_cam_to_robot = T_world_to_robot * T_cam_to_world
    T_cam_to_robot = np.dot(T_world_to_robot, T_cam_to_world)

    extrinsic_dict = {
        "position": pos_in_world.tolist(),
        "rotation_matrix": rot_mat_in_world.tolist(),
        "matrix": T_cam_to_robot.tolist() # 这里存放的是相机到机械臂基座的变换
    }

    # 组装数据
    camera_para[dataset_key] = {
        "intrinsic": intrinsic_dict,
        "extrinsic": extrinsic_dict
    }

# 保存为 json 文件
output_file = "work/gen_camera_params/small_camera_params.json"
with open(output_file, "w") as f:
    json.dump(camera_para, f, indent=4)

print(f"相机参数已保存至 {output_file}")