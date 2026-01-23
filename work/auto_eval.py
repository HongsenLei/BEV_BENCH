import os
import time
import subprocess
import signal

# --- 路径配置 ---
MODEL_NAME = "get_coff_simple_100_gpu_4_bz_16_spatial_softmax"
CKPT_ROOT = f"/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lhs/code/lerobot_policy_bevvla/outputs/bev/{MODEL_NAME}/checkpoints"
LOG_FILE = f"/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lhs/code/lerobot_policy_bevvla/outputs/bev/{MODEL_NAME}/evaluated_checkpoints.txt"
POLL_INTERVAL = 3000

# --- Conda 环境 Python 路径配置 ---
# 请通过 `conda activate env_name && which python` 获取
SERVER_PYTHON = "/XYAIFS00/HDD_POOL/sysu_grwang/sysu_grwang_1/.local/opt/miniforge3/envs/lhs_bev/bin/python"
CLIENT_PYTHON = "/XYAIFS00/HDD_POOL/sysu_grwang/sysu_grwang_1/.local/opt/miniforge3/envs/vlabench/bin/python"

# 项目路径配置 (请确保这些是文件夹的绝对路径)
SERVER_PROJECT_PATH = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lhs/code/lerobot_policy_bevvla"
CLIENT_PROJECT_PATH = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lhs/code/VLABench"

# --- 显存与负载配置 ---
MIN_FREE_MEMORY = 4000  # 单位 MiB
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

def get_gpu_status():
    """
    获取 GPU 状态
    返回格式: { gpu_id: {'free_mem': int, 'utilization': int} }
    """
    try:
        # 查询项：index, 剩余显存, GPU利用率
        cmd = "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
        
        gpu_status = {}
        for line in output.split('\n'):
            idx, free, util = map(int, line.split(', '))
            if idx in GPU_LIST:
                gpu_status[idx] = {'free_mem': free, 'utilization': util}
        return gpu_status
    except Exception as e:
        print(f"⚠️ Error querying NVIDIA-SMI: {e}")
        return {}

def find_best_gpu():
    """
    策略：
    1. 显存必须大于 MIN_FREE_MEMORY
    2. 在满足条件的显卡中，选择 utilization (利用率) 最低的
    """
    while True:
        status = get_gpu_status()
        candidates = []
        
        for gpu_id, info in status.items():
            if info['free_mem'] >= MIN_FREE_MEMORY:
                candidates.append((gpu_id, info['utilization']))
        
        if candidates:
            # 按利用率升序排序，取第一个
            candidates.sort(key=lambda x: x[1])
            best_gpu = candidates[0][0]
            best_util = candidates[0][1]
            print(f"✅ Best GPU found: ID {best_gpu} (Utilization: {best_util}%, Free Mem: {status[best_gpu]['free_mem']}MiB)")
            return best_gpu
        
        print(f"❌ No GPU meets the memory requirement ({MIN_FREE_MEMORY} MiB). Waiting 60s...")
        time.sleep(60)

def run_evaluation(ckpt_path, ckpt_name):
    # 自动择优
    gpu_id = find_best_gpu()
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🚀 [Eval Start] {ckpt_name} on GPU {gpu_id}")
    
    # 启动 Server
    server_cmd = [
        SERVER_PYTHON, "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lhs/code/lerobot_policy_bevvla/src/web_evaluate/server.py",
        "--dataset.repo_id=lerobot/get_coffee_random_pos_100",
        "--dataset.root=/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lhs/data/datasets/omnid_vlabench_dataset_v_0/get_coff_simple_100_v30",
        f"--policy.path={ckpt_path}/pretrained_model"
    ]
    
    server_proc = subprocess.Popen(server_cmd, env=env, cwd=SERVER_PROJECT_PATH)
    time.sleep(60) # 预留模型加载时间

    # 启动 Client
    client_cmd = [
        CLIENT_PYTHON, "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lhs/code/VLABench/scripts/evaluate_multiviewvla.py",
        "--tasks", "get_coffee",
        "--n-episode", "50",
        "--visulization"
    ]
    
    try:
        subprocess.run(client_cmd, env=env, check=True, cwd=CLIENT_PROJECT_PATH)
        # 记录成功
        with open(LOG_FILE, "a") as f:
            f.write(f"{ckpt_name}\n")
        print(f"✨ [Eval Done] {ckpt_name}")
    except subprocess.CalledProcessError:
        print(f"❌ [Eval Failed] {ckpt_name}")
        
    finally:
        # 释放资源
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print(f"♻️  Resources released for GPU {gpu_id}")

def main():
    print("🛰️  VLA Auto-Evaluation Monitor Started.")
    while True:
        if not os.path.exists(CKPT_ROOT):
            time.sleep(60)
            continue

        # 扫描并排序
        all_ckpts = [d for d in os.listdir(CKPT_ROOT) if os.path.isdir(os.path.join(CKPT_ROOT, d)) and d != "last" ]
        all_ckpts.sort(key=lambda x: int(x) if x.isdigit() else 0, reverse=True)
        
        # 读取已完成列表
        evaluated = set()
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                evaluated = set(line.strip() for line in f)

        for ckpt in all_ckpts:
            if ckpt not in evaluated:
                ckpt_path = os.path.abspath(os.path.join(CKPT_ROOT, ckpt))
                if os.path.exists(os.path.join(ckpt_path, "pretrained_model")):
                    run_evaluation(ckpt_path, ckpt)
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()