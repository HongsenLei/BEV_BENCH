import os
os.environ["MUJOCO_GL"]= "egl"

import argparse
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.evaluator import MultiViewVLAEvaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy, DummyPolicy
from VLABench.tasks import *
from VLABench.robots import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=None, help="Specific tasks to run, work when eval-track is None")
    parser.add_argument('--eval-track', default=None, type=str, choices=["track_1_in_distribution", "track_2_cross_category", "track_3_common_sense", "track_4_semantic_instruction", "track_6_unseen_texture"], help="The evaluation track to run")
    parser.add_argument('--n-episode', default=1, type=int, help="The number of episodes to evaluate for a task")
    parser.add_argument('--policy', default="lerobotmultiview", help="The policy to evaluate")
    parser.add_argument('--model_ckpt', default="/remote-home1/sdzhang/huggingface/openvla-7b", help="The base model checkpoint path")
    parser.add_argument('--lora_ckpt', default="/remote-home1/pjliu/openvla/weights/vlabench/select_fruit+CSv1+lora/", help="The lora checkpoint path")
    parser.add_argument('--save-dir', default="eval", help="The directory to save the evaluation results")
    parser.add_argument('--visulization', action="store_true", default=False, help="Whether to visualize the episodes")
    parser.add_argument('--metrics', nargs='+', default=["success_rate"], choices=["success_rate", "intention_score", "progress_score"], help="The metrics to evaluate")
    parser.add_argument('--host', default="127.0.0.1", type=str, help="The host to the remote server")
    parser.add_argument('--port', default=10123, type=int, help="The port to the remote server")
    parser.add_argument('--replanstep', default=4, type=int, help="The step to replan")
    parser.add_argument('--observation_images', nargs='+', default=["observation.image_0","observation.image_1","observation.image_2","observation.image_3","observation.image_4", "observation.image_wrist"]  , help="Specific view to run")
    parser.add_argument('--camera-perturbation', type=str, choices=["fix", "small", "medium", "large"], help="Add camera position and pose perturbation in a predefined way")
    args = parser.parse_args()
    return args

def evaluate(args):
    if args.tasks is not None:
        tasks = args.tasks
    assert isinstance(tasks, list)

    if args.observation_images is not None:
        observation_images=args.observation_images
    else:
        raise ValueError("Invalid observation_images, can not be empty!")
    assert isinstance(observation_images, list)

    if args.policy.lower() == "openvla":
        policy = OpenVLA(
            model_ckpt=args.model_ckpt,
            lora_ckpt=args.lora_ckpt,
            norm_config_file=os.path.join(os.getenv("VLABENCH_ROOT"), "configs/model/openvla_config.json") # TODO: re-compuate the norm state by your own dataset
        )
    elif args.policy.lower() == "gr00t":
        from VLABench.evaluation.model.policy.gr00t import Gr00tPolicy
        policy = Gr00tPolicy(host=args.host, port=args.port, replan_steps=args.replanstep)
    elif args.policy.lower() == "openpi":
        from VLABench.evaluation.model.policy.openpi import OpenPiPolicy
        policy = OpenPiPolicy(host=args.host, port=args.port, replan_steps=args.replanstep)
    elif args.policy.lower() == "lerobotmultiview":
        from VLABench.evaluation.model.policy.lerobot_multiview import LerobotMultiviewPolicy
        policy = LerobotMultiviewPolicy(host=args.host, port=args.port, observation_images=observation_images)
    elif args.policy.lower() == "random":
        policy = RandomPolicy(None)
    elif args.policy.lower() == "dummy":
        policy = DummyPolicy(None)
    else:
        raise ValueError("Invalid policy")

    if args.policy.lower()=="dummy" or args.policy.lower()=="random":
        args.save_dir = os.path.join("outputs",args.policy.lower())
    else:
        args.save_dir = os.path.join(os.path.dirname(policy._server_metadata['pretrained_path']), args.save_dir)
    episode_config = None
    # if args.eval_track is not None:
    #     args.save_dir = os.path.join(args.save_dir, args.eval_track)
    #     with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", f"{args.eval_track}.json"), "r") as f:
    #         episode_config = json.load(f)
    #         tasks = list(episode_config.keys())

    # 在这里修改xml_file的路径
    if args.camera_perturbation is not None and args.camera_perturbation != "fix":
        xml_file=f"base/camera_{args.camera_perturbation}_env.xml"
        print(f"Start evaluating camera perturbation {args.camera_perturbation} .......")
    else:
        xml_file = "base/default.xml"
        print(f"Start evaluating default camera setup .......")
        
        
    
    evaluator = MultiViewVLAEvaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_config,
        max_substeps=1, # repeat step in simulation
        save_dir=args.save_dir,
        visulization=args.visulization,
        metrics=args.metrics,
        observation_images=observation_images,
        xml_file = xml_file # 可以控制相机位置
    )
    
    result = evaluator.evaluate(policy)
    # if args.eval_track:
    #     os.makedirs(os.path.join(args.save_dir, args.eval_track), exist_ok=True)
    #     with open(os.path.join(args.save_dir, args.eval_track, "evaluation_result.json"), "w") as f:
    #         json.dump(result, f)

if __name__ == "__main__":
    args = get_args()
    evaluate(args)