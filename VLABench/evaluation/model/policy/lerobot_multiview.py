import websockets
import functools
import time
import collections
from typing import Dict, Tuple
import websockets.sync.client
from typing_extensions import override
import msgpack
import numpy as np
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

OBSERVATION = {
    "observation.image_0": 0,
    "observation.image_1": 1,
    "observation.image_2": 2,
    "observation.image_3": 3,
    "observation.image_4": 4,
    "observation.image_wrist": 5,
}  # 转换为lerobot的数据


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def preprocess_image(image):
    """
    convert image to [C, H, W] format from [H, W, C]
    """
    image = np.array(image, dtype=np.float32)
    image_chw = image.transpose(2, 0, 1)
    image_chw = image_chw / 255.0  # normalize to [0,1]
    return image_chw


class LerobotMultiviewPolicy:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10123,
        observation_images=["observation.image_0"],
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()
        print(self._server_metadata)
        self.name = "lerobotmultiview"
        self.control_mode = "ee"
        self.observation_images = observation_images

    def predict(self, observation, **kwargs):
        policy_obs = {}
        ee_state = observation["ee_state"]
        ee_pos, ee_quat, gripper = ee_state[:3], ee_state[3:7], np.array([ee_state[7]])
        ee_pos -= np.array([0, -0.4, 0.78])  # 转换为lerobot的数据
        ee_euler = quaternion_to_euler(ee_quat)
        ee_state = np.concatenate([ee_pos, ee_euler, gripper], axis=0)
        state = np.array(ee_state, dtype=np.float32)
        policy_obs["observation.state"] = state
        for img in self.observation_images:
            policy_obs[img] = np.array(
                preprocess_image(observation["rgb"][OBSERVATION[img]]), dtype=np.float32
            )
        policy_obs["task"] = observation["instruction"]
        raw_action = self.infer(policy_obs)
        target_pos, target_euler, gripper = raw_action[:3], raw_action[3:6], raw_action[-1]
        if gripper >= 0.1:
            gripper_state = np.ones(2)*0.04
        else:
            gripper_state = np.zeros(2)
        target_pos = target_pos.copy()
        target_pos += np.array([0, -0.4, 0.78])
        return target_pos, target_euler, gripper_state

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        print(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, 
                    compression=None, 
                    max_size=None,
                    ping_interval=None
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)
    
    @override
    def reset(self) -> None:
        self._ws.send(b'reset')
        response = self._ws.recv()
        if isinstance(response, bytes) and response == b'reset_complete':
            return
        else:
            raise RuntimeError("Error resetting the server")
