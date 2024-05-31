import os

from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.registration import register

from RLEnvs.MyFetchRobot.FetchEnv import MujocoFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push.xml")


class MujocoFetchPushEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description
    """

    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            tgt_obj_pos=[0.1, 0],
            tgt_obj_yaw=0.785,
            random_goal=False,
            random_init=False,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


register(
    id="MyFetchRobot/Push-Jnt-Sparse-v0",
    entry_point="RLEnvs.MyFetchRobot.push:MujocoFetchPushEnv",
    max_episode_steps=200,
    kwargs={"reward_type": "sparse", "init_x_pos": 1.2955407},
)
