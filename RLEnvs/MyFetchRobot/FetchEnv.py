import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
import transformations as tft

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


def goal_distance(goal_a, goal_b, goal_type="pos"):
    assert goal_a.shape == goal_b.shape
    pos_dis = np.linalg.norm(goal_a - goal_b, axis=-1)
    if goal_type == "rot":
        quat_diff = tft.quaternion_multiply(goal_b, tft.quaternion_conjugate(goal_a))
        yaw_diff = tft.euler_from_quaternion(quat_diff)[2]
        rot_dis = np.abs(yaw_diff)
        return pos_dis + rot_dis
    elif goal_type == "pos":
        return pos_dis


def get_base_fetch_env(RobotEnvClass: MujocoRobotEnv):
    """Factory function that returns a BaseFetchEnv class that inherits
    from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings.
    """

    class BaseFetchEnv(RobotEnvClass):
        """Superclass for all Fetch environments."""

        def __init__(
                self,
                gripper_extra_height,
                block_gripper,
                has_object: bool,
                target_in_the_air,
                target_offset,
                obj_range,
                target_range,
                distance_threshold,
                reward_type,
                goal_type="pos",  # "pos" or "rot"
                tgt_obj_pos=[0.5, 0.5],
                tgt_obj_yaw=0.785,
                random_goal=True,
                random_init=True,
                init_x_pos=1.3455407,
                **kwargs
        ):
            """Initializes a new Fetch environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                gripper_extra_height (float): additional height above the table when positioning the gripper
                block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
                has_object (boolean): whether or not the environment has an object
                target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
                target_offset (float or array with 3 elements): offset of the target
                obj_range (float): range of a uniform distribution for sampling initial object positions
                target_range (float): range of a uniform distribution for sampling a target
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            """

            self.goal_type = goal_type
            self.tgt_obj_pos = tgt_obj_pos
            self.tgt_obj_yaw = tgt_obj_yaw
            self.random_goal = random_goal
            self.random_init = random_init
            self.init_x_pos = init_x_pos

            self.gripper_extra_height = gripper_extra_height
            self.block_gripper = block_gripper
            self.has_object = has_object
            self.target_in_the_air = target_in_the_air
            self.target_offset = target_offset
            self.obj_range = obj_range
            self.target_range = target_range
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type

            self.use_joint_obs = True  # enable joint space observation
            self.use_joint_act = False  # enable joint space action

            if not self.use_joint_act:
                super().__init__(n_actions=4, **kwargs)
            else:
                super().__init__(n_actions=8, **kwargs)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            # Compute distance between goal and the achieved goal.
            d = goal_distance(achieved_goal, goal, self.goal_type)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            if not self.use_joint_act:
                assert action.shape == (4,)
                action = action.copy()  # ensure that we don't change the action outside of this scope
                pos_ctrl, gripper_ctrl = action[:3], action[3]

                pos_ctrl *= 0.05  # limit maximum change in position
                rot_ctrl = [
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                ]  # fixed rotation of the end effector, expressed as a quaternion
                gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
                assert gripper_ctrl.shape == (2,)
                if self.block_gripper:
                    gripper_ctrl = np.zeros_like(gripper_ctrl)
                action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
            else:
                assert action.shape == (8,)  # 7 arm joints + 1 gripper displacement
                action = action.copy()
            return action

        def _get_obs(self):
            if not self.use_joint_obs:
                (
                    grip_pos,
                    object_pos,
                    object_rel_pos,
                    gripper_state,
                    object_rot,
                    object_velp,
                    object_velr,
                    grip_velp,
                    gripper_vel,
                ) = self.generate_mujoco_observations(rot_type="euler" if self.goal_type == "pos" else "quat")
            else:
                (
                    grip_pos,
                    object_pos,
                    object_rel_pos,
                    gripper_state,
                    object_rot,
                    object_velp,
                    object_velr,
                    grip_velp,
                    gripper_vel,
                    joint_pos,
                    joint_vel,
                ) = self.generate_mujoco_observations(rot_type="euler" if self.goal_type == "pos" else "quat")

            if not self.has_object:
                achieved_goal = grip_pos.copy()
            else:
                if self.goal_type == "pos":
                    achieved_goal = np.squeeze(object_pos.copy())
                elif self.goal_type == "rot":
                    achieved_goal = np.squeeze(np.concatenate([object_pos.copy(), object_rot.copy()]))

            obs = np.concatenate(
                [
                    grip_pos,
                    object_pos.ravel(),
                    object_rel_pos.ravel(),
                    gripper_state,
                    object_rot.ravel(),
                    object_velp.ravel(),
                    object_velr.ravel(),
                    grip_velp,
                    gripper_vel,
                ]
            )
            if self.use_joint_obs:
                obs = np.concatenate([obs, joint_pos, joint_vel])

            return {
                "observation": obs.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.copy(),
            }

        def generate_mujoco_observations(self):
            raise NotImplementedError

        def _get_gripper_xpos(self):
            raise NotImplementedError

        def _sample_goal(self):
            if self.has_object:  # push or pick
                if self.random_goal:
                    goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                        -self.target_range, self.target_range, size=3
                    )
                    goal += self.target_offset
                    goal[2] = self.height_offset
                    if self.target_in_the_air and self.np_random.uniform() < 0.5:
                        goal[2] += self.np_random.uniform(0, 0.45)
                    if self.goal_type == "rot":
                        rand_yaw = self.np_random.uniform(0, 2 * np.pi)
                        quat = tft.quaternion_from_euler(0, 0, rand_yaw)
                        goal = np.concatenate([goal, quat])
                else:
                    goal = self.initial_gripper_xpos[:3] + [self.tgt_obj_pos[0], self.tgt_obj_pos[1], 0]
                    goal += self.target_offset
                    goal[2] = self.height_offset
                    if self.goal_type == "rot":
                        quat = tft.quaternion_from_euler(0, 0, self.tgt_obj_yaw)
                        goal = np.concatenate([goal, quat])
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
            return goal.copy()

        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal, self.goal_type)
            return (d < self.distance_threshold).astype(np.float32)

    return BaseFetchEnv


class MujocoFetchEnv(get_base_fetch_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        if self.block_gripper:
            self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", 0.0)
            self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", 0.0)
            self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self, rot_type="euler"):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt  # transnatial velocity
        """joint names:
            'robot0:slide0'
            'robot0:slide1'
            'robot0:slide2' 
            'robot0:torso_lift_joint',
            'robot0:head_pan_joint',
            'robot0:head_tilt_joint',
            'robot0:shoulder_pan_joint',
            'robot0:shoulder_lift_joint',
            'robot0:upperarm_roll_joint',
            'robot0:elbow_flex_joint',
            'robot0:forearm_roll_joint',
            'robot0:wrist_flex_joint',
            'robot0:wrist_roll_joint',
            'robot0:r_gripper_finger_joint',
            'robot0:l_gripper_finger_joint'
        """
        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.model, self.data, self._model_names.joint_names)
        if self.has_object:
            object_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            if rot_type == "euler":
                object_rot = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "object0"))
            elif rot_type == "quat":
                m = self._utils.get_site_xmat(self.model, self.data, "object0")
                rot_mat = np.zeros((4, 4))
                rot_mat[:3, :3] = m
                rot_mat[3, 3] = 1
                object_rot = tft.quaternion_from_matrix(rot_mat)
            object_velp = self._utils.get_site_xvelp(self.model, self.data, "object0") * dt
            object_velr = self._utils.get_site_xvelr(self.model, self.data, "object0") * dt  # rotational velocity
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if self.use_joint_obs:
            joint_pos = robot_qpos[6:-2]
            joint_vel = robot_qvel[6:-2]

        ret = (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )
        if self.use_joint_obs:
            ret = ret + (joint_pos, joint_vel)
        return ret

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        """site include robot0:grip, target0 and object0"""
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0")
        self.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            if self.random_init:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.2:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                        -self.obj_range, self.obj_range, size=2
                    )
            else:
                object_xpos = [self.init_x_pos, 0.74902433]
            object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            if self.goal_type == "rot":
                rand_yaw = self.np_random.uniform(0, 2 * np.pi)
                quat = tft.quaternion_from_euler(0, 0, rand_yaw)
                object_qpos[3:] = quat
            self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        )
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", gripper_rotation)
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(self.model, self.data, "object0")[2]
