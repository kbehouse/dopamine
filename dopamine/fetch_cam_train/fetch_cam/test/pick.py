import numpy as np
import gym
import time
from matplotlib import pyplot as plt

# env = gym.make('FetchPickAndPlace-v1')
from gym.envs.robotics import FetchPickAndPlaceEnv
from fsm import FSM

env = FetchPickAndPlaceEnv()
obs = env.reset()
done = False

GRIPPER_STATE = 1
LIMIT_Z = .415
SCALE_SPEED = 2.0


times = 0


g = GRIPPER_STATE

print("obs['achieved_goal'] = ", obs['achieved_goal'])
goal = obs['achieved_goal'].copy()
goal[-1] = goal[-1] + .1
# kbe: obs['eeinfo'][0] -> (x,y,z), obs['eeinfo'][1] quaternion
simple_policy = FSM(np.append(obs['eeinfo'][0], g), obs['achieved_goal'], goal, LIMIT_Z)
total_reward = 0


ENV_RENDER = True
s_time = time.time()

step = 0
while not simple_policy.done:
    x, y, z, g = simple_policy.execute()
    # scale up action
    a = np.array([x, y, z, g]) * SCALE_SPEED
    print('a = ', a)
    obs, r, done, info = env.step(a)

    print("obs['eeinfo'] = ", obs['eeinfo'])
    # update robot state
    simple_policy.robot_state = np.append(obs['eeinfo'][0], g)
    
    step += 1
    total_reward += r
    
    print('r = ', r)

    print('----------')

    if ENV_RENDER:
        env.render()

        rgb_obs = env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                mode='offscreen', device_id=-1)
        rgb_obs1 = env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
        plt.figure(1)
        plt.imshow(rgb_obs)
        plt.figure(2)
        plt.imshow(rgb_obs1)
        plt.show(block=False)
        plt.pause(0.001)


print('total_reward = ', total_reward)
print('Use time = ', time.time() - s_time)