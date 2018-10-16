import numpy as np
import gym
import time
from matplotlib import pyplot as plt

# env = gym.make('FetchPickAndPlace-v1')
from gym.envs.robotics import FetchPickAndPlaceEnv

env = FetchPickAndPlaceEnv()
obs = env.reset()
done = False

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()

times = 0

# while not done:
while True:
    action = policy(obs['observation'], obs['desired_goal'])
    
    obs, reward, done, info = env.step(action)

    times+=1
    print('obs',obs)
    print('action',action,'done', done,'times', times)
    env.render()
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(
        reward, substitute_reward))


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

    
    # time.sleep(1)