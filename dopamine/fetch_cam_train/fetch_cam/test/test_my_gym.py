import fetch_cam

import gym
# env = gym.make("CartPoleKbe-v0")
env = gym.make("FetchCameraEnv-v0")
observation = env.reset()
for i in range(1000):
  
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print('i = ',i, 'done = ', done)
  if done:
      env.reset()
  else:
      env.render()