import numpy as np
import gym
import time
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from fetch_cam import FetchDiscreteCamEnv
import cv2
from PIL import Image
import os, shutil
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

def create_dir(dir_name):
    if os.path.exists(dir_name):
       shutil.rmtree(dir_name) 
    os.makedirs(dir_name)



class PickRobot:
    def __init__(self,is_render):
        
        self.env = FetchDiscreteCamEnv(is_render=is_render, dis_tolerance = 0.001, step_ds=0.005, gray_img=False, only_show_obj0 = False)
        self.is_render = is_render
        self.step_ds = 0.005

    def reset(self):
        obs = self.env.reset()
        self.s_time = time.time()
        self.step_count = 0
        self.sum_r = 0

    def go_target_pos(self, target_pos_x, target_pos_y):
        while True:
            if self.is_render:
                self.env.render()
            diff_x = target_pos_x - self.env.pos[0] #self.env.obj_pos[0] - self.env.pos[0]
            diff_y = target_pos_y - self.env.pos[1] # self.env.obj_pos[1] - self.env.pos[1]
            if diff_x > self.step_ds:
                a = 0 # [1, 0, 0, 0, 0]
            elif diff_x < 0 and abs(diff_x) >  self.step_ds:
                a = 2 # [0, 0 , 1, 0, 0]
            elif diff_y > self.step_ds:
                a = 1 # [0, 1, 0, 0, 0]
            elif diff_y < 0 and abs(diff_y) >  self.step_ds:
                a = 3 # [0, 0 , 0, 1, 0]
            else:
                break
            self.step_count +=1
            s,r, d, info =  self.env.step(a)
            self.sum_r += r  
            print('r = {}, sum_r={}'.format(r, self.sum_r))
            # print('gripper_state = ', self.env.gripper_state, ',is_gripper_close=', self.env.is_gripper_close)
            
            rgb_img = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.save_dir + '/%03d.jpg' % self.step_count, rgb_img)

    def pick_place(self, ep = 1, noise = False):
        
        step_ds = 0.005
        

        # ori_pos = (obs['eeinfo'][0]).copy()
        print('---ori_pos = ' , self.env.pos,'----')
        

        # for i in range(20):
        
        # self.env.gripper_close(False)
        self.env.render()
        self.save_dir = 'tmp/camself.env_run_pic_%02d' %ep 
        create_dir(self.save_dir)
        self.step_count = 0
        print('------start ep %03d--------' % ep)
        sum_r = 0

        noise_x = 0.00 if noise else 0.0
        noise_y = 0.04 if noise else 0.0

        # ----pick object----
        print('-------before pick--------------')
        target_pos_x = self.env.obj_pos[0] + noise_x
        target_pos_y = self.env.obj_pos[1] + noise_y
        self.go_target_pos(target_pos_x, target_pos_y)
            
        # close gripper and up
        a = 4 # [0, 0, 0, 0, 1]
        s,r, d, info =  self.env.step(a)
        # print('r = {}, sum_r={}'.format(r, self.sum_r))
            
        print('-------after pick--------------')
        self.sum_r += r
        # ----place object----
        target_pos_x = self.env.red_tray_pos[0] + noise_x
        target_pos_y = self.env.red_tray_pos[1] + noise_y
        self.go_target_pos(target_pos_x, target_pos_y)
        
        # open gripper and up
        a = 5 # [0, 0, 0, 0, 1]
        s,r, d, info =  self.env.step(a)
        # print('r = {}, sum_r={}'.format(r, self.sum_r))
            
        self.sum_r += r  
        if self.sum_r <=0:
            print("!!!!!!!!!!!!!!!!!!!!!! strange")
        print('sum_r = ', self.sum_r)
        print("use step = ", self.step_count)

        self.env.render()
            

        print('use time = {:.2f}'.format(time.time()-self.s_time))

if os.path.exists('tmp/'):
    shutil.rmtree('tmp/') 

pickbot = PickRobot(is_render = False)

for _ in range(10):
    pickbot.reset()
    pickbot.pick_place()
