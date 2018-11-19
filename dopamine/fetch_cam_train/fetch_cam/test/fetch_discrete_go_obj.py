import numpy as np
import gym
import time
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from fetch_cam.fetch_discrete_cam import FetchDiscreteCamEnv, IMG_TYPE
from fsm import FSM
import cv2
from PIL import Image
import os, shutil
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

def create_dir(dir_name):
    if os.path.exists(dir_name):
       shutil.rmtree(dir_name) 
    os.makedirs(dir_name)


def to_digit(img, path):
    fo = open(path, "w")
    for i in range(img.shape[0]):
        fo.write('%02d:' % i )
        for j in range(img.shape[1]):
            # fo.write('(%03d,%03d,%03d) ' % ( img[i][j] )
            fo.write('{:2d}'.format(img[i][j]) )

        fo.write('\n' )
    fo.close()

def go_obj_savepic_with_camenv(is_render = True, img_type = IMG_TYPE.GRAY, noise=False, only_show_obj0 = False):
    if os.path.exists('tmp/'):
        shutil.rmtree('tmp/') 
    
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteCamEnv(dis_tolerance = 0.001, step_ds=0.005, img_type=img_type,use_tray=False, is_render=is_render, only_show_obj0 = only_show_obj0)
    # obs = env.reset()
    # done = False

    # ori_pos = (obs['eeinfo'][0]).copy()
    print('---ori_pos = ' , env.pos,'----')
    # step  = 0
    # robot_step = 0
    # # env.render()
    s_time = time.time()
    # env.render()
    # step_count = 0
    

    for i in range(20):
        obs = env.reset()
        # env.gripper_close(False)
        env.render()
        save_dir = 'tmp/camenv_run_pic_%02d' %i 
        create_dir(save_dir)
        step_count = 0
        print('------start ep %03d--------' % i)
        sum_r = 0

        noise_x = 0.00 if noise else 0.0
        noise_y = 0.04 if noise else 0.0

        target_pos_x = env.obj_pos[0] + noise_x
        target_pos_y = env.obj_pos[1] + noise_y
        while True:
            if is_render:
                env.render()
            diff_x = target_pos_x - env.pos[0] #env.obj_pos[0] - env.pos[0]
            diff_y = target_pos_y - env.pos[1] # env.obj_pos[1] - env.pos[1]
            if diff_x > step_ds:
                a = 0 # [1, 0, 0, 0, 0]
            elif diff_x < 0 and abs(diff_x) >  step_ds:
                a = 2 # [0, 0 , 1, 0, 0]
            elif diff_y > step_ds:
                a = 1 # [0, 1, 0, 0, 0]
            elif diff_y < 0 and abs(diff_y) >  step_ds:
                a = 3 # [0, 0 , 0, 1, 0]
            else:
                break
            step_count +=1
            s,r, d, info =  env.step(0)
            # print('r = ', r)
            sum_r += r  
            
            # print('env.obj_pos = ', env.obj_pos)
            
            if img_type==IMG_TYPE.BIN:
                to_digit(s,save_dir + '/%03d.txt' % step_count )
            if img_type==IMG_TYPE.BIN or img_type==IMG_TYPE.SEMANTIC:
                s=s*255.0
                
            cv2.imwrite(save_dir + '/%03d.jpg' % step_count, s)

        a = 4 # [0, 0, 0, 0, 1]
        # s,r, d, info =  env.step(a)
        s,r, d, info =  env.step(a)
        sum_r += r  
        print('sum_r = ', sum_r)
        print("use step = ", step_count)

        env.render()
        

    print('use time = {:.2f}'.format(time.time()-s_time))


go_obj_savepic_with_camenv(img_type=IMG_TYPE.RAW, noise=False, is_render=True, only_show_obj0 = True)