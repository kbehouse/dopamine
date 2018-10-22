import numpy as np
import gym
import time
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from fetch_cam import FetchDiscreteEnv
# from fetch_cam import FetchCameraEnv
from fetch_cam import FetchDiscreteCamEnv
from fetch_cam.fetch_discrete_cam_siamese import FetchDiscreteCamSiamenseEnv
from fsm import FSM
import cv2
from PIL import Image
import os, shutil
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

def create_dir(dir_name):
    if os.path.exists(dir_name):
       shutil.rmtree(dir_name) 
    os.makedirs(dir_name)

def go_obj():
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
    obs = env.reset()
    
    done = False

    ori_pos = (obs['eeinfo'][0]).copy()
    print('---ori_pos = ' , obs['eeinfo'][0],'----')
    step  = 0
    robot_step = 0
    # env.render()
    s_time = time.time()
    sum_r = 0

    while True:
        env.render()
        diff_x = env.obj_pos[0] - env.pos[0]
        diff_y = env.obj_pos[1] - env.pos[1]
        if diff_x > step_ds:
            a = [1, 0, 0, 0, 0]
        elif diff_x < 0 and abs(diff_x) >  step_ds:
            a = [0, 0 , 1, 0, 0]
        elif diff_y > step_ds:
            a = [0, 1, 0, 0, 0]
        elif diff_y < 0 and abs(diff_y) >  step_ds:
            a = [0, 0 , 0, 1, 0]
        else:
            break
        
        s,r, d, info =  env.step(a) 
        sum_r += r
        
    

    a = [0, 0, 0, 0, 1]
    s,r, d, info =  env.step(a)
    sum_r +=r
    env.render()
    print('epsoide sum_r = ', sum_r)

    print('---final_pos = ' , obs['eeinfo'][0],'----')
    pos_diff = obs['eeinfo'][0] - ori_pos
    formattedList = ["%.2f" % member for member in pos_diff]
    print('---pos_diff = ' ,formattedList ,'----')

    print('use time = {:.2f}'.format(time.time()-s_time))




def go_obj_savepic(is_render = True):
    save_dir = 'z_fetch_run_pic'
    create_dir(save_dir)
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
    # obs = env.reset()
    # done = False

    # ori_pos = (obs['eeinfo'][0]).copy()
    print('---ori_pos = ' , env.pos,'----')
    render# step  = 0
    # robot_step = 0
    # # env.render()
    s_time = time.time()
    # env.render()
    # step_count = 0

    for i in range(5):
        obs = env.reset()
        env.gripper_close(False)
        env.render()
        save_dir = 'z_fetch_run_pic_%02d' %i 
        create_dir(save_dir)
        step_count = 0
        print('------start ep %03d--------' % i)
        sum_r = 0
        while True:
            if is_render:
                env.render()
            diff_x = env.obj_pos[0] - env.pos[0]
            diff_y = env.obj_pos[1] - env.pos[1]
            if diff_x > step_ds:
                a = [1, 0, 0, 0, 0]
            elif diff_x < 0 and abs(diff_x) >  step_ds:
                a = [0, 0 , 1, 0, 0]
            elif diff_y > step_ds:
                a = [0, 1, 0, 0, 0]
            elif diff_y < 0 and abs(diff_y) >  step_ds:
                a = [0, 0 , 0, 1, 0]
            else:
                break
            step_count +=1
            s,r, d, info =  env.step(a)
            sum_r += r  
            # rgb_external = env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
            #         mode='offscreen', device_id=-1)
            # rgb_gripper = env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            #     mode='offscreen', device_id=-1)
            rgb_external = env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
            rgb_gripper = env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
                mode='offscreen', device_id=-1)

            # print('type(rgb_gripper) = ', type(rgb_gripper),', shape=', np.shape(rgb_gripper))
            img = Image.fromarray(rgb_gripper, 'RGB')
            # img.save(save_dir + '/%03d.jpg' % step_count)
            img.save(save_dir + '/%03d_r%3.2f.jpg' % (step_count,r ))
        


        a = [0, 0, 0, 0, 1]
        s,r, d, info =  env.step(a)
        sum_r += r  
        print('sum_r = ', sum_r)
        print("use step = ", step_count)

        env.render()


    # print('---final_pos = ' , obs['eeinfo'][0],'----')
    # pos_diff = obs['eeinfo'][0] - ori_pos
    # formattedList = ["%.2f" % member for member in pos_diff]
    # print('---pos_diff = ' ,formattedList ,'----')

    print('use time = {:.2f}'.format(time.time()-s_time))




def go_obj_savepic_with_camenv(is_render = True, gray_img = False):
    save_dir = 'z_fetch_run_pic'
    create_dir(save_dir)
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteCamEnv(dis_tolerance = 0.001, step_ds=0.005, gray_img=gray_img)
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

    for i in range(5):
        obs = env.reset()
        # env.gripper_close(False)
        env.render()
        save_dir = 'z_fetch_run_pic_%02d' %i 
        create_dir(save_dir)
        step_count = 0
        print('------start ep %03d--------' % i)
        sum_r = 0
        while True:
            if is_render:
                env.render()
            diff_x = env.obj_pos[0] - env.pos[0]
            diff_y = env.obj_pos[1] - env.pos[1]
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
            s,r, d, info =  env.step(a)
            sum_r += r  
            
            # print('s shape = ', np.shape(s))
            # gray image

            if gray_img:
                cv2.imwrite(save_dir + '/%03d.jpg' % step_count, s[:,:,0])
            else:
                rgb_img = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_dir + '/%03d.jpg' % step_count, rgb_img)
            
            # img = Image.fromarray(s[:,:,0], 'RGB')
            # # img.save(save_dir + '/%03d.jpg' % step_count)
            # img.save(save_dir + '/%03d_r%3.2f.jpg' % (step_count,r ))
        


        a = 4 # [0, 0, 0, 0, 1]
        s,r, d, info =  env.step(a)
        sum_r += r  
        print('sum_r = ', sum_r)
        print("use step = ", step_count)

        env.render()
        

    print('use time = {:.2f}'.format(time.time()-s_time))

def go_obj_savepic_siamese(is_render = True, hsv_color = False, rand_pick_obj = False, noise = False):
    if os.path.exists('tmp/'):
        shutil.rmtree('tmp/') 
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteCamSiamenseEnv(dis_tolerance = 0.001, step_ds=0.005, 
                        gray_img=False, is_render=True, hsv_color = hsv_color)
    s_time = time.time()
    all_ep_steps = 0

    for i in range(10):

        # tmp_s_t = time.time() 
        obs = env.reset()
        # print('reset use time  =' , time.time() - tmp_s_t)
        env.render()
        save_dir = 'tmp/z_fetch_run_pic_%02d' %i 
        create_dir(save_dir)
        step_count = 0
        print('------start ep %03d--------' % i)

        if hsv_color:
            obs[0] = hsv_to_rgb(obs[0])*255.0
            obs[1] = hsv_to_rgb(obs[1])*255.0
        
            obs[0] = obs[0].astype(int)
            obs[1] = obs[1].astype(int)
            # print('obs[0]=' , obs[0])
            # print('np.shape(obs[0]) = ',np.shape(obs[0]))

            cv2.imwrite(save_dir + '/reset.jpg', obs[0])
            cv2.imwrite(save_dir + '/target.jpg', obs[1])

        else:
            rgb_img = cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_dir + '/reset.jpg', rgb_img)
            # if env.target_pic!=None:
            rgb_img_target = cv2.cvtColor(obs[1] , cv2.COLOR_BGR2RGB) 
            cv2.imwrite(save_dir + '/target.jpg', rgb_img_target)
        
        
        sum_r = 0
        target_obj_id = 0 if rand_pick_obj==False else np.random.randint(3)
        print('Target object -> object%d' % target_obj_id)
        noise_x = 0.00 if noise else 0.0
        noise_y = 0.04 if noise else 0.0

        target_pos_x = env.get_obj_pos(target_obj_id)[0] + noise_x
        target_pos_y = env.get_obj_pos(target_obj_id)[1] + noise_y
        
        while True:
            if is_render:
                env.render()
            
            diff_x = target_pos_x - env.pos[0]
            diff_y = target_pos_y - env.pos[1]
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
            s,r, d, info =  env.step(a)
            sum_r += r  
         
            prefix = ''
            if hsv_color:
                prefix = 'hsv_'
                # s[0] = cv2.cvtColor(s[0], cv2.COLOR_HSV2BGR)
                # s[1] = cv2.cvtColor(s[1], cv2.COLOR_HSV2BGR)
                # print('s[0] -> ', s[0])
                # print('s[1] -> ', s[1])
                s[0] = hsv_to_rgb(s[0])*255.0
                s[1] = hsv_to_rgb(s[1])*255.0
                s[0] = s[0].astype(int)
                s[1] = s[1].astype(int)
                s[0] = s[0][...,::-1]
                s[1] = s[1][...,::-1]
                # print('s[0] shape -> ', np.shape(s[0]) , 'data -> ',s[0])
                # print('s[1] shape -> ', np.shape(s[1]) , 'data -> ', s[1])

                cv2.imwrite(save_dir + '/%s%03d.jpg' % (prefix, step_count) , s[0])
                cv2.imwrite(save_dir + '/%s%03d_target.jpg' % (prefix, step_count) , s[1])

            else:
                # print('s[0] shape -> ', np.shape(s[0]) , 'data -> ',s[0])
                # print('s[1] shape -> ', np.shape(s[1]) , 'data -> ', s[1])
                
                rgb_img = cv2.cvtColor(s[0], cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_dir + '/%s%03d.jpg' % (prefix, step_count) , rgb_img)
                rgb_img = cv2.cvtColor(s[1], cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_dir + '/%s%03d_target.jpg' % (prefix, step_count) , rgb_img)
            

        a = 4 # [0, 0, 0, 0, 1]
        s,r, d, info =  env.step(a)
        sum_r += r  
        print('sum_r = ', sum_r)
        print("use step = ", step_count)
        all_ep_steps+=step_count
        env.render()

        # for_end = time.time() 

    use_time = time.time()-s_time
    print('use time = {:.2f}'.format(use_time))
    print('steps / second = {:.2f}'.format(all_ep_steps / use_time))
    
# go_obj_savepic()
# go_obj()

go_obj_savepic_with_camenv(gray_img=False)
# go_obj_savepic_siamese()
# go_obj_savepic_siamese(hsv_color=True, rand_pick_obj=True, noise=False)