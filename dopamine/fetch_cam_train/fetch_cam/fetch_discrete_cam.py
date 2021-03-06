
from fetch_cam import FetchDiscreteEnv
import cv2
import numpy as np


IMG_W_H = 84
# because thread bloack the image catch (maybe), so create the shell class 
class FetchDiscreteCamEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005, gray_img = True, use_tray = True, is_render = False, only_show_obj0=False):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005, use_tray=use_tray, is_render = is_render)
        self.gray_img = gray_img
        self.is_render = is_render
        self.only_show_obj0 = only_show_obj0


    def state_preprocess(self, img):
        resize_img = cv2.resize(img, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
        return np.reshape(gray_img,(IMG_W_H,IMG_W_H,1))
        

    def step(self,action):
        # print('i action = ', action)
        a_one_hot = np.zeros(6)
        a_one_hot[action] = 1
        s, r, d, _ = self.env.step(a_one_hot)

        # no use, but you need preserve it; otherwise, you will get error image
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
        
        rgb_gripper =  cv2.cvtColor(rgb_gripper, cv2.COLOR_BGR2RGB)
        if self.is_render:
            if self.gray_img:
                # resize_img = cv2.resize(rgb_gripper, (256, 256), interpolation=cv2.INTER_AREA)
                gray_img = cv2.cvtColor(rgb_gripper, cv2.COLOR_RGB2GRAY)
                cv2.imshow('Gripper Image',gray_img)
                cv2.waitKey(50)
            else: 
                self.render_gripper_img(rgb_gripper)

        # s = self.state_preprocess(rgb_gripper)
        if self.gray_img:
            s = self.state_preprocess(rgb_gripper)
            return s, r, d, None
        else:
            resize_img = cv2.resize(rgb_gripper, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            return resize_img, r, d, None

        # return s, r, d, None

    @property
    def pos(self):
        return self.env.pos

    @property
    def obj_pos(self):
        return self.env.obj_pos

    @property
    def red_tray_pos(self):
        return self.env.red_tray_pos

    @property
    def gripper_state(self):
        return self.env.gripper_state

    @property
    def is_gripper_close(self):
        return self.env.is_gripper_close

    def reset(self):
        # self.env.rand_objs_color(exclude_obj0 = True)
        # self.env.rand_red_or_not(obj_name='object0', use_red_color=True)
        self.env.rand_red_or_not(obj_name='object1', use_red_color=False)
        self.env.rand_red_or_not(obj_name='object2', use_red_color=False)
        self.env.reset()
        if self.only_show_obj0:
            self.env.hide_obj1_obj2()
        self.env.render()
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
    
        if self.gray_img:
            s = self.state_preprocess(rgb_gripper)
            return s
        else:
            resize_img = cv2.resize(rgb_gripper, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            return resize_img

    def render(self):
        self.env.render()

    
    def render_gripper_img(self, gripper_img):
        # if self.is_render:
        # rgb_img = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('Gripper Image',gripper_img)
        cv2.waitKey(50)