
from fetch_cam import FetchDiscreteEnv
import cv2
import numpy as np
import time
# because thread bloack the image catch (maybe), so create the shell class 

IMG_W_H = 84
class FetchDiscreteCamSiamenseEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005, gray_img = True, is_render=False):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005, is_render = is_render)
        self.gray_img = gray_img

        self.target_pic = None

    def state_preprocess(self, img):
        resize_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
        return np.reshape(gray_img,(84,84,1))
        

    def step(self,action):
        # print('i action = ', action)
        a_one_hot = np.zeros(5)
        a_one_hot[action] = 1
        s, r, d, _ = self.env.step(a_one_hot)

        # no use, but you need preserve it; otherwise, you will get error image
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)

        # s = self.state_preprocess(rgb_gripper)
        if self.gray_img:
            s = self.state_preprocess(rgb_gripper)
            return s, r, d, None
        else:
            resize_img = cv2.resize(rgb_gripper, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            # return resize_img, r, d, None
            return [resize_img, self.target_pic], r, d, None

        # return s, r, d, None

    @property
    def pos(self):
        return self.env.pos

    @property
    def obj_pos(self):
        return self.env.obj_pos

    @property
    def gripper_state(self):
        return self.env.gripper_state


    def take_only_obj0_pic(self):
        
        try:
            self.env.rand_obj0_hide_obj1_obj2()
            self.env.render()
            # time.sleep(2)
            rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                        mode='offscreen', device_id=-1)
            rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
                mode='offscreen', device_id=-1)
            resize_img = cv2.resize(rgb_gripper, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            self.target_pic = resize_img.copy()

            self.env.recover_obj0_obj1_obj2_pos()
            self.env.render()
            # time.sleep(2)
        except Exception as e:
            print(' Exception e -> ', e )
            pass
            # print(' Exception e -> ', e )
        

    def reset(self):
        # self.env.reset()
        # self.env.render()
        # self.env.rand_objs_color()
        # self.env.rand_obj0_hide_obj1_obj2()
        # time.sleep(3)
        # self.env.render()
        # try:
        #     rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
        #                 mode='offscreen', device_id=-1)
        #     rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
        #         mode='offscreen', device_id=-1)
        #     resize_img = cv2.resize(rgb_gripper, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
        #     self.target_pic = resize_img
        # except Exception as e:
        #     print(' Exception e -> ', e )
        self.env.rand_objs_color()

        self.env.reset()
        self.take_only_obj0_pic()
        

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
            return [resize_img, self.target_pic]

    def render(self):
        self.env.render()

