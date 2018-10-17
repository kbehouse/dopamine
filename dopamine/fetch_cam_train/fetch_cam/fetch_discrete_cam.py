
from fetch_cam import FetchDiscreteEnv
import cv2
import numpy as np

# because thread bloack the image catch (maybe), so create the shell class 
class FetchDiscreteCamEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005, gray_img = True):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
        self.gray_img = gray_img


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
            resize_img = cv2.resize(rgb_gripper, (128, 128), interpolation=cv2.INTER_AREA)
            return resize_img, r, d, None

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

    def reset(self):
        self.env.reset()
        self.env.render()
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
    
        if self.gray_img:
            s = self.state_preprocess(rgb_gripper)
            return s
        else:
            resize_img = cv2.resize(rgb_gripper, (128, 128), interpolation=cv2.INTER_AREA)
            return resize_img

    def render(self):
        self.env.render()

