import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import random

# color_space = []
# HUE_MAX = 180
# for h in np.linspace(0,1,HUE_MAX,endpoint=False):
#     for s in np.linspace(0.1, 1.0, 10,endpoint=True):
#         for v in [0.5,1.0]:
#             color_space.append([h, s, v])

# color_space.append([0,0,  0])
# color_space.append([0,0,0.5])
# color_space.append([0,0,1.0])
# # print(color_space)

# print('----  Color rgb->HSV  -----')
# for i, hsv in enumerate( color_space):
#     rgb = hsv_to_rgb(hsv)
#     rgb = rgb * 255.0
#     rgb = rgb.astype(int)
#     print('{:4d}:({:5.3f},{:3.1f},{:3.1f}) -> ({:3d},{:3d},{:3d})'.format(i, hsv[0], hsv[1], hsv[2],rgb[0], rgb[1], rgb[2]))



# print('----  Color sample  -----')
# print(random.sample(color_space, 3))




'''switch color'''
# rgb = bgr[...,::-1]

a = [ [5, 4, 3 ,2, 1], [50, 40, 30 ,20, 10] ]
a_inv = np.array(a)[..., ::-1]
print(a_inv)