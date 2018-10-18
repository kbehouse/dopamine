#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer, load_model_from_xml
from mujoco_py.modder import TextureModder
import os
import numpy as np



model = load_model_from_path("fetch/main.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
modder = TextureModder(sim)

t = 0

while True:
    for name in sim.model.geom_names:
        # choice = self.random_state.randint(len(choices))

        # modder.rand_all(name)
        print(name)
        modder.set_rgb(name,  np.array([0, 255,0 ],  dtype=np.uint8)   )
        # return self.set_rgb(name, rgb)

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break
