# Update Dopamine  with Siamese Network

# 84 x 84 gray
```
python -um dopamine.fetch_cam_train.train \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_reward_0 \
  --gin_files='dopamine/fetch_cam_train/rainbow.gin'
```

## train only one object with rgb color and 4 pic (12 channels)
```
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_rgb_128_1obj \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```

```
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_rgb_84_3obj_white \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```


```
python -um dopamine.fetch_cam_train.train_siamese \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_siamese_128_5kQupdate \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin'
```

## train 3obj 
 python -um dopamine.fetch_cam_train.train_rgb   --agent_name=rainbow   --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_rgb_84_3obj_white_r0_measure_r1_dismodify   --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'



## hsv 
```
python -um dopamine.fetch_cam_train.train_siamese \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_siamese_84_hsv_r0_r1 \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin' \
  --hsv=True
```
```
python -um dopamine.fetch_cam_train.train_siamese \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_siamese_128_hsv \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin' \
  --hsv=True
```

```
python -um dopamine.fetch_cam_train.train_siamese \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_siamese_84_hsv_r_measure \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin' \
  --hsv=True
```

## run render
```
python -um dopamine.fetch_cam_train.train_siamese_render \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/_collection_fetch_cam_rainbow_siamese/fetch_cam_rainbow_siamese_84_hsv \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin' \
  --hsv=True
```


# run pick and place
```

python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_pick_place_test \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```

### Acknowledgement

If you use Dopamine in your work, we ask that you cite this repository as a
reference. The preferred format (authors in alphabetical order) is:

Marc G. Bellemare, Pablo Samuel Castro, Carles Gelada, Saurabh Kumar, Subhodeep Moitra.
Dopamine, https://github.com/google/dopamine, 2018.

