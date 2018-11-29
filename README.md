# Update Dopamine  with Siamese Network


# segnet
```
python -um dopamine.fetch_cam_train.train_segnet \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_segnet \
  --gin_files='dopamine/fetch_cam_train/rainbow_segnet.gin'
```

# 84 x 84 gray
```
python -um dopamine.fetch_cam_train.train \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_reward_0 \
  --gin_files='dopamine/fetch_cam_train/rainbow.gin'
```

```
python -um dopamine.fetch_cam_train.train \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_84_gray_r_measure_realbot_tex \
  --gin_files='dopamine/fetch_cam_train/rainbow.gin'
```

```
python -um dopamine.fetch_cam_train.train \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_bin_r_measure \
  --gin_files='dopamine/fetch_cam_train/rainbow.gin'
```

```
python -um dopamine.fetch_cam_train.train \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_bin_r_0_norotate_diffgripperZ \
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
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_rgb_test \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```

# train 3obj 

## range red color
```
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_rgb_84_3obj_range_red_color_r_measure_iter19  \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```

## spatial softmax
```
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_rgb_84_3obj_range_red_color_r_measure  \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```

# complex object
```
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_spatialmax_rgb_84_3obj_complex_obj_r_measure \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```


## try get last layer
```
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_rgb_84_3obj_white \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```

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

```
python -um dopamine.fetch_cam_train.train_siamese \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/_collection_fetch_cam_rainbow_siamese/fetch_cam_rainbow_siamese_84_hsv_r_measure_subtract \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin' \
  --hsv=True
```

# segnet 
```
python -um dopamine.fetch_cam_train.train_segnet \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_segnet \
  --gin_files='dopamine/fetch_cam_train/rainbow_segnet.gin'
```

# run pick & place


```
python -um dopamine.fetch_cam_train.train_pick_place \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/rainbow_pick_r_measure_place_r_measure \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```

# run RGB gripper

```
python -um dopamine.fetch_cam_train.train_rgb_gripper \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_rgb_gripper \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb_gripper.gin'
```


note the rainbow agent img_div=1.0
```
python -um dopamine.fetch_cam_train.train \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/semantic_r_measure_no_pick \
  --gin_files='dopamine/fetch_cam_train/rainbow.gin'
```

### Acknowledgement

If you use Dopamine in your work, we ask that you cite this repository as a
reference. The preferred format (authors in alphabetical order) is:

Marc G. Bellemare, Pablo Samuel Castro, Carles Gelada, Saurabh Kumar, Subhodeep Moitra.
Dopamine, https://github.com/google/dopamine, 2018.

