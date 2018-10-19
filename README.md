# Update Dopamine  with Siamese Network


```
python -um dopamine.fetch_cam_train.train \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_test \
  --gin_files='dopamine/fetch_cam_train/rainbow.gin'
```


```
python -um dopamine.fetch_cam_train.train_rgb \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_rgb_256 \
  --gin_files='dopamine/fetch_cam_train/rainbow_rgb.gin'
```


```
python -um dopamine.fetch_cam_train.train_siamese \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_siamese_128_5kQupdate \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin'
```


```
python -um dopamine.fetch_cam_train.train_siamese \
  --agent_name=rainbow \
  --base_dir=/home/iclab/phd/DRL/dopamine/log/fetch_cam_rainbow_siamese_84 \
  --gin_files='dopamine/fetch_cam_train/rainbow_siamese.gin'
```
### Acknowledgement

If you use Dopamine in your work, we ask that you cite this repository as a
reference. The preferred format (authors in alphabetical order) is:

Marc G. Bellemare, Pablo Samuel Castro, Carles Gelada, Saurabh Kumar, Subhodeep Moitra.
Dopamine, https://github.com/google/dopamine, 2018.

