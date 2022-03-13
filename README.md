# Modeling Explicit Concerning States for Reinforcement Learning in Visual Dialogue 

Pytorch Implementation of the paper:

**[Modeling Explicit Concerning States for Reinforcement Learning in Visual Dialogue](https://www.bmvc2021-virtualconference.com/assets/papers/0533.pdf)**.  
Zipeng Xu, Fandong Meng, Xiaojie Wang, Duo Zheng, Chenxu Lv and Jie Zhou.
In Proccedings of BMVC 2021.

(The Appendix is included in our arXiv version: https://arxiv.org/pdf/2107.05250.pdf.)

This code is adapted from [vmurahari3/visdial-diversity][1], we thank for their open sourcing.

### Download data

**Download preprocessed dialog data for VisDial v1.0:**
```
sh scripts/download_preprocessed.sh
```
**Download extracted features:**

We use bottom-up image features with 10-100 proposals for each image.
We use the features provided by [Gi-Cheon Kang et al.][2].
We thanks for their release.

Please download the files and put them under `data/image_features`.
  * [`train_btmup_f.hdf5`][3]: Bottom-up features of 10-100 proposals from images of `train` split (32GB).
  * [`train_imgid2idx.pkl`][4]: `image_id` to bounding box index file for `train` split 
  * [`val_btmup_f.hdf5`][5]: Bottom-up features of 10-100 proposals from images of `val` split (0.5GB).
  * [`val_imgid2idx.pkl`][6]: `image_id` to bounding box index file for `val` split

### Training
For Supervised Learning pre-training:

`SL: Q-Bot `

```
python train_sl.py -useGPU -trainMode sl-qbot -saveName SL_QBot 
```

`SL: A-Bot `

```
python train_sl.py -useGPU -trainMode sl-abot -a_learningRate 4e-4 -lrDecayRate 0.75 -saveName SL_ABot 
```

For Reinforcement Learning fine-tuning with ECS-based rewards:

```
python train_rl.py -dropout 0 -useGPU -useNDCG -trainMode rl-full-QAf -startFrom checkpoints/SL_ABOT.vd -qstartFrom checkpoints/SL_QBOT.vd -saveName RL-ECS
```

### Pre-trained checkpoints

Will be released this week.

## Reference
```
@inproceedings{xu2021ecsvisdial,
author = {Xu, Zipeng and Meng, Fandong and Wang, Xiaojie and Zheng, Duo and Lv, Chenxu and Zhou, Jie},
title = {modeling Explicit Concerning States for Reinforcement Learning in Visual Dialogue},
booktitle = {Proceedings of the 32nd British Machine Vision Conference (BMVC)},
year = {2021}
}
```
[1]: https://github.com/vmurahari3/visdial-diversity
[2]: https://github.com/gicheonkang/dan-visdial
[3]: https://drive.google.com/file/d/1NYlSSikwEAqpJDsNGqOxgc0ZOkpQtom9/view?usp=sharing
[4]: https://drive.google.com/file/d/1QSi0Lr4XKdQ2LdoS1taS6P9IBVAKRntF/view?usp=sharing
[5]: https://drive.google.com/file/d/1NI5TNKKhqm6ggpB2CK4k8yKiYQE3efW6/view?usp=sharing
[6]: https://drive.google.com/file/d/1nTBaLziRIVkKAqFtQ-YIbXew2tYMUOSZ/view?usp=sharing
