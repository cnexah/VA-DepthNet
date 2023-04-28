# [VA-DepthNet: A Variational Approach to Single Image Depth Prediction](https://openreview.net/forum?id=xjxUjHa_Wpa) 
>We introduce VA-DepthNet, a simple, effective, and accurate deep neural network approach for the single-image depth prediction (SIDP) problem. The proposed approach advocates using classical first-order variational constraints for this problem. While state-of-the-art deep neural network methods for SIDP learn the scene depth from images in a supervised setting, they often overlook the invaluable invariances and priors in the rigid scene space, such as the regularity of the scene. The paper's main contribution is to reveal the benefit of classical and well-founded variational constraints in the neural network design for the SIDP task. It is shown that imposing first-order variational constraints in the scene space together with popular encoder-decoder-based network architecture design provides excellent results for the supervised SIDP task. The imposed first-order variational constraint makes the network aware of the depth gradient in the scene space, i.e., regularity. The paper demonstrates the usefulness of the proposed approach via extensive evaluation and ablation analysis over several benchmark datasets, such as KITTI, NYU Depth V2, and SUN RGB-D. The VA-DepthNet at test time shows considerable improvements in depth prediction accuracy compared to the prior art and is accurate also at high-frequency regions in the scene space. At the time of writing this paper, our method -- labeled as VA-DepthNet, when tested on the [KITTI depth-prediction evaluation set benchmarks](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), shows state-of-the-art results, and is the top-performing published approach. 


## Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python vadepthnet/train.py configs/arguments_train_nyu.txt
```

Training the KITTI model:
```
python vadepthnet/train.py configs/arguments_train_kittieigen.txt
```


## Evaluation
Evaluate the NYUv2 model:
```
python vadepthent/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the KITTI model:
```
python vadepthnet/eval.py configs/arguments_eval_kittieigen.txt
```

## Pretrained Models
[NYU](https://drive.google.com/file/d/1gQQ_9awBhElHTlsVxgbfuy2540ZjlZWq/view?usp=sharing)
[KITTI Eigen](https://drive.google.com/file/d/1AYkBW8iJW36HbG0nBK_Pi477YIrQ711l/view?usp=sharing)

## Acknowledgements
Thanks to Jin Han Lee for opening source of the excellent work [BTS](https://github.com/cleinc/bts).
Thanks to Microsoft Research Asia for opening source of the excellent work [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
Thanks to Alibaba for opening source of the excellent work [NeWCRFs](https://github.com/aliyun/NeWCRFs).
