# High-Throughput Decomposition-Inspired Deep Unfolding Network for Image Compressed Sensing


## Abstract
Deep Unfolding Network (DUN) has achieved great success in the image Compressed Sensing (CS) field benefiting from its great interpretability and performance. However, existing DUNs suffer from limited information transmission capacity with increasingly complex structures, leading to undesirable results. Besides, current DUNs are mostly established based on one specific optimization algorithm, which hampers the development and understanding of DUN. In this paper, we propose a new unfolding formula combining the Approximate Message Passing algorithm (AMP) and Range-Nullspace Decomposition (RND), which offers new insights for DUN design. To maximize information transmission and utilization, we propose a novel High-Throughput Decomposition-Inspired Deep Unfolding Network (HTDIDUN) based on the new formula. Specifically, we design a powerful Nullspace Information Extractor (NIE) with high-throughput transmission and stacked residual channel attention blocks. By modulating the dimension of the feature space, we provide three implementations from small to large. Extensive experiments on natural and medical images manifest that our HTDIDUN family members outperform other state-of-the-art methods by a large margin. Our codes and pre-trained models are available on GitHub to facilitate further exploration.

## Network Architecture
![net](figs/htdidun.png)

![net](figs/DUNs.png)
## Test

Download and unzip the pretrained models [pretrained_models.zip](https://drive.google.com/file/d/1jRryzrtNbXzuUMUnkpb68-xsm3LpdVgr/view?usp=drive_link). Put them into  `./model/` then run:

```shell
python test.py --testset_name Set11 --cs_ratio=0.1/0.2/0.3/0.4/0.5
```

- We also provide a 128 channel HTDIDUN for 0.3 CS ratio in the pretrained models which owns superior performance but requires longer time for training and inference.

## Train 
Download the dataset of [Waterloo Exploration Database](https://kedema.org/project/exploration/index.html) and put the `pristine_images` directory (containing 4744 `.bmp` image files) into `./data`, then run:
```shell
python train.py --cs_ratio=0.1/0.2/0.3/0.4/0.5
```

## Results


<p align="center">
  <img src="figs/result1.png" width="50%">
</p>

<p align="center">
  <img src="figs/result2.png" width="100%">
</p>







## Acknowledgement

Our work is built on the codes of [PRL](https://github.com/Guaishou74851/PRL). Thanks for their sharing.
