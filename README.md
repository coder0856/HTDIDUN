# High-Throughput Decomposition-Inspired Deep Unfolding Network for Image Compressed Sensing


## Abstract
Deep Unfolding Network (DUN) has achieved great success in the image Compressed Sensing (CS) field benefiting from its great interpretability and performance. However, existing DUNs suffer from limited information transmission capacity with increasingly complex structures, leading to undesirable results. Besides, current DUNs are mostly established based on one specific optimization algorithm, which hampers the development and understanding of DUN. In this paper, we propose a new unfolding formula combining the Approximate Message Passing algorithm (AMP) and Range-Nullspace Decomposition (RND), which offers new insights for DUN design. To maximize information transmission and utilization, we propose a novel High-Throughput Decomposition-Inspired Deep Unfolding Network (HTDIDUN) based on the new formula. Specifically, we design a powerful Nullspace Information Extractor (NIE) with high-throughput transmission and stacked residual channel attention blocks. By modulating the dimension of the feature space, we provide three implementations from small to large. Extensive experiments on natural and medical images manifest that our HTDIDUN family members outperform other state-of-the-art methods by a large margin. Our codes and pre-trained models are available on GitHub to facilitate further exploration.


## Test

Download the pretrained models [model.zip](https://drive.google.com/file/d/1jRryzrtNbXzuUMUnkpb68-xsm3LpdVgr/view?usp=drive_link) and put it into  `./model/` then run:
