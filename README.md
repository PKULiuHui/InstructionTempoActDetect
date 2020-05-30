# End-to-End Dense Video Captioning with Masked Transformer

Based on the code of the paper [End-to-End Dense Video Captioning with Masked Transformer](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0037.pdf).


## Requirements (Recommended)
1) [Miniconda3](https://conda.io/miniconda.html) for Python 3.6

2) CUDA 9.2 and CUDNN v7.1

3) [PyTorch 0.4.0](https://pytorch.org/get-started/locally/). Follow the instructions to install pytorch and torchvision.

4) Install other required modules (e.g., torchtext)

`pip install -r requirements.txt`

Optional: If you would like to use visdom to track training do `pip install visdom`

Optional: If you would like to use spacy tokenizer do `pip install spacy`

Note: The code has been tested on a variety of GPUs, including 1080 Ti, Titan Xp, P100, V100 etc. However, for the latest RTX GPUs (e.g., 2080 Ti), CUDA 10.0 and hence PyTorch 1.0 are required. The code needs to be upgraded to PyTorch 1.0.


## Data Preparation
### Annotation and feature

We use YouCook2 dataset here. The annotation files are available [here](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/yc2.tar.gz) and should be placed under `data`. The feature files are [[train (9.6GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/training_feat_yc2.tar.gz), [val (3.2GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/validation_feat_yc2.tar.gz), [test (1.5GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/testing_feat_yc2.tar.gz)].
