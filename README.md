# Visual Adversarial Imitation Learning using Variational Models (VMAIL)
This is the official implementation of the NeurIPS 2021 paper.

- [Project website][website]
- [Research paper][paper]
- [Datasets used in the paper][data]

[website]: https://sites.google.com/view/variational-mail
[paper]: https://arxiv.org/abs/2107.08829
[data]: https://drive.google.com/drive/folders/1JZmOVmlCqScqu0DDmn7857D5FtHZr6Un


## Method

![VMAIL](/images/VMAIL.png)

VMAIL simultaneously learns a variational dynamics model and trains an on-policy 
adversarial imitation learning algorithm in the latent space using only model-based 
rollouts. This allows for stable and sample efficient training, as well as zero-shot
imitation learning by transfering the learned dynamics model



## Instructions

Get dependencies:

```
conda env create -f vmail.yml
conda activate vmail
cd robel_claw/robel
pip install -e .
```

To train agents for each environmnet download the expert data from the provided link and run:

```
python3 -u vmail.py --logdir .logdir --expert_datadir expert_datadir
```

The training will generate tensorabord plots and GIFs in the log folder:

```
tensorboard --logdir ./logdir
```

## Citation

If you find this code useful, please reference in your paper:

```
@article{rafailov2021visual,
      title={Visual Adversarial Imitation Learning using Variational Models}, 
      author={Rafael Rafailov and Tianhe Yu and Aravind Rajeswaran and Chelsea Finn},
      year={2021},
      journal={Neural Information Processing Systems}
}
```
## Modified infomation

### 增加背景特征提取网络

增加了背景提取网络 $\mu(\cdot)$ ,使用不变提取网络 $\phi(\cdot)$ 以及背景提取网络 $\mu(\cdot)$ 同时对图片进行重建。

### 以互信息约束特征提取过程：

  （1）不同环境中 expert_data 的 label 与不变特征提取器 $\phi(\cdot)$ 为0.

      使用post['stoch'] 作为互信息的约束条件

  (2) 不同环境 expert 的 背景特征提取器 $\mu(\cdot)$ 的互信息为0