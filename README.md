# meta_complex_noiselayer
* 使用多层nn.linear()建模转移矩阵
* meta-machine详见meta_c_noiselayer.py
* NoiseLayer()详见resnet.py定义

## 训练过程
* 在带噪数据集上训练CE.py，保存模型参数为model_best_X_X_X；
```
python3 CE.py --dataset cifar10 --corruption_type flip_smi --corrution_prob 0.8
```
* 训练meta_c_noiselayer.py
  * 以CE的模型参数model_best_X_X_X为网络初值；
  * 利用CE model和get_C_hat()函数估计样本整体的噪声转移矩阵[CCN]，作为meta_machine的权重初值。
```
python3 meta_c_noiselayer.py --dataset cifar10 --corruption_type flip_smi --corrution_prob 0.8
```
