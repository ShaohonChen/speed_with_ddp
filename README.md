# 速度记录

## 实验命令

**1XRTX3090 GPU**

```
accelerate launch --config_file=accelerate_configs/1gpu.yaml train_cifar_acc.py 1gpu
```

**2XRTX3090 GPU**

```
accelerate launch --config_file=accelerate_configs/2gpu.yaml train_cifar_acc.py 2gpu
```

**2XRTX3090 GPU NVLINK**

```
accelerate launch --config_file=accelerate_configs/2gpu_link.yaml train_cifar_acc.py 2gpu_nvlink
```

**2XRTX3090 bridge**

```
accelerate launch --config_file=accelerate_configs/2gpu_2c.yaml train_cifar_acc.py 2gpu_bridge
```

**4XRTX3090 2nvlink**
```
accelerate launch --config_file=accelerate_configs/4gpu.yaml train_cifar_acc.py 4gpu_2nvlink
```

**4XRTX3090 bridge**
```
accelerate launch --config_file=accelerate_configs/4gpu_2c.yaml train_cifar_acc.py 4gpu_bridge
```

**8XRTX3090 bridge**
```
accelerate launch --config_file=accelerate_configs/8gpu_4nvlink.yaml train_cifar_acc.py 8gpu_4nvlink
```

**2node 4XRTX3090 2nvlink**

node 0

```
accelerate launch --config_file=accelerate_configs/2n4g_n0.yaml train_cifar_acc.py 2n4gpu_2nvlink
```

node 1

```
accelerate launch --config_file=accelerate_configs/2n4g_n1.yaml train_cifar_acc.py 2n4gpu_2nvlink
```

**2node 16XRTX3090 bridge**
```
accelerate launch --config_file=accelerate_configs/8gpu_4nvlink.yaml train_cifar_acc.py 8gpu_4nvlink
```