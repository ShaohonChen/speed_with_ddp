# 速度记录

## 实验命令

**暂时无法使用 1kernal AMD EPYC 7R32 48-Core Processor**

```
accelerate launch --config_file=accelerate_configs/1cpu.yaml train_cifar_acc.py 1cpu
```

**暂时无法使用 16kernal AMD EPYC 7R32 48-Core Processor**

```
accelerate launch --config_file=accelerate_configs/16cpu.yaml train_cifar_acc.py 16cpu
```

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

node 0 (172.16.6.4)

```
accelerate launch --config_file=accelerate_configs/2n4g_n0.yaml train_cifar_acc.py 2n4gpu_2nvlink
```

node 1 (172.16.6.3)

```
accelerate launch --config_file=accelerate_configs/2n4g_n1.yaml train_cifar_acc.py 2n4gpu_2nvlink
```

**2node 8XRTX3090 4nvlink**

node 0 (172.16.6.4)

```
accelerate launch --config_file=accelerate_configs/2n8g_n0.yaml train_cifar_acc.py 2n8gpu_4nvlink
```

node 1 (172.16.6.3)

```
accelerate launch --config_file=accelerate_configs/2n8g_n1.yaml train_cifar_acc.py 2n8gpu_4nvlink
```

**2node 8XRTX3090 4nvlink bf16**

node 0 (172.16.6.4)

```
accelerate launch --config_file=accelerate_configs/2n8g_n0_bf16.yaml train_cifar_acc.py 2n8gpu_4nvlink_bf16
```

node 1 (172.16.6.3)

```
accelerate launch --config_file=accelerate_configs/2n8g_n1_bf16.yaml train_cifar_acc.py 2n8gpu_4nvlink_bf16
```
