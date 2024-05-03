# Simple.py

Just simply flattening the entire image into vectors (dropping the spatial information altogether) before applying the KANs network takes roughly, 30s to train on a CUDA device and yields about **83%** accuracy.

```bash
> /home/menghuan/miniconda3/envs/kan/bin/python /home/menghuan/Code/KAN/Test3_a.py
Using cuda device
train loss: 8.09e-01 | test loss: 7.75e-01 | reg: 7.38e+02 : 100%|██| 50/50 [00:38<00:00,  1.28it/s]
Test accuracy: 82.29%
```
![kan](https://github.com/Menghuan1918/KAN-Models/assets/122662527/85746cc2-6387-4977-958a-11902d1ae818)
