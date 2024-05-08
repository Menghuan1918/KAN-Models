# Simple.py

Just simply flattening the entire image into vectors (dropping the spatial information altogether) before applying the KANs network takes roughly, 30s to train on a CUDA device and yields about **83%** accuracy.

```bash
> /home/menghuan/miniconda3/envs/kan/bin/python /home/menghuan/Code/KAN/MINST.py
Using cuda device
train loss: 8.09e-01 | test loss: 7.75e-01 | reg: 7.38e+02 : 100%|██| 50/50 [00:38<00:00,  1.28it/s]
Test accuracy: 82.29%
```

> Please follow the pykan installation guide. And the versoin of `torchvision` is 0.17.2