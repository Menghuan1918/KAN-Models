# Simple way

Just simply flattening the entire image into vectors (dropping the spatial information altogether) before applying the KANs network takes roughly, 30s to train on a CUDA device and yields about **83%** accuracy.

See [Simple.md](MNIST/Simple/Simple.md) for more.

# Replace the linear

Replace the linear function in a normal CNN network. Let's call the modified network KAN_CNN, we get:

| Accuracy | CNN | KAN_CNN |
| --- | --- | --- |
| 15 epochs | 0.9922 | 0.9917 |

See [KAN_CNN.md](MNIST/Replace_linear/KAN_CNN.md) for more.