# Neko: a Library for Exploring Neuromorphic Learning Rules
## Update:
add batch trainer: the trainer can directly read inputs from torch dataloader without to load .npy files. It saves a lot memory. 
## Paper
https://arxiv.org/abs/2105.00324

## Installation

```bash
git clone https://github.com/byin-cwi/neko.git
cd neko
pip install -e .
```

## Code Example
Train a RSNN with ALIF neurons with e-prop on MNIST:

```python
from neko.backend import pytorch_backend as backend
# from neko.datasets import MNIST
from neko.evaluator import Evaluator
from neko.layers import ALIFRNNModel
from neko.learning_rules import Eprop
from neko.trainers import Trainer
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root='./', train=True, transform=trans, download=True)
test_set = dset.MNIST(root='./', train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

model = ALIFRNNModel(128, 10, backend=backend, task_type='classification', return_sequence=False)
evaluated_model = Evaluator(model=model, loss='categorical_crossentropy', metrics=['accuracy', 'firing_rate'])
algo = Eprop(evaluated_model, mode='symmetric')
trainer = Trainer(algo)
trainer.train(train_loader,input_size=[28,28], T = 28, n_classes=10, epochs=30)
```

## Example Scripts
### Learning with e-prop
Training on the MNIST dataset with the same setting above, but more options available.
For example, you can learn with BPTT and three variations of e-prop.
```bash
python examples/mnist.py
```

Training on the TIMIT dataset. You need to place the `timit_processed` folder the same place as the script containing the processed dataset
produced by a [script](https://github.com/IGITUGraz/eligibility_propagation/blob/master/Figure_2_TIMIT/timit_processing.py) from the original authors of e-prop.
```bash
python examples/timit.py
```

Regularization enabled:
```bash
python timit.py --reg --eprop_mode symmetric --reg_coeff 5e-7
# Test: {'loss': 0.8918977379798889, 'accuracy': 0.7501428091397849, 'firing_rate': 12.973159790039062}
```
Faster training (~7.5X, 28s per epoch with RTX3090) with regularization enabled:
```bash
python timit.py --reg --eprop_mode symmetric --reg_coeff 3e-8 --batch_size 256  --learning_rate 0.01
# Test: {'loss': 0.8605409860610962, 'accuracy': 0.7542506720430108, 'firing_rate': 13.105131149291992}
```

### Probabilistic learning with HMC
Training on the [MNIST-1D dataset](https://github.com/greydanus/mnist1d) with HMC:
```bash
python examples/mnist_1d_hmc.py
```
### Analogue Neural Network Training with Manhattan Rule
Training on the MNIST dataset with the simple Manhattan rule or Mahattan material rule:
```bash
python examples/mnist_manhattan.py
```

### Gradient Comparison Tool
Compare the gradients from BPTT with the three varients of e-prop:
```bash
python examples/mnist_gradcompare.py
```
This is a visualization from the results of the script above.

![](https://i.ibb.co/M2XDQVY/grads.png)
