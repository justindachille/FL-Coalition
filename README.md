Code for the following paper:
When Federated Learning Meets Oligppoly Competition: Stability and Model Differentiation.
Chao Huang, Justin Dachille, and Xin Liu.
Submitted to IEEE Transactions on Network Science and Engineering, under revision.

## Usage

This code runs as a multi-stage pipeline. First, we train all coalitions ABC, AB, AC, BC, A, B, C using `scaffold_train.py`. Then, we fine tune individual clients from coalitions ABC, AB, AC, and BC with `fine_tuning.py`. We then use `extract_data.py` to parse all log files and generate profit tables using algorithms from `bestresponse.py`.


Example training run:
```
python scaffold_train.py --model=resnet18 \
    --dataset=cifar10 \
    --alg=fedprox \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=10 \
    --n_parties=3 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cuda:0'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=1 \
    --init_seed=0 \
    --coalition ABC
```

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `simple-cnn`, `vgg`, `resnet`, `mlp`. Default = `mlp`. |
| `dataset`      | Dataset to use. Options: `mnist`, `cifar10`, `fmnist`, `svhn`, `generated`, `femnist`, `a9a`, `rcv1`, `covtype`. Default = `mnist`. |
| `alg` | The training algorithm. Options: `fedavg`, `fedprox`, `scaffold`, `fednova`. Default = `fedavg`. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `5`. |
| `n_parties` | Number of parties, default = `2`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `rho` | The parameter controlling the momentum SGD, default = `0`. |
| `comm_round`    | Number of communication rounds to use, default = `50`. |
| `partition`    | The partition way. Options: `custom-quantity`, `noniid-labeldir`, Default = `noniid-labeldir` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `sample` | Ratio of parties that participate in each communication round, default = `1`. |
| `init_seed` | The initial seed, default = `0`. |
| `abc` | The coalition to train, Input as ABC, AB, AC, BC, A, B, or C |
| `C_size` |	Data points that C has if `partition` is set to `custom-quantity` |

Example fine tuning run: `python fine_tuning.py --reg 0.0001 --rho 0.1 --ft_epochs 10 --partition iid --init_seed 42 --alg fedavg --logdir "./custom_logs/" --net_num 5 --train_all_layers --optimizer adam --lr 0.001 --device 'cuda:0' --C_size 10000 --beta 0.2 --abc AB` 

| Parameter          | Description                                                                                                                                                     | Type       | Default Value        |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|----------------------|
| `--reg`            | L2 regularization strength.                                                                                                                                     | float      | `1e-5`               |
| `--rho`            | Parameter controlling the momentum SGD.                                                                                                                         | float      | `0`                  |
| `--ft_epochs`      | Number of fine-tuning epochs.                                                                                                                                   | int        | `5`                  |
| `--partition`      | The data partitioning strategy.                                                                                                                                 | string     | `noniid-labeldir`    |
| `--init_seed`      | Random seed for initialization.                                                                                                                                 | int        | `0`                  |
| `--alg`            | Specifies the federated learning algorithm (e.g., fedavg/fedprox/scaffold/fednova/moon).                                                                        | string     | `scaffold`           |
| `--logdir`         | Directory path for storing logs.                                                                                                                                | string     | `"./logs_ft/"`       |
| `--net_num`        | Number of networks to load.                                                                                                                                     | int        | `0`                  |
| `--train_all_layers` | Flag to train all layers. If not set, only the last two layers are trained.                                                                                   | bool       | `False`              |
| `--optimizer`      | Specifies the optimizer to use (e.g., sgd, adam).                                                                                                               | string     | `sgd`                |
| `--lr`             | Learning rate.                                                                                                                                                  | float      | `0.01`               |
| `--device`         | The device to run the program on (e.g., 'cuda:0', 'cpu').                                                                                                       | string     | `cuda:0`             |
| `--C_size`         | Number of data points that C has.                                                                                                                               | int        | `8000`               |
| `--beta`           | The parameter for the Dirichlet distribution for data partitioning.                                                                                             | float      | `0.1`                |
| `--abc`            | Input specifying the combination of subsets to be used (options: ABC, AB, AC, BC, A, B, or C).                                                                  | string     | `None`               |

Now, run `python bestresponse.py`:

Example output:
