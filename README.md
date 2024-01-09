# When Federated Learning Meets Oligopoly Competition: Stability and Model Differentiation

Official Code for the paper "When Federated Learning Meets Oligopoly Competition: Stability and Model Differentiation" by Chao Huang, Justin Dachille, and Xin Liu. Currently under revision and submitted to IEEE Transactions on Network Science and Engineering.

## Usage

The codebase is structured as a multi-stage pipeline to simulate the dynamics of federated learning in an oligopoly competition scenario:

1. **Training All Coalitions**: Begin by training all coalitions (ABC, AB, AC, BC, A, B, C) using the script `scaffold_train.py`. This step sets the foundation for the federated learning process under various coalition structures.

2. **Fine-Tuning Individual Clients**: Once the initial training is complete, individual clients from coalitions ABC, AB, AC, and BC can be fine-tuned using `fine_tuning.py`. This stage focuses on optimizing the performance of each client within their respective coalition.

3. **Data Extraction and Analysis**: Use `extract_data.py` to parse all log files generated during the training and fine-tuning stages. This script uses best response algorithms defined in `bestresponse.py` to generate profit tables.

## Main Requirements

- **Python**: Tested with Python 3.11.
- **torch**: PyTorch library for neural networks.
- **dill**: Extended version of the `pickle` module for serialization.

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

Example fine tuning run: 

```
python fine_tuning.py --reg 0.0001 \
    --rho 0.1 \
    --ft_epochs 10 \
    --partition iid \
    --init_seed 42 \
    --alg fedavg \
    --logdir "./custom_logs/" \
    --net_num 5 \
    --train_all_layers \
    --optimizer adam \
    --lr 0.001 \
    --device 'cuda:0' \
    --C_size 10000 \
    --beta 0.2 \
    --abc AB
```

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
```
----- Non-iid Label Dirichlet: A=3491, B=3029, C=2480 -----
[0.8681 0.8668 0.8655]
steps: 14
Optimal prices: [13.154915825526617, 3.756131282512217, 1.8780656386065406] with order [0, 1, 2] for partition ABC
Profits: [7.672878752867925, 1.252043780858635, 0.1566228337100408]
[0.71925 0.8453  0.5932 ]
steps: 14
Optimal prices: [1135.999856172593, 299.8844572101612, 149.94222716295994] with order [1, 0, 2] for partition AB_C
Profits: [654.3713301905041, 99.96148285447156, 13.590089153995377]
[0.73765 0.6394  0.8359 ]
steps: 20
Optimal prices: [893.2609172972938, 240.50894419216425, 120.25447442949151] with order [2, 0, 1] for partition AC_B
Profits: [516.1115185489128, 80.16964887328162, 10.688584714998505]
[0.6085 0.8347 0.7216]
steps: 13
Optimal prices: [1015.2738596896423, 270.3724450973251, 135.18622205924453] with order [1, 2, 0] for partition A_BC
Profits: [585.6127149835347, 90.12414803376706, 12.148368790986924]
[0.6085 0.6394 0.5776]
steps: 14
Optimal prices: [224.1180964121798, 62.63509374724026, 31.31754668350752] with order [1, 0, 2] for partition A_B_C_
Profits: [130.26135388683616, 20.878364707044057, 2.6760590163498432]
Final profit table for iid case:
 [[  0.11020622   5.3993184    0.88119978]
 [ 38.38066936 240.33925349   4.94311905]
 [101.19292112  13.71953067 660.8335754 ]
 [ 20.03420664 141.02442414 961.68241137]
 [  9.44998197  69.59548349 455.09273649]]
Final profit table for non-iid case:
 [[  7.67287875   1.25204378   0.15662283]
 [ 99.96148285 654.37133019  13.59008915]
 [ 80.16964887  10.68858471 516.11151855]
 [ 12.14836879 585.61271498  90.12414803]
 [ 20.87836471 130.26135389   2.67605902]]

--- IID Quantity Core stability ---
ABC: stable ABC?: (False, 'Not stable due to AB_C')
AB_C: stable AB_C?: (False, 'Not stable due to AC_B')
AC_B: stable AC_B?: (False, 'Not stable due to A_BC')
A_BC: stable A_BC?: (False, 'Not stable due to AB_C')
A_B_C_: stable A_B_C_?: (False, 'Not stable due to AB_C')
--- IID Quantity Individually stability ---
ABC: stable ABC?: (False, 'Not stable due to A in A_B_C_')
AB_C: stable AB_C?: (False, 'Not stable due to AC_B')
AC_B: stable AC_B?: (False, 'Not stable due to A_BC')
A_BC: stable A_BC?: (False, 'Not stable due to AB_C')
A_B_C_: stable A_B_C_?: (False, 'Not stable due to AB_C')
--- Non-IID Label Core stability ---
ABC: stable ABC?: (False, 'Not stable due to AB_C')
AB_C: stable AB_C?: (True, 'Core stable')
AC_B: stable AC_B?: (False, 'Not stable due to AB_C')
A_BC: stable A_BC?: (False, 'Not stable due to AB_C')
A_B_C_: stable A_B_C_?: (False, 'Not stable due to AB_C')
--- Non-IID Label Individual stability ---
ABC: stable ABC?: (False, 'Not stable due to A in A_B_C_')
AB_C: stable AB_C?: (True, 'Individually stable')
AC_B: stable AC_B?: (False, 'Not stable due to AB_C')
A_BC: stable A_BC?: (False, 'Not stable due to AC_B')
A_B_C_: stable A_B_C_?: (False, 'Not stable due to AB_C')
--- IID Accuracy Testing ---
ABC: stable ABC?: (True, 'Core stable')
AB_C: stable AB_C?: (False, 'Not stable due to ABC')
AC_B: stable AC_B?: (False, 'Not stable due to ABC')
A_BC: stable A_BC?: (False, 'Not stable due to ABC')
A_B_C_: stable A_B_C_?: (False, 'Not stable due to ABC')
--- Non IID Accuracy Testing ---
ABC: stable ABC?: (True, 'Core stable')
AB_C: stable AB_C?: (False, 'Not stable due to ABC')
AC_B: stable AC_B?: (False, 'Not stable due to ABC')
A_BC: stable A_BC?: (False, 'Not stable due to ABC')
A_B_C_: stable A_B_C_?: (False, 'Not stable due to ABC')
```
