# Graph-aided Transformer Agent (GATA) Replication

## Dependency Installations
Install Python 3.7 or above and run the following commands.
```bash
$ pip install -r requirements.txt
# if developing
$ pip install -r requirements-dev.txt
```

## Observation Generation
First, download the training data by following the instructions [here](https://github.com/xingdi-eric-yuan/GATA-public/tree/master/obs_gen.0.1), then unzip under `data/obs_gen.0.1`.

```bash
# train graph updater via observation generation with one GPU
$ python train_graph_updater.py +pl_trainer.gpus=1
```

## Reinforcement Learning
Download the training data by following the instructions [here](https://github.com/xingdi-eric-yuan/GATA-public/tree/master/rl.0.2), then unzip under `data/rl.0.2`.

```bash
# train GATA at difficulty level 5 and training size 100 with one GPU
$ python train_gata.py +pl_trainer.gpus=1

# train GATA at difficulty level 3 and training size 20 with one GPU
$ python train_gata.py +pl_trainer.gpus=1 data.difficulty_level=3 data.train_data_size=20
```

## Fairscale
*(We don't use Fairscale anymore, but leaving it for posterity.)*

If `pip install fairscale` fails, try with `--no-build-isolation`. If it then fails with `unsupported GNU version! gcc versions later than 7 are not supported!`, run the following commands to have `nvcc` use the correct `gcc`:

```bash
$ sudo ln -s /usr/bin/gcc-7 /usr/local/cuda/bin/gcc
$ sudo ln -s /usr/bin/g++-7 /usr/local/cuda/bin/g++
```

If it fails with `fatal error: cublas_v2.h: No such file or directory` or `fatal error: cublas_api.h: No such file or directory`, take a look at the include directories of the compiler command, and symlink properly. An example set of commands:

```bash
$ sudo ln -s /usr/local/cuda-10.2/targets/x86_64-linux/include/cublas_v2.h /usr/local/cuda/include/cublas_v2.h
$ sudo ln -s /usr/local/cuda-10.2/targets/x86_64-linux/include/cublas_api.h /usr/local/cuda/include/cublas_api.h
```
