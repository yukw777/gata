# Graph-aided Transformer Agent (GATA) Replication

## Fairscale
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
