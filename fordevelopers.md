# Some additional info. for developers

## An installation TIPS for developers

By releasing the test, you will be using the `pip install`, but if you use this, the development status will not be reflected until the version goes up.
If you want to use the latest branch that you have modified or developed at hand, or you want to use the latest branch, raise the priority of the one that is currently in development.

If you do it in a file, for example:

```python
import sys
import os
sys.path.insert(1, os.getcwd())
# When searching a library, the current directory has the next highest priority after the executable file path
```

If you want to do it from outside the file, for example, change the PYTHONPATH setting. for example,
```bash
export PYTHONPATH="/path/to/dir:$PYTHONPATH"
```
If you want to make the settings permanent, write the settings in `.bashrc` etc.



## A recommended set up for use of ARM server
Setting up virtual environment at the ARM server with many cores.

- First create a directory for the environments
```bash
$ mkdir ~/virt_env
```

- Then load the module

```bash
$ module load loadonly/python_wrapper
```

- It is a good idea to add this line to your ~/.bash_rc.

- Then create a virtual environment. 
- You create a python3 environment called mynewenv. The environment will be immediately activated with

```bash
$ mkvirtualenv -p /usr/bin/python3 mynewenv
```

- You may directly install the GPU version of tensorflow with

```bash
$ pip install tensorflow-gpu tensorboard
```

- You can deactivate the current environment with

```bash
$ deactivate mynewenv
```

- You can activate it with

```bash
$ workon mynewenv
```

Activate the virtual environment

```bash
$ module load loadonly/python_wrapper
$ source ~/virt_env/mynewenv/bin/activate
```

Installing BriCA2 parallel (beta)

```bash
$ git clone https://github.com/BriCA/BriCA2
$ git checkout wip/python
$ cd BriCA2
```

activate the virtual environment

```bash
$ pip install .
```


