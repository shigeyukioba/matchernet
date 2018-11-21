- Setting up virtual environment at the ARM server
First create a directory for the environments

 mkdir ~/virt_env

Then load the module

 module load loadonly/python_wrapper

It is a good idea to add this line to your ~/.bash_rc.

Then create a virtual environment. With

 mkvirtualenv -p /usr/bin/python3 mynewenv

you create a python3 environment called mynewenv. The environment will be immediately activated. With

 pip install tensorflow-gpu tensorboard

you may directly install the GPU version of tensorflow. With

 deactivate mynewenv
you can deactivate the current environment. With

 workon mynewenv

you can activate it


- Activate the virtual environment
$ module load loadonly/python_wrapper
$ source ~/virt_env/mynewenv/bin/activate

- Installing BriCA2 parallel (beta)
$ git clone https://github.com/BriCA/BriCA2
$ git checkout wip/python
$ cd BriCA2
-- activate the virtual environment
$ pip install .
