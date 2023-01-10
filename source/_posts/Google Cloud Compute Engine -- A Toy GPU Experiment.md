---
title: Google Cloud Compute Engine -- A Toy GPU Experiment
date: 2023-01-10 20:03:08
categories:
tags:
---

Recording How to Request a Google Cloud Compute Engine VM with personalized GPU.

<!-- more -->

## 1. Request a Virtual Machine

### 1.1 Expand Individual Quotas Limitation
For a new account who want to request a VM with GPU on the google cloud compute engine, the first thing to do is to expand the individual quotas. Otherwise, you will fail and receieve some error infomation like "Your project has reached its limit for GPUS_ALL_REGIONS globally".

To expand the individual quotas, click as below order: __"Google Cloud" > "IAM & Admin" > "Quotas(left bar)" > "Filter: Metric:compute.googleapis.com/gpus_all_regions" > "click the item" > "Edit Quotas" > "Set new limit as 1"__.

And for the _request description_, you can write you purpose just like: Do some deep learning expriements.

<center>
    <img src="request1.png", width="80%">
</center>


### 1.2 Request a VM with GPU
After individual GPU quota limitation being expanded successfully, a virtual machine with GPU can begin to be requested.

First of all, just open google cloud homepage and click the "Create a VM" buttom.

<center>
    <img src="homepage.png", width="80%">
</center>

Secondly, some basic settings, like VM name, region, disk and so on should be put in. And especially, for the Machine Configuration, just click GPU to pick up the target GPU type.

To avoid some annoying cuda configurations, you can just click the _Switch Image_ bottom in the _Boot Disk_ buttom and select your favourite image among different OS system, disk size and cuda version.

<center>
    <img src="image.png", width="80%">
</center>

<center>
    <img src="boot.png", width="80%">
</center>

When all settings have been done, you can then create your VM.
___
## 2. Configure the Project Dependencies

### 2.1 Remote ssh Configuration
Once VM is constructed on Google Cloud, not only access on the website directly is allowed, but also local ssh to access it can be applied to. To achieve it, simple two steps should be done:

#### 2.1.2 Online
For online Google Cloud page setting side, the user's ssh key is supposed to save in the [Meta & dataSSH KEYS Website](https://console.cloud.google.com/compute/metadata). 

The personal ssh key can be found in private server as the local path: ```~/.ssh/id_rsa.pub```. If the ssh key haven't be created, this website https://cloud.google.com/compute/docs/connect/create-ssh-keys may be helpfully.

Once the user's ssh key is added to the Google Platform, it will be able to identify the specified user and allow him to access it by ssh. 

> BTW, the id_rsa.pub content behind the ```= ``` should be replaced to the username as which you would like to login in.

#### 2.1.3 Local
To access the Google cloud VM more efficient and convenient, assigning an alias to the VM is a good method.

The VM ip address can be easy to get on its detail home page, and just append the ip address with its alias to the last line of ```/etc/hosts``` with special format. 

Just like:

```shell
192.168.0.1 ServerAlias
```

The following command can be used to access the VM locally:

```
ssh UserName@ServerAlias
```

### 2.2 Check VM Info

In session 2, selecting the "GPU-optimized Debian 10 with CUDA 11.0" image helps user to avoid the additional work at no additional cost. A simple command ```nvidia-smi``` can be used to check the gpu detail, as shown below"

<center>
    <img src="linux-gpu-info.png">
</center>

### 2.3 Pycharm ssh Python Interpreter

#### 2.3.1 Install Pyenv interpreter on VM
Pyenv and virtualenv are powerful software to help users to manage their different python versions. Just using below commands to install the pyenv, virtualenv and specified verson ofpython, like 3.6.9.

```shell
# python dependencies
sudo apt-get update 
sudo apt-get upgrade
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# pyenv 
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc 
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc 
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
exec bash

# template py 3.6.9
pyenv install 3.6.9

# virtualenv pyenv plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
exec bash

# rename darts environment
pyenv virtualenv 3.6.9 darts-env
```

#### 2.3.2 Pycharm Remote SSH Configuration

To achieve the goal of editting the code locally while running it on the server, some steps should be taken to the Pycharm.

First of all, clicking following the order: the "Pycharm > References > Python Interpreter > Add Interpreter > On SSH".

<center>
    <img src="ssh-interpreter1.png", width=80%>
</center>

Setting the SSH connection as Existing, and put Username@ServerAlias in the SSH Server.
<center>
    <img src="ssh-interpreter2.png", width=80%>
</center>

When "Introspection completed" appeared means the server is connected successfully.  And then click next, setting the Project directory and Python runtime configuration.
<center>
    <img src="ssh-interpreter4.png", width=80%>
</center>

Now remote setting is finished, and the saved file will be upload to the server to keep the synchronization. To run the project on the server, opening "Tools > Start SSH Session" to ssh to server and run the project code.

___

## 3. Result
Try some toy code on the server to check whether the gpu is accessible.

```python
print("PyTorch version: ", torch.__version__)
print("GPU support: ", torch.cuda.is_available())
print("Available devices count: ", torch.cuda.device_count())

# move tensor to GPU if available
tensor = torch.rand(4, 6)
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor = tensor.to(device)
print(f"Device tensor is stored on: {tensor.device}")

# toy model Attention
v = torch.rand((2,4,8))
k = v
q = torch.rand((2,4,8))
d_k = 8


def softmax(x, dim=0):
    x_exp = x.exp()
    partition = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / partition


def attention(q, k, v):
    # bmm batch matrix multiply
    return torch.bmm(softmax(torch.bmm(q, k.transpose(2, 1)) / math.sqrt(d_k), dim=1), v)


print(attention(q, k, v))
```

<center>
    <img src="demo.png", width="80%">
</center>

GPU works, have fun!
