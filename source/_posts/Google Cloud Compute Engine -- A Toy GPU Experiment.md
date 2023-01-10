---
title: Google Cloud Compute Engine -- A Toy GPU Experiment
date: 2023-01-10 20:03:08
categories:
tags:
---

Recording How to Deploy a model on Google Cloud GPU Compute Engine.

<!-- more -->

## 1. Request a Virtual Machine

### 1.1 Expand Individual Quotas Limitation
For a new google platform account who tries to request a VM with GPU, the first thing to do is to expand the individual quotas, or you will failed to request it and receieve some error infomation like "Your project has reached its limit for GPUS_ALL_REGIONS globally".

To expand the individual quotas, click that: __"Google Cloud" > "IAM & Admin" > "Quotas(left bar)" > "Filter: Metric:compute.googleapis.com/gpus_all_regions" > "click the item" > "Edit Quotas" > "Set new limit as 1"__

And for the request description, you can write you purpose like: Do some deep learning expriements.

<center>
    <img src="request1.png", width="80%">
</center>


### 1.2 Request a VM with GPU
After you successing to expand your quota limitation for gpu, you can try to request a virtual machine with GPU.

First of all, just open google cloud homepage and click the "Create a VM" buttom.

<center>
    <img src="homepage.png", width="80%">
</center>

Secondly, you can set your perfered VM name, region, disk and so on. And for Machine Configuration, just click GPU to pick up GPU type.

And especially, to avoid some annoying cuda configuration, you can click the Switch Image bottom in the Boot Disk buttom and select your favourite image among different OS system, disk size and cuda version.

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
When the requesting VM was constructed in Google Cloud, we can not only access it on the website, but we can apply ssh to access it locally. To achieve it, we need to do two steps:

#### 3.1.2 Online
For online Google Cloud page setting, the user's ssh key is supposed to save in the [Meta & dataSSH KEYS Website](https://console.cloud.google.com/compute/metadata). 

We can find our ssh key in local path: ```~/.ssh/id_rsa.pub```. If you haven't create the ssh, you can refer to https://cloud.google.com/compute/docs/connect/create-ssh-keys.

Then, we can copy the content in the Google Platform website to make it able to identify you. 

> BTW, the id_rsa.pub content behind the ```= ``` should be replaced to the username as which you would like to login in.

#### 3.1.3 Local
To access the Google cloud VM more efficient, we can assign an alias to the VM ip.

The VM ip address is easy to find on its detail page, and we should append it with its alias to ```/etc/hosts```. 

Just like:

```shell
192.168.0.1 ServerAlias
```

The you can use ssh to access your vm:

```
ssh UserName@ServerAlias
```

### 2.2 Check VM Info

In session 2, we select the "GPU-optimized Debian 10 with CUDA 11.0" image which help us to avoid the additional work at no additional cost. We can use command ```nvidia-smi``` to check the gpu detail, as shown below"

<center>
    <img src="linux-gpu-info.png">
</center>

### 2.3 Pycharm ssh Python Interpreter
Pyenv and virtualenv help us to manage different the python versions. Using below commands to install pyenv, virtualenv and python3.6.9.

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

Then, we can use ssh python interpreter and project synchronization to edit the code in the local code editor and run it on the server.

Firstly, open the "Pycharm > References > Python Interpreter > Add Interpreter > On SSH".

<center>
    <img src="ssh-interpreter1.png", width=80%>
</center>

Secondly, setting the SSH connection as Existing, and put Username@ServerAlias in the SSH Server.
<center>
    <img src="ssh-interpreter2.png", width=80%>
</center>

When "Introspection completed" appeared means the server is connected successfully.  And then click next, we should set the Project directory and Python runtime configuration.
<center>
    <img src="ssh-interpreter4.png", width=80%>
</center>

Now we finished the remote setting, the saved file will be upload to the server to keep the synchronization. To run the project on the server, we can open "Tools > Start SSH Session" to ssh to server and run the project code.

___

## 3. Result
We can run some toy code on the server to check whether the gpu is accessible.

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
