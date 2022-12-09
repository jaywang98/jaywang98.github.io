---
title: Beautify Mac Terminal By Iterm2 And zsh
date: 2022-08-21 13:08:44
tags:
    - mac
    - zsh
    - iterm2
categories: 
    - Mac
---
在公司实习需要使用mac电脑来进行工作，其中bash是非常重要的工具，本文也是对bash进行了美化，提升使用的效率。
本文采用的方案是利用ITerm2 与 oh-my-zsh来进行美化,同时也要将其配置在vscode中。

在配置的过程中，发现如果路径层级过长，主机名的信息较长，会导致代码容易换行，影响观看。所以选了隐藏路径以及主机名，仅保留当前文件名，效果如图所示：

<!-- more -->

# 1. Iterm2安装及其陪配置
手动安装: [打开Iterm2 的官方网址](https://iterm2.com/)，下载相应的Iterm2 安装包，并安装。
Homebrew安装: 为了方便后期管理，采用homebrew进行安装。
```bash
# use brew to install iterm2
brew install iTerm2
```

设置iTerm2为默认的终端

<center>
    <img src="default-iterm2.jpg" width=80% >
</center>

设置iTerm2的字体为 *Meslo LG L DZ for powerline*，之后vscode的terminal字体也设置为*Meslo LG L DZ for powerline*。

# 2. oh-my-zsh

 首先，在进入oh-my-zsh之前，要先了解zsh。
 
 >  **zsh又称为z-shell**，是一款可用作交互式登录的shell及脚本编写的命令解释器。Zsh对Bourne shell做出了大量改进，同时加入了Bash、ksh及tcsh的某些功能。自2019年起，macOS的默认Shell已从Bash改为Zsh。


 zsh 有如下几个特点：
 - 可帮助用户键入常用命令选项及参数的可编程命令行补全功能，自带对数百条命令的支持
-  可与任意Shell共享命令历史
- 多种兼容模式（例如，Zsh可在运行为/bin/sh的情况下伪装成Bourne shell）
- 自带where命令，其与which命令类似，但是显示指定于$PATH中所指定指令的全部位置，而不是仅显示所使用指令的位置。

其中，用户社区网站"Oh My Zsh"收集Z shell的第三方插件及主题。

本文也是利用oh-my-zsh来美化oh-my-zsh，利用如下指令进行安装，
``` bash
# curl 安装
export REMOTE=https://gitee.com/imirror/ohmyzsh.git
sh -c "$(curl -fsSL https://cdn.jsdelivr.net/gh/ohmyzsh/ohmyzsh/tools/install.sh)"

# wget安装
export REMOTE=https://gitee.com/imirror/ohmyzsh.git
sh -c "$(wget -O- https://cdn.jsdelivr.net/gh/ohmyzsh/ohmyzsh/tools/install.sh)"

# fetch安装
export REMOTE=https://gitee.com/imirror/ohmyzsh.git
sh -c "$(fetch -o - https://cdn.jsdelivr.net/gh/ohmyzsh/ohmyzsh/tools/install.sh)"
```
安装界面如下：
<center>
<img src="oh-my-zsh安装.jpg" width=80% >
</center>

# 3. 主题配置
oh-my-zsh主题很多，其中 *agnoster* 是比较常用的主题，本文也是打算使用该主题（更多的主题可以查看[theme主题](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)。

# 3.1 ZSH_THEME配置
打开 ~/.zshrc,修改ZSH-THEME这一配置.
> *zshrc* file is where *you'd place customizations to the z shell*.
```shell
ZSH_THEME ="agnoster
```
<center>
<img src="ZSH_THEME.jpg" width=80% >
</center>

# 3.2  字体安装与配置
注意agnoster需要安装额外的字体[Meslo for Powerline](https://cdn.jsdelivr.net/gh/powerline/fonts/Meslo%20Slashed/Meslo%20LG%20S%20Regular%20for%20Powerline.ttf),　下载安装相应的ttf文件。

配置完字体之后，打开iTerm -> Preferences -> Profiles -> Text -> Change Font，选择Meslo LG S Regular for Powerline。

<center>
    <img src="iterm-pref.jpg", width=80% >
</center>


<center>
    <img src="zsh-font.jpg", width=80% >
</center>

<center>
    <img src="zsh-demo.jpg", width=80% >
</center>

# 4. 优化zsh
在3.2 中，看到安装oh-my-zsh后的效果。但是存在一定的不足，比如命令行开头都有主机名，文件路径，能够看到当层级增多的时候，会显得非常的长，影响输入，需要一定的优化。笔者选择的方式，是隐去主机名和层级路径，仅保留用户名和当前文件路径，当需要查看路径的时候调用指令pwd即可。

# 4.1 隐藏主机名
根据3.1可知，zsh的配置信息主要在~/.zshrc这个文件，在该**文件底部增加**下添加
```bash
# * 方法一：隐藏主机和用户名-是笔者选择的方案
prompt_context() {}

# 方法二：隐藏主机名
prompt_context() {
  if [[ "$USER" != "$DEFAULT_USER" || -n "$SSH_CLIENT" ]]; then
    prompt_segment black default "%(!.%{%F{yellow}%}.)$USER"
  fi
}

# 方法三：隐藏用户名
prompt_context() {
  if [[ "$USER" != "$DEFAULT_USER" || -n "$SSH_CLIENT" ]]; then
    prompt_segment black default "%(!.%{%F{yellow}%}.)$HOST"
  fi
}

#  更新 ~/.zshrc
source ~/.zshrc
```

# 4.2 隐藏层级路径，保留当前路径

在路径 /Users/username/.oh-my-zsh/themes/agnoster.zsh-theme 下，打开后找到prompt_dir() {}这个函数，然后将prompt_segment blue black '%~' , 最后面的波浪线改为c即可：prompt_segment blue black '%c',。

最后调用指令 source /User/username/.zshrc, 完成刷新，实现了保留用户名以及当前层级用户。

<center>
    <img src="hide-host-name.jpg", width=80% >
</center>

# 5. 插件
为了进一步提高zsh的效率，笔者添加了部分插件。但是，调用终端的好处在于其快速，便捷，过多的插件反而会使得终端显得臃肿，得不偿失，我们故仅添加必要的插件。
首先安装oh-my-zsh，打开~/.zshrc文件找到plugins=( git )，这里是我们已经启用了那些插件.如果想要启用某个插件，装好之后直接修改

> plugins = (插件A 插件B 插件C)



1. zsh-autosuggestions
非常好用的一个插件，会记录你之前输入过的所有命令，并且自动匹配你可能想要输入命令，然后按→补全

2. zsh-syntax-highlighting
这个插件直接在输入过程中就会提示你，当前命令是否正确，错误红色，正确绿色



# 6. vscode 配置
在vscode中有可能会出现乱码的情况，这是因为终端的字体没有设置好。在设置搜索中，打开setting.json文件，加入
```bash
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.fontFamily": "Meslo LG L DZ for Powerline",
```

# 7. 拓展
## 7.1 不同环境的profile切换
在开发过程，我们会经常掉用到ssh，连接服务器进行开发，此时会产生如下需求：

**希望在不同的环境，自动切换对应的profile。**

比如，我们在local的时候能够使用local profile， 在ssh连接 服务器时，能切换到开发profile。

### 7.1.1 构建profiles
根据我们需求，构建指定profile。
例如笔者在开发过程中，会存在本地，与开发环境两种环境，所以构造了三种不同的profile（Mars_Default, Mars_Server, Mars_other）,其中profile命名可以自行定义。
<center>
<img src="profiles.jpg" width=80% >
</center>

### 7.1.2 修改域名映射

实现不同环境自动切换profile的原理，为识别server名中指定部分，触发profile的切换。所以我们需要修改本地的域名映射, 添加上前缀即可。
```shell
vi /etc/hosts
# hosts example
# [server ip1] ser_spider
# [server ip2] ser_stats1
```


### 7.1.3 创建custom shell 脚本
进入路径, 创建```iTrem2-ssh.zsh```，其中有样例```example.zsh```可做参考。
```shell
cd ~/.oh-my-zsh/custom
```
往```iTrem2-ssh.zsh```文件写入脚本，注意其中Mars_Default, Mars_Server, Mars_other 需要根据自身需求进行替换。

```python
## reference: https://gist.github.com/erangaeb/78614828234323f57b328b61a588209e#file-itrem2-ssh-zsh
# tabc <profile name> do the profile change
function tabc() {
  NAME=$1; if [ -z "$NAME" ]; then NAME="Mars_Default"; fi 
  # if you have trouble with this, change
  # "Default" to the name of your default theme
  echo -e "\033]50;SetProfile=$NAME\a"
}

# reset the terminal profile to Default  when exit from the ssh session
function tab-reset() {
    NAME="Mars_Default"
    echo -e "\033]50;SetProfile=$NAME\a"
}

# selecting different terminal profile according to ssh'ing host
# tabc <profile name> do the profile change
#   1. Production profile to production server (ssh eranga@production_box) 
#   2. Staging profile to staging server(ssh eranga@staging_box) 
#   3. Other profile to any other server(test server, amazon box etc)
function colorssh() {
    if [[ -n "$ITERM_SESSION_ID" ]]; then
        trap "tab-reset" INT EXIT
        if [[ "$*" =~ "ser_*" ]]; then
            tabc Mars_Server
        else
            tabc Mars_other
        fi
    fi
    ssh $*
}
compdef _ssh tabc=ssh

# creates an alias to ssh
# when execute ssh from the terminal it calls to colorssh function
alias ssh="colorssh"
```

### 7.1.4 效果
<center>
<img src="local_profile.jpg" width=80% >
</center>

<center>
<img src="ser_profile.jpg" width=80% >
</center>

## 自定义oh-my-zsh 颜色
###  自定义提示项颜色
在使用iterm2的过程，出现了如下的问题。
在pycharm， vscode中集成了zsh, 由于iterm2默认背景为黑色，而pycharm，vscode背景颜色为白色，导致提示项（标签以及hostname）的颜色不协调。
1. 原iterm2颜色
<center>
<img src="defalut_zsh_color.jpg" width=80% >
</center>
2. 原pycharm终端颜色
<center>
<img src="defalut_pycahrm_zsh_color.jpg" width=80% >
</center>
3. 原vscode终端颜色
<center>
<img src="default_vscode_zsh_color.jpg" width=80% >
</center>

希望能够对提示项的颜色自定义。

修改 ~/.oh-my-zsh/themes/agnoster.zsh-theme 文件（该文件取决与主题选择）中prompt_context() ，将原black修改为自定颜色代码(Xterm Number，Xterm Name皆可name注意大小写)，参考[256 Colors Cheat Sheet](https://www.ditig.com/256-colors-cheat-sheet).
```shell
# the theme file depend on your customer theme
vi ~/.oh-my-zsh/themes/agnoster.zsh-theme

### Prompt components
# Each component will draw itself, and hide itself if no information needs to be shown

# Context: user@hostname (who am I and where am I)
prompt_context() {
  if [[ "$USERNAME" != "$DEFAULT_USER" || -n "$SSH_CLIENT" ]]; then
      # prompt_segment black default "%(!.%{%F{black}%}.)%n@%m"
     prompt_segment cyan default "%(!.%{%F{black}%}.)%n@%m"
  fi
}
```

修改完成后，调用```source ~/.zshrc```激活设置。

笔者选用cyan，效果如下：
1. iterm2 终端效果
<center>
<img src="beautify_zsh.jpg" width=80% >
</center>
2. vscode 效果
<center>
<img src="beautify_vscode_zsh.jpg" width=80% >
</center>

## 自定义hostname颜色
虽然完成了提示项颜色的自定义，但是发现在终端中cyan背景和绿色字体显示效果不佳，希望能将hotsname修改为黑色。修改hostname与修改提示项颜色类似，在原prompt_context()，中```%n@%m```代表用户名与主机名，在该项前添加颜色项%F{custom_color}即可，笔者最终代码如下：
```shell
# the theme file depend on your customer theme
vi ~/.oh-my-zsh/themes/agnoster.zsh-theme

### Prompt components
# Each component will draw itself, and hide itself if no information needs to be shown

# Context: user@hostname (who am I and where am I)
prompt_context() {
  if [[ "$USERNAME" != "$DEFAULT_USER" || -n "$SSH_CLIENT" ]]; then
      # prompt_segment black default "%(!.%{%F{black}%}.)%n@%m"
     prompt_segment cyan default "%(!.%{%F{black}%}.)%F{black}%n@%m"
  fi
}
```
修改完成后，调用```source ~/.zshrc```激活设置,最终iterm效果如下：
<center>
<img src="beautify_hostname.jpg" width=80% >
</center>



# 参考文档：
[1. Iterm2官方文档以及下载链接](https://iterm2.com/)

[2. mac之 iTerm2 + Oh My Zsh 终端安装教程](https://segmentfault.com/a/1190000039834490)

[3. oh-my-zsh theme主题方案](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)

[4. oh-my-zsh 隐藏主机名](https://0clickjacking0.github.io/2020/10/04/oh-my-zsh%E9%9A%90%E8%97%8F%E7%94%A8%E6%88%B7%E5%90%8D%E6%88%96%E8%80%85%E4%B8%BB%E6%9C%BA%E5%90%8D/)

[5. Bash shell / Zsh 里修改前缀 (隐藏用户@主机，添加Git分支名称)](https://www.jianshu.com/p/ee442cb4d6c2)

[6.macOS使用oh-my-zsh美化Terminal,iTerm2,VSCode命令行终端配置教程](https://www.ioiox.com/archives/34.html)

[7. Change terminal color when SSH
](https://medium.com/rahasak/change-terminal-color-when-ssh-e2a13ccee681)