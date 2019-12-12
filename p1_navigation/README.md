[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This project trains an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Pre-requisites (Windows)

1. You  will need the following pre-requisite software installed.
    - Miniconda: Miniconda is a free minimal installer for conda.
    - Microsoft Visual Studio - C++ Build Tools: As a minimum you'll need "C++ CMake tools for windows" and "Windows SDK"
    - GIT: I chose to install smartgit

### Getting Started

1. Launch miniconda command prompt and create a new environment.

        conda create --name drlnd python=3.6
        activate drlnd`

2. Install swig

        conda install swig
    
3. Install OpenLAI Gym

        pip install gym
        pip install gym[box2d]
        pip install gym[classic_control]

4. Install pytorch using command from pytorch.org. Note that we need an older version to match requirements for this project.     

        conda install pytorch=0.4.0 -c pytorch

5. Clone the project to local machine

        cd %HOMEPATH%\
        git clone https://github.com/bheemboy/drlnd.git

6. Install additional project pre-requisites

        cd %HOMEPATH%\drlnd\python
        pip install .

7. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
8. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

1. Launch miniconda prompt to launch Jupyter notebook
        
        activate drlnd
        cd  %HOMEPATH%\drlnd\p1_navigation
        Jupyter Notebook

2. Click on `Navigation.ipynb` and follow the instructions to get started with training your own agent!  
