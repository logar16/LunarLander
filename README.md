# Lunar Lander
#### By Logan Jones

This is adapted from a project I did for my master's degree in the GA Tech OMSCS program.
The goal was to solve the [Lunar Lander (v2) environment](http://gym.openai.com/envs/LunarLander-v2/) provided with the OpenAI Gym.
I was very excited about the semi-recent advancement of DeepMind's Deep Q-Networks, 
and so I did a custom implementation built only using the DQN paper "Human-level control through deep reinforcement learning." 
for reference (Mnih Volodymyr et al. 2015).

At the time, I used Keras (and Python 2.7), but I have since grown more fond of PyTorch (and Python 3), 
so this is essentially a port of that project.  I made several improvements in how the exploration is done and how the results are recorded.
I also noticed a big speedup in processing as it seems PyTorch/Python 3 is much better at parallel processing than Keras/Python 2.7.

See [a video](https://github.com/logar16/LunarLander/blob/master/figures/Flight%20Demonstration.mp4) of a simple, randomly picked agent configuration trained for 15,000 episodes and flying really good (or just enjoy the shorter clip below).

![video](docs/FlightDemo.gif)

Note that the agent is only given 8 inputs (floating point numbers stacked over 3 frames) and has 4 actions it can take.  
The fact that it can identify where it is and how to land itself is pretty impressive.

## Agent Performance
If you use my pretrained models, they had the following outcomes

#### Linear Model for 15,000
![linear15000](figures/training15000_20-42-23.png)

#### Memory (LSTM) Model for 25,000
![memory25000](figures/training25000_09-33-03.png)
-----
As you can see, the models quickly learn how to land the craft and achieve a good score within the first 1,500 episodes.  The agent falls like a leaf as it overcompensates for tilting.
However, they soon thereafter learn that they can avoid really bad crashes by flying until the end of the episode.
This is annoying both because it slows down episodes and eliminates strong, positive rewards. 
I at first thought that I had an overfitting problem on my hands, but when I eventually just let the training run for 25,000 steps, 
I found that the agent could eventually learn how to land again, but with the added ability to control its flight much better this time around.
Performance essentially plateaued at about 15,000 episodes and I never trained past 25,000.
The agent still isn't perfect, as it has occassional crashes and will sometimes run a side rocket after landing to straighten itself, but which delays the end of the episode and reduces reward.

An enhancement that I added that is different from DQN was the fact that I only add a fraction of the memories to be saved to the experience buffer.  
I was saving them in buffered form (for simpler replay) and so to avoid redundancy and to focus on memories with most information,
I only saved them if they were a terminal step else with a coin flip (50% chance).  This ensured important memories were kept and others faded out (could have done 67% probably with the 3 frame buffer, but oh well).
  


## Installation
You should be able to install all the dependencies by (creating a virtual environment) 
and then running the following command:

```shell script
pip install -r .\requirements.txt
``` 

Note that I used a conda environment and then used pip for anything that conda didn't support.

If installing Box2D (for the gym env) gives you issues and you are on Windows,
check out [this article](https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5).


## Training

You will need the following directories to be present or errors will be thrown
* `figures/`
* `models/` 
    * `configs/`
    * `networks/`
    
To do a random search of hyperparameters and model structures use the following command:
```shell script
python main.py -e -i 1800
```
Where `-e` indicates exploration and `-i` is number of episodes to train for.  
Optionally add `-v` for more detailed progress reports.

Note that only models with a good mean score will be saved, and they are saved at checkpoints or at the end of training.
Figures should be populated automatically. This random selection isn't designed to use the RNN model, just the linear one (which gets pretty similar performance I found)
I put here 1800 iterations as that can help determine if the agent is learning how to land early on, an important first step.  Once that is established, you should train for about 15,000 episodes.

To train a specific configuration, create a config file (see others for examples or the config section below) 
specifying what you want the hyperparameters and model architecture to be.
Add that config to the `models/configs/` directory and run it with the following command:

```shell script
python main.py -c models/configs/best.yaml -i 15000 -t
```
Where `-c` is the config file, `-t` indicates training should happen, and `-i` is number of episodes (iterations) to train for.
The trained file should be saved for you using the time started as part of the filename.

To evaluate a saved model, combine both the proper configuration 
(if the saved agent was found during exploration, you will need to make a config file matching some or all of the trained hyperparameters, sorry)
and the PyTorch neural network weights file using the following command

```shell script
python main.py -c models/configs/best.yaml -i 100 -l models/networks/best_23-14-22.pt 
```

Where the `-l` is the model save-file to load. `-i` indicates the number of evaluations to run against the loaded model.
Note that you could add `-t` to do additional training.

Finally, if you want to watch it fly, you can add the `-r` flag for "render".  
You will need the `Box2D` library working with `pyglet` in order for it to render properly.


### Config Files
A standard config file looks like this

```yaml
agent:    # See the agents.py file and DQNAgent class `__init__()` method for more details
  gamma: 0.9992
  rar: 1.05
  rar_decay: 0.99997
  memory_size: 200000
  minibatch_size: 2048
  replay_freq: 40
  target_update: 80000
  optim_type: torch.optim.adadelta.Adadelta   # Use the fully specified class name
  optim_args: 
    lr: 1   # Adadelta has a higher learning rate.  Adam would be more like 0.001
  device: null  # `null` is check for cuda and use if available.  Other options are "cpu" and "cuda"

# Define the model.  Currently only the number of units in the two linear layers are needed
model:
  units: [64, 16] 
  # memory: LSTM   #Memory is optional and should be either "GRU" or "LSTM"  
```

As noted in the configuration example, you can use either "cuda" or "cpu" for hosting the the PyTorch tensors.  
I found that (for my setup) CPU is usually faster (not always) but it takes up a lot of CPU (about 50% or 2 of my 4 cores).
When I use CUDA, I only increase CPU usage by about 15% but had about a 20-30% reduction in speed for linear models.  
However, I was able to run three instances for the price of one (CPU-wise), and thus was able to cover more ground. 