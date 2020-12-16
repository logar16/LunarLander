# Lunar Lander
#### By Logan Jones

This is adapted from a project I did for my master's degree in the GA Tech OMSCS program.


## Installation
You should be able to install all the dependencies by (creating a virtual environment) 
and then running the following command:

```shell script
pip install -r .\requirements.txt
``` 

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
python main.py -e -i 2000
```
Where `-e` indicates exploration and `-i` is number of episodes to train for.  
Optionally add `-v` for more detailed progress reports.

Note that only models with a good mean score will be saved, and they are saved at checkpoints or at the end of training.
Figures should be populated automatically.


To train a specific configuration, create a config file (see others for examples) 
specifying what you want the hyperparameters and model architecture to be.
Add that config to the `models/configs/` directory and run it with the following command:

```shell script
python main.py -c models/configs/best.yaml -i 1500 -t
```
Where `-c` is the config file, `-t` indicates training should happen, and `-i` is number of episodes (iterations) to train for.
The trained file should be saved.

To evaluate a saved model, combine both the proper configuration 
(if the saved agent was found during exploration, you will need to make a config file matching some or all of the trained hyperparameters, sorry)
and the load file using the following command

```shell script
python main.py -c models/configs/best.yaml -i 100 -l models/networks/best_23-14-22.pt 
```

Where the `-l` is the model save-file to load. `-i` indicates the number of evaluations to run against the loaded model.
Note that you could add `-t` to do additional training.

