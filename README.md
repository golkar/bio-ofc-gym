# Bio-OFC gym implementation and Gym-Fly environment

This repository includes the gym compatible implementation of the Bio-OFC algorithm from the paper "Neural optimal feedback control with local learning rules". The implementation used for plots other than the Gym-Fly environment are provided in a different folder.

### Installation
For installation instructions in a Conda environment and environment dependencies see installation.txt. 

### Gym environments
A gym environment for delayed linear dynamical systems is implemented in delayed_linear_systems.py. For a demo of this see Bio_OFC linear system demo.ipynb.

A gym environment simulating winged flight in a 2-d environment with simplified dynamics is implemented in gym-fly. For a demo of this see Bio_OFC gym-fly demo.ipynb. A video demo of the gym-fly environment training with Bio-OFC can be found under gym-fly-demo.mp4.