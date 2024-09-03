# Manim routines for neural networks

## manim_nn.py
  
library of routines to draw neural networks using manim
  
## manim_nn_mnist.py
  
implementation of the `manim_nn` routines to illustrate mnist at work, with animations for training and prediction - but SLOW, depending on your hardware. i'm using this on an M1 macbook air.  
  
## User commands for `manim_nn_mnist.py`
  
```ipython
# load mnist dataset
In [1]: self.usr_load_data()

# train neural net, animating only the last training data element at every (epochs % epoch_anim_interval) ... because SLOW
In [2]: self.usr_train(epochs = 3, animate_weights = True, epoch_anim_interval = 1)

# predict using mnist test dataset
In [21]: self.usr_predict(self.x_test[15],self.y_test[15])
```

## Interactively addressing neurons and edges
  
Examples:

```ipython
# animate the 784th neuron of the 1st layer
In [1]: self.play(myNetwork.layers[0].real_neurons[783].animate.shift(LEFT*3))

# animate the edge (weight) between the 784th neuron of the 1st layer and the 4th neuron of the next (2nd) layer
In [2]: self.play(myNetwork.layers[0].real_neurons[783].edge_to_neuron[3].animate.shift(LEFT*3)) 
```

## Manim and Environment Setup and Issues
  
Key requirements:
- miniforge 3.24.0 (python 3.10 default)
- ManimGL v1.6.1
- ffmpeg (via brew)
- latex (i'm using mactex via pkg installer)
- for mnist demo (`manimgl_nn_mnist.py`), you'll need my neural-net repository [Neural-Network](https://github.com/mkiiim/Neural-Network)
  
NOTE - Had to reinstall miniforge 3.24.0 in order to solve environment and dependency issues including:
- `python` version issues (suggested use of `python==3.10` which is default for miniforge `3.24.0` )
- `manimpango` and `cairo.h` issues
- numpy issues (suggested version 1.24 to resolve [ValueError: operands could not be broadcast together with shapes (24,3) (0,3)](https://github.com/3b1b/manim/issues/2053#top))
  
### Create miniforge conda environment

```zsh
conda create --name manimgl --clone base
conda activate manimgl
conda install numpy==1.24
conda install keras
conda install tensorflow
pip install manimgl
```