import sys
from nn_manim import *

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

sys.path.append('/Users/mark/Projects/Neural-Network-withmath/')
from network import NeuralNetwork
from dense import Dense
from activations import Sigmoid

class MnistNeuralNetwork(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def construct(self):

        # create neural network model
        nn = NeuralNetwork()
        nn.add(Dense(28 * 28, 36))
        nn.add(Sigmoid())
        nn.add(Dense(36, 25))
        nn.add(Sigmoid())
        nn.add(Dense(25, 16))
        nn.add(Sigmoid())
        nn.add(Dense(16, 10))
        nn.add(Sigmoid())
        self.nn = nn


        # Iterate through the elements of the network to get the dimensions for drawing
        layer_dims = []
        for layer in nn.network:
            # Check if the layer is an object instance of type Dense
            if isinstance(layer, Dense):
                # if layer_dims is empty, add the input and output dimensions
                if not layer_dims:
                    layer_dims.append(layer.weights.shape[1]) # input dimension element
                    layer_dims.append(layer.weights.shape[0]) # output dimension element
                # if layer_dims is not empty, add the output dimension
                else:
                    layer_dims.append(layer.weights.shape[0])
            else:
                pass
        self.layer_dims = layer_dims

        # Create the neural network visualization
        myManimNetwork = NeuralNetworkMobject(layer_dims)
        myManimNetwork.label_input_neurons('x')
        myManimNetwork.label_output_neurons('\hat{y}')
        myManimNetwork.label_output_neurons_with_text([str(i) for i in range(layer_dims[-1])])
        myManimNetwork.scale(0.7)
        myManimNetwork.to_edge(RIGHT)
        self.myManimNetwork = myManimNetwork

        # Create the 2D image grid visualization of input data
        square_side_length = math.floor(math.sqrt(layer_dims[0]))
        myGrid = NeuralNetwork2DInputMobject(square_side_length, square_side_length)
        myGrid.scale_to_fit_left_third()
        self.myGrid = myGrid

        self.play(ShowCreation(myManimNetwork), FadeIn(myGrid))
        self.embed()

    def usr_load_data(self, train_size=60000, test_size=10000):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, y_train = self.nn.preprocess_data(x_train, y_train, train_size)
        x_test, y_test = self.nn.preprocess_data(x_test, y_test, test_size)
        
        print(f'data loaded:{x_train.shape}, {y_train.shape}')
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def usr_train(self, epochs = 100, learning_rate = 0.1, animate_weights=False, epoch_anim_interval=10):
        self.nn.epochs = epochs
        self.nn.learning_rate = learning_rate
        self.nn.add_scene(self, animate_weights, epoch_anim_interval)
        self.nn.train(self.x_train, self.y_train, epochs=self.nn.epochs, learning_rate=self.nn.learning_rate)

    def usr_predict(self, x_test, y_test):
        self.nn.predict(x_test)
        print(f'output:{y_test}')

    def go_animate_grid(self, x):

        # Reshape the input data (728 array of values representing a 28x28 image) into cols and rows
        square_side_length = math.floor(math.sqrt(x.shape[0]))
        x = x.reshape(square_side_length, square_side_length)

        # Animate the input data grid
        animations_in = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # Access the corresponding square in the grid and set the grey-scale color based on the pixel value
                color = interpolate_color(WHITE, BLACK, x[i, j])
                if tuple(hex_to_rgb(self.myGrid.get_cell(i,j).get_fill_color())) != tuple(color_to_rgb(color)):
                    animations_in.append(ApplyMethod(self.myGrid.get_cell(i,j).set_fill, {'color': color, 'opacity': 1}))
        
        self.play(AnimationGroup(*animations_in, run_time=0.001))
                    
    def go_animate_weights(self, layer_index, weights):        
        # input_neuron_index is available in myManimNetwork.layers[layer_index].indices_to_draw
        input_indices = self.myManimNetwork.layers[layer_index].indices_to_draw        
        output_indices = self.myManimNetwork.layers[layer_index + 1].indices_to_draw
        
        # Animate the output neurons and weights in an AnimationGroup
        animations_in = []
        animations_out = []
        for output_neuron_index in output_indices:
            for input_neuron_index in input_indices:
                # Access the weight
                weight = weights[output_neuron_index, input_neuron_index]

                # Normalize the weight to the range [-1, 1]
                normalized_weight = (weight - weights.min()) / (weights.max() - weights.min()) * 2 - 1
                
                # Interpolate color based on the normalized weight
                color = interpolate_color(BLUE, RED, (normalized_weight + 1) / 2) 

                # Access the corresponding edge (weight) if it exists
                if output_neuron_index in self.myManimNetwork.layers[layer_index].real_neurons[input_neuron_index].edge_to_neuron:
                    edge = self.myManimNetwork.layers[layer_index].real_neurons[input_neuron_index].edge_to_neuron[output_neuron_index]

                    original_weight_stroke_width = edge.get_stroke_width()
                    animations_in.append(ApplyMethod(edge.set_stroke, {'width': 1.5, 'color': color}))
                    animations_out.append(ApplyMethod(edge.set_stroke, {'width': original_weight_stroke_width}))

            # print(f"{layer_index}, {output_neuron_index}")
            # original_neuron_color = self.myManimNetwork.layers[layer_index + 1].real_neurons[output_neuron_index].rendered_mobject.get_color()
            # original_neuron_stroke_width = self.myManimNetwork.layers[layer_index + 1].real_neurons[output_neuron_index].rendered_mobject.get_stroke_width()
            # animations_in.append(ApplyMethod(self.myManimNetwork.layers[layer_index + 1].real_neurons[output_neuron_index].rendered_mobject.set_color, color))
            # animations_in.append(ApplyMethod(self.myManimNetwork.layers[layer_index + 1].real_neurons[output_neuron_index].rendered_mobject.set_stroke, {'width': 3}))
            # animations_out.append(ApplyMethod(self.myManimNetwork.layers[layer_index + 1].real_neurons[output_neuron_index].rendered_mobject.set_color, original_neuron_color))
            # animations_out.append(ApplyMethod(self.myManimNetwork.layers[layer_index + 1].real_neurons[output_neuron_index].rendered_mobject.set_stroke, {'width': original_neuron_stroke_width}))
        
        # Animate all edges and neurons for the current layer at the same time
        self.play(AnimationGroup(*animations_in, run_time=0.001))
        self.wait(.01)
        self.play(AnimationGroup(*animations_out, run_time=0.001))

    def go_animate_neurons(self, layer_index, output):
        # input_neuron_index is available in myManimNetwork.layers[layer_index].indices_to_draw
        input_indices = self.myManimNetwork.layers[layer_index].indices_to_draw
        output_indices = self.myManimNetwork.layers[layer_index + 1].indices_to_draw

        # Animate the output neurons and weights in an AnimationGroup
        animations_in = []
        animations_out = []
        for output_neuron_index in output_indices:
            # Access the output neuron
            neuron = self.myManimNetwork.layers[layer_index + 1].real_neurons[output_neuron_index]
            original_neuron_color = neuron.rendered_mobject.get_fill_color()
            original_neuron_stroke_width = neuron.rendered_mobject.get_stroke_width()

            # Normalize the output value to the range [0, 1]
            normalized_output = (output[output_neuron_index] - output.min()) / (output.max() - output.min())
            
            # Interpolate color based on the normalized output
            color = interpolate_color(WHITE, RED, normalized_output)

            # Animate the neuron
            animations_in.append(ApplyMethod(neuron.rendered_mobject.set_fill, {'color': color, 'opacity': 1}))
            # animations_in.append(ApplyMethod(neuron.rendered_mobject.set_stroke, {'width': 3}))
            if layer_index != len(self.layer_dims) - 2:
                animations_out.append(ApplyMethod(neuron.rendered_mobject.set_fill, {'color': original_neuron_color, 'opacity': 1}))
            # animations_out.append(ApplyMethod(neuron.rendered_mobject.set_stroke, {'width': original_neuron_stroke_width}))
        
        # Animate all neurons for the current layer at the same time
        self.play(AnimationGroup(*animations_in, run_time=0.001))
        # self.wait(.01)
        self.play(AnimationGroup(*animations_out, run_time=0.1))

if __name__ == '__main__':
    scene01 = MnistNeuralNetwork()
    scene01.run()