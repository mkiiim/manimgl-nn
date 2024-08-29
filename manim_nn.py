from manimlib import *
# from manim import MathTex
from manimlib import Tex
import itertools as it

class myNeuralNetwork(Scene):
    def construct(self):
        myNetwork = NeuralNetworkMobject([28*28, 100, 50, 20, 10])
        myNetwork.label_inputs('x')
        myNetwork.label_outputs('\hat{y}')
        myNetwork.label_outputs_text([str(i) for i in range(10)])

        myNetwork.scale(0.65)
        # self.play(Create(myNetwork))
        self.play(ShowCreation(myNetwork))
        self.embed()
        # Add the edge animation
        # self.play(myNetwork.animate_edges())

class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.10,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "output_neuron_color": WHITE,
        "input_neuron_color": YELLOW,
        "hidden_layer_neuron_color": MAROON,
        "neuron_stroke_width": 2,
        "neuron_fill_color": GREEN,
        "edge_color": GREY,
        "edge_stroke_width": 0.25,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 2,
        "max_shown_neurons": 24,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 1
    }

    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network

        # Initialize the dictionary to store edges
        self.edges_dict = {}

        self.neuron_radius = self.CONFIG["neuron_radius"]
        self.neuron_to_neuron_buff = self.CONFIG["neuron_to_neuron_buff"]
        self.layer_to_layer_buff = self.CONFIG["layer_to_layer_buff"]
        self.output_neuron_color = self.CONFIG["output_neuron_color"]
        self.input_neuron_color = self.CONFIG["input_neuron_color"]
        self.hidden_layer_neuron_color = self.CONFIG["hidden_layer_neuron_color"]
        self.neuron_stroke_width = self.CONFIG["neuron_stroke_width"]
        self.neuron_fill_color = self.CONFIG["neuron_fill_color"]
        self.edge_color = self.CONFIG["edge_color"]
        self.edge_stroke_width = self.CONFIG["edge_stroke_width"]
        self.edge_propogation_color = self.CONFIG["edge_propogation_color"]
        self.edge_propogation_time = self.CONFIG["edge_propogation_time"]
        self.max_shown_neurons = self.CONFIG["max_shown_neurons"]
        self.brace_for_large_layers = self.CONFIG["brace_for_large_layers"]
        self.average_shown_activation_of_large_layer = self.CONFIG["average_shown_activation_of_large_layer"]
        self.include_output_labels = self.CONFIG["include_output_labels"]
        self.arrow = self.CONFIG["arrow"]
        self.arrow_tip_size = self.CONFIG["arrow_tip_size"]
        self.left_size = self.CONFIG["left_size"]
        self.neuron_fill_opacity = self.CONFIG["neuron_fill_opacity"]

        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        # layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        if self.include_output_labels:
            self.label_outputs_text()

    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.output_neuron_color
        if index == 0:
            return self.input_neuron_color
        else:
            return self.hidden_layer_neuron_color

    def get_layer(self, size, index=-1):
        layer = VGroup()
        layer.index = index  # Add this line to set the index attribute
        n_neurons = size
        neurons = VGroup()
        if n_neurons > self.max_shown_neurons:
            half_max = self.max_shown_neurons // 2
            indices_to_draw = list(range(half_max)) + list(range(n_neurons - half_max, n_neurons))
        else:
            indices_to_draw = list(range(n_neurons))

        for i in indices_to_draw:
            neuron = Circle(
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            neuron.true_index = i
            neurons.add(neuron)

        # neurons.arrange_submobjects(DOWN, buff=self.neuron_to_neuron_buff)
        neurons.arrange(DOWN, buff=self.neuron_to_neuron_buff)
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > self.max_shown_neurons:
            dots = Tex("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(dots, UP, MED_SMALL_BUFF)
            VGroup(*neurons[len(neurons) // 2:]).next_to(dots, DOWN, MED_SMALL_BUFF)
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                if n1.true_index < self.layer_sizes[l1.index] and n2.true_index < self.layer_sizes[l2.index]:
                    edge = self.get_edge(n1, n2)
                    edge_group.add(edge)
                    n1.edges_out.add(edge)
                    n2.edges_in.add(edge)
                    # Add edge to the dictionary
                    self.edges_dict[(l1.index, n1.true_index, n2.true_index)] = edge
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def get_edge(self, neuron1, neuron2):
        if self.arrow:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=self.edge_color,
                stroke_width=self.edge_stroke_width,
                tip_length=self.arrow_tip_size
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

    def get_specific_edge(self, start_layer, start_neuron, end_neuron):
        return self.edges_dict.get((start_layer, start_neuron, end_neuron), None)

    def label_inputs(self, l):
        self.output_labels = VGroup()
        for neuron in self.layers[0].neurons:
            label = Tex(f"{l}_"+"{"+f"{neuron.true_index}"+"}")
            # label.set(height=0.3 * neuron.height)
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_outputs(self, l):
        self.output_labels = VGroup()
        for neuron in self.layers[-1].neurons:
            label = Tex(f"{l}_"+"{"+f"{neuron.true_index}"+"}")
            # label.set(height=0.4 * neuron.height)
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_outputs_text(self, outputs):
        self.output_labels = VGroup()
        for neuron in self.layers[-1].neurons:
            label = Tex(outputs[neuron.true_index])
            # label.set(height=0.75*neuron.height)
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            # label.shift((neuron.width + label.width/2)*RIGHT)
            label.shift((neuron.get_width() + (label.get_width() / 2))*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_hidden_layers(self, l):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for neuron in layer.neurons:
                label = Tex(f"{l}_{neuron.true_index}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)

    def animate_edges(self):
        animations = []
        for edge_group in self.edge_groups:
            for edge in edge_group:
                animations.append(
                    edge.animate.scale(1.2).set_stroke(width=edge.get_stroke_width() * 1.2)
                )
                animations.append(
                    edge.animate.scale(1/1.2).set_stroke(width=edge.get_stroke_width() / 1.2)
                )
        return AnimationGroup(*animations, lag_ratio=0.1)