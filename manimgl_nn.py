from manimlib import *
from manimlib import Tex
import itertools as it

class myNeuralNetwork(Scene):
    def __init__(self, layer_sizes = [16, 8, 4, 2], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_sizes = layer_sizes

    def construct(self):
        myNetwork = NeuralNetworkMobject(self.layer_sizes)
        myNetwork.label_input_neurons('x')
        myNetwork.label_output_neurons('\hat{y}')
        myNetwork.label_output_neurons_with_text([str(i) for i in range(10)])

        myNetwork.scale(0.7)
        self.play(ShowCreation(myNetwork))
        # self.embed()

class NeuronReal:
    def __init__(self, index, index_of_rendered_mobject=None, is_rendered=False, *args, **kwargs):
        self.index = index
        self.index_of_rendered_mobject = index_of_rendered_mobject
        self.edge_to_neuron = {}  # Dictionary to map edge to target neuron index
        self.rendered_mobject = NeuronRendered(index=index_of_rendered_mobject, real_index=index, *args, **kwargs) if is_rendered else None
        self.edges_in = []
        self.edges_out = []

class NeuronRendered(Circle):
    def __init__(self, index, real_index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.real_index = real_index
        self.edges_in = VGroup()
        self.edges_out = VGroup()

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

        self.build_neuralnet_layers()
        self.build_and_add_edges()
        self.add_to_back(self.layers)

    def build_neuralnet_layers(self):
        layers = VGroup(*[
            self.build_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        if self.include_output_labels:
            self.label_output_neurons_with_text()

    def determine_neuron_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.output_neuron_color
        if index == 0:
            return self.input_neuron_color
        else:
            return self.hidden_layer_neuron_color

    def build_layer(self, size, index=-1):
        layer = VGroup()
        layer.index = index  # Add this line to set the index attribute

        real_neurons = []
        rendered_neurons = VGroup()

        # determine which neurons to draw based on the size of the layer and max_shown_neurons
        if size > self.max_shown_neurons:
            half_max = self.max_shown_neurons // 2
            first_range = list(range(half_max))
            second_range = list(range(size - self.max_shown_neurons + half_max, size))
            indices_to_draw = first_range + second_range
        else:
            indices_to_draw = list(range(size))
        layer.indices_to_draw = indices_to_draw

        # Create the "real" neurons
        for i in range(size):
            
            # adds "real" neurons to the layer
            neuron_real = NeuronReal(
                index=i,
                index_of_rendered_mobject=indices_to_draw.index(i) if (i in indices_to_draw) else None,
                is_rendered=(i in indices_to_draw),
                radius=self.neuron_radius,
                stroke_color=self.determine_neuron_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            real_neurons.append(neuron_real)
            
            # adds any corresponding "rendered" neurons to the layer
            if i in indices_to_draw:
                rendered_neurons.add(neuron_real.rendered_mobject)

        rendered_neurons.arrange(DOWN, buff=self.neuron_to_neuron_buff)
        layer.real_neurons = real_neurons
        layer.rendered_neurons = rendered_neurons
        layer.add(rendered_neurons)
        

        if size > self.max_shown_neurons:
            dots = Tex("\\vdots")
            dots.move_to(rendered_neurons)
            VGroup(*rendered_neurons[:len(rendered_neurons) // 2]).next_to(dots, UP, MED_SMALL_BUFF)
            VGroup(*rendered_neurons[len(rendered_neurons) // 2:]).next_to(dots, DOWN, MED_SMALL_BUFF)
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def build_and_add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.rendered_neurons, l2.rendered_neurons):
                if n1.real_index < self.layer_sizes[l1.index] and n2.real_index < self.layer_sizes[l2.index]:
                    edge = self.create_edge(n1, n2)
                    edge_group.add(edge)
                    n1.edges_out.add(edge)
                    n2.edges_in.add(edge)
                    # Add edge to the dictionary
                    self.edges_dict[(l1.index, n1.real_index, n2.real_index)] = edge
                    # Add edge to the to_neuron dictionary in NeuronReal
                    l1.real_neurons[n1.real_index].edge_to_neuron[n2.real_index] = edge
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def create_edge(self, neuron1, neuron2):
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

    def retrieve_edge_by_indices(self, start_layer, start_neuron, end_neuron):
        return self.edges_dict.get((start_layer, start_neuron, end_neuron), None)

    def label_input_neurons(self, l):
        self.output_labels = VGroup()
        for neuron in self.layers[0].rendered_neurons:
            label = Tex(f"{l}_"+"{"+f"{neuron.real_index}"+"}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_output_neurons(self, l):
        self.output_labels = VGroup()
        for neuron in self.layers[-1].rendered_neurons:
            label = Tex(f"{l}_"+"{"+f"{neuron.real_index}"+"}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_output_neurons_with_text(self, outputs):
        self.output_labels = VGroup()
        for neuron in self.layers[-1].rendered_neurons:
            label = Tex(outputs[neuron.real_index])
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + (label.get_width() / 2))*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    def label_hidden_layer_neurons(self, l):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for neuron in layer.rendered_neurons:
                label = Tex(f"{l}_{neuron.real_index}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)
    
class my2DInput(Scene):
    def __init__(self, rows = 3, cols = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = rows
        self.cols = cols

    def construct(self):
        myGrid = NeuralNetwork2DInputMobject(self.rows, self.cols)
        self.play(ShowCreation(myGrid))
        self.embed()

class NeuralNetwork2DInputMobject(VGroup):
    def __init__(self, rows, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = rows
        self.cols = cols
        self.grid = []
        self.create_grid(rows, cols)

    def create_grid(self, rows, cols):
        for i in range(rows):
            row = VGroup()
            for j in range(cols):
                cell = Square(side_length=1, fill_color = None, fill_opacity = 1, stroke_color=GREEN, stroke_width=1, stroke_opacity=.5)
                # Invert the y-coordinate to place row 0 at the top
                cell.move_to(np.array([j - cols / 2, (rows / 2 - i - 1), 0]))
                row.add(cell)
            self.add(row)
            self.grid.append(row)
        self.center()
        self.scale_to_fit_left_third()

    def scale_to_fit_left_third(self):
        # Scale the grid to fit within the left-third of the canvas
        self.scale(0.4)
        # Move the grid to the left-third of the canvas
        self.to_edge(LEFT)

    def get_cell(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        else:
            raise IndexError("Grid index out of range")

class myMnist(Scene):
    def __init__(self, layer_sizes = [28*28, 100, 50, 20, 10], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_sizes = layer_sizes  

    def construct(self):
        # Create the neural network visualization
        myNetwork = NeuralNetworkMobject(self.layer_sizes)
        myNetwork.label_input_neurons('x')
        myNetwork.label_output_neurons('\hat{y}')
        myNetwork.label_output_neurons_with_text([str(i) for i in range(10)])
        myNetwork.scale(0.65)
        myNetwork.to_edge(RIGHT)  # Position the neural network on the right

        # Create the 2D image grid visualization
        square_side_length = math.floor(math.sqrt(self.layer_sizes[0]))
        myGrid = NeuralNetwork2DInputMobject(square_side_length, square_side_length)
        myGrid.scale_to_fit_left_third()  # Ensure the grid fits within the left-third

        # Add both elements to the scene
        self.play(ShowCreation(myNetwork), ShowCreation(myGrid))
        self.embed()

    # Instantiate the neural network model


    def usr_load_data(self):
        print("Loading data from MNIST dataset")


    def animate_weights(self, layer_index, weights):
        print(f"Animating weights for layer {layer_index}")

if __name__ == "__main__":

    # scene01 = myNeuralNetwork([28*28, 100, 50, 20, 10])
    # scene01.run()

    # scene02 = my2DInput(28, 28)
    # scene02.run()

    scene03 = myMnist([20,13,11,1])
    scene03.run()