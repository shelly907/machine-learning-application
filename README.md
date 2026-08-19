# Neural Network Implementation in C++

A complete neural network implementation in C++ featuring a graph-based architecture, backpropagation training, and an interactive web-based visualization tool for understanding network computation.

## Features

- **Graph-Based Architecture**: Neural networks built on top of a generic directed graph data structure
- **Multiple Activation Functions**: Support for Identity, ReLU, and Sigmoid activation functions
- **Flexible Training Pipeline**: Train/eval modes with customizable learning rates and batch processing
- **Backpropagation**: Full gradient computation using depth-first traversal
- **CSV Data Loading**: Built-in data loader for training and evaluation from CSV files
- **Model Persistence**: Save and load trained models with weights, biases, and architecture
- **Interactive Visualization**: Step-through debugger for forward pass (predict) and backward pass (backpropagation) computation
- **Web-Based UI**: View network computation in real-time with phase indicators and node/edge highlighting

## Project Structure

```
.
├── main.cpp                 # Entry point and example training code
├── NeuralNetwork.{hpp,cpp}  # Neural network class (inherits from Graph)
├── Graph.{hpp,cpp}          # Generic graph data structure with nodes and connections
├── DataLoader.{hpp,cpp}     # CSV data loading utilities
├── Trace.{hpp,cpp}          # Execution tracing for visualization
├── utility.{hpp,cpp}        # Activation functions and utilities
├── VisualizerDriver.cpp     # Generates trace files for web visualization
├── tdd.{hpp,cpp}            # Unit tests
├── Makefile                 # Build configuration
├── data/                    # Sample datasets (diabetes prediction)
│   ├── diabetes_train.csv   # Training data
│   └── diabetes_test.csv    # Test data
├── models/                  # Pre-trained model architectures
│   ├── diabetes.init        # Diabetes prediction model
│   └── neuralnet_*.init     # Example models
└── web-viz/                 # Web-based visualization
    ├── index.html           # UI
    ├── viewer.js            # Visualization logic
    └── *.trace              # Execution traces
```

## Prerequisites

- **C++17 or later**
- **g++ compiler** (or compatible C++ compiler supporting C++17)
- **Python 3** (for running the web server)
- Any modern web browser

## Building

```bash
# Compile all targets
make

# This produces three executables:
# - neuralnet      : Main application
# - test_neuralnet : Unit tests
# - visualizer     : Trace file generator for web visualization

# Clean build artifacts
make clean
```

## Usage

### Basic Training Example

```cpp
// Load a network architecture from file
NeuralNetwork nn("./models/diabetes.init");
nn.setLearningRate(0.001);

// Load training data
DataLoader dl("./data/diabetes_train.csv");

// Train the network
nn.train();  // Enable gradient accumulation

for (int epoch = 0; epoch < 3; epoch++) {
    for (const auto& instance : dl.getData()) {
        auto predictions = nn.predict(instance);
    }
    cout << "Epoch " << epoch << " accuracy: " 
         << nn.assess("./data/diabetes_test.csv") << endl;
    nn.update();  // Apply accumulated gradients
}

// Evaluate on test set
cout << "Final accuracy: " << nn.assess("./data/diabetes_test.csv") << endl;
```

### Running the Main Program

```bash
# Build and run
make
./neuralnet
```

### Interactive Visualization

The visualization tool lets you step through forward and backward passes:

```bash
# Generate trace files
./visualizer

# Start a local web server
python3 -m http.server 8080 --directory ./web-viz

# Open in browser
# http://localhost:8080
```

**Visualization Controls:**
- **Evaluation/Training Toggle**: Switch between forward pass (predict) and backward pass (backpropagation)
- **Previous/Next**: Navigate through computation steps frame-by-frame
- **Reset**: Return to initial state
- **Phase Indicator**: Shows current computation phase (Predict, Contribute, Update)
- **Highlighting**: 
  - Yellow nodes: Currently active node
  - Purple edges: Active connections
  - Node values and weights displayed in real-time

## API Overview

### NeuralNetwork Class

**Constructors:**
```cpp
NeuralNetwork();                          // Empty network
NeuralNetwork(int size);                  // Preallocate nodes
NeuralNetwork(std::string filename);      // Load from file
```

**Training Methods:**
```cpp
void train();                             // Enable gradient accumulation
void eval();                              // Disable gradient accumulation
void setLearningRate(double lr);          // Set learning rate
std::vector<double> predict(DataInstance instance);  // Forward pass
bool update();                            // Apply accumulated gradients
```

**Evaluation Methods:**
```cpp
double assess(std::string filename);      // Accuracy on CSV dataset
double assess(DataLoader dl);             // Accuracy on DataLoader
void saveModel(std::string filename);     // Save weights and architecture
```

**Configuration:**
```cpp
void setInputNodeIds(std::vector<int> ids);   // Mark input layer nodes
void setOutputNodeIds(std::vector<int> ids);  // Mark output layer nodes
std::vector<int> getInputNodeIds() const;
std::vector<int> getOutputNodeIds() const;
const std::vector<std::vector<int>>& getLayers() const;
```

### Graph Class

The base class for NeuralNetwork, providing node and connection management:

```cpp
NodeInfo* getNode(int id) const;                    // Get node by ID
void updateNode(int id, NodeInfo n);                // Modify node
void updateConnection(int source, int dest, double weight);  // Update edge weight
```

### DataLoader Class

Load and iterate through CSV data:

```cpp
DataLoader(std::string filename);        // Initialize from file
const std::vector<DataInstance>& getData() const;  // Get all instances
```

## Model File Format

Model files (`.init`) define the network architecture and initial weights:

```
<num_nodes> <num_layers>
<layer1_size> <activation_function>
<layer2_size> <activation_function>
...
<num_weights>
<source_id> <dest_id> <weight>
...
```

**Supported Activation Functions:**
- `identity`: f(x) = x
- `ReLU`: f(x) = max(0, x)
- `sigmoid`: f(x) = 1 / (1 + e^(-x))

## CSV Data Format

Input CSV files should have features followed by a label column:

```
feature1,feature2,feature3,...,featureN,label
80.0,0,1,0,25.19,6.6,140,0
54.0,0,0,1,27.32,6.6,80,0
...
```

## Running Tests

```bash
# Compile tests
make test_neuralnet

# Run tests
./test_neuralnet
```

## Implementation Details

### Forward Pass (Prediction)

1. Input values are set in input layer nodes
2. For each subsequent layer:
   - Compute weighted sum: z = Σ(weight × input) + bias
   - Apply activation function: a = activate(z)
   - Pass activations to next layer as inputs
3. Return output layer activations

### Backward Pass (Backpropagation)

1. Compute output error: delta = ∂L/∂output
2. For each layer (reverse order):
   - Propagate deltas backward through edges
   - Accumulate weight gradients: weight_delta = delta × input
   - Accumulate bias gradients: bias_delta = delta
3. Update phase applies: weight -= learning_rate × weight_delta

### Activation Functions

Each node stores both pre- and post-activation values for gradient computation:
- **Pre-activation (z)**: Weighted sum + bias, needed for derivative computation
- **Post-activation (a)**: Activated value, passed to next layer

## Performance Considerations

- **Batch Training**: Gradients accumulate across a batch before updating (configurable batch size)
- **Learning Rate**: Controls update magnitude; typical range: 0.0001 to 0.01
- **Epochs**: Full passes over training data; typically 3-100 depending on problem

## Authors

Shelly Parekh, Paul Clayton


