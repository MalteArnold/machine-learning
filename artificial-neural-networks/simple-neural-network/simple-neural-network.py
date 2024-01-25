import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

# Input and output data for XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a Sequential model
model = Sequential()

# Add a hidden layer with 3 neurons, tanh activation, and uniform weight initialization
model.add(
    Dense(
        3,
        activation="tanh",
        input_dim=2,
        name="hidden_layer",
        kernel_initializer="uniform",
        bias_initializer="glorot_uniform",
    )
)

# Add an output layer with 1 neuron and sigmoid activation
model.add(
    Dense(
        1,
        activation="sigmoid",
        name="output_layer"
      )
  )

# Display information about the model layers
for layer in model.layers:
    print(layer)

# Display initial weights and biases for each layer
for layer in model.layers:
    print(layer.get_weights())

# Define SGD and Adam optimizers
sgd = SGD(learning_rate=0.1)
adam = Adam()

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

# Display a summary of the model architecture
model.summary()

# Train the model on XOR data for 800 epochs
model.fit(X, y, epochs=800)

# Make predictions for specific input patterns
prediction_1 = model.predict(np.array([[0, 0]]))
prediction_2 = model.predict(np.array([[1, 0]]))

# Display predictions
print("Prediction for [0, 0]:", prediction_1)
print("Prediction for [1, 0]:", prediction_2)
