# Handwritten Digit Recognition using ANN

This project builds a neural network to recognize handwritten numbers, trained on the MNIST dataset. The dataset contains images of handwritten digits ranging from 0 to 9. The model is primarily constructed using TensorFlow and its Keras API.

Each image in the dataset is a grayscale image with a resolution of 28x28 pixels. The neural network consists of two layers: a **dense (hidden) layer** and an **output layer**.

- **First Layer (Dense Layer):**  
  This layer contains 128 neurons, with each neuron receiving all 784 input values (28x28 pixels flattened into a single vector). Each neuron computes its output based on initialized weights and biases.

- **Output Layer:**  
  This layer contains 10 neurons, each corresponding to one digit (0-9). The output is a probability distribution, predicting the most likely digit.

## Output

The model achieves high performance with:  
- **Training Accuracy:** 98%  
- **Testing Accuracy:** 97%  

The loss during training was reduced to **4%**, and during testing to **7%**.

## prediction.py

A user interface is provided where users can upload their own handwritten digit images. The trained model (`model1.keras`) will predict the digit based on the uploaded image.

A sample image (`imageTest5.jpg`) is provided as an example to test the prediction on unseen images

https://github.com/user-attachments/assets/535a7381-cddd-49a8-b60a-688be536a982

## Future Scope

There is potential to further enhance the model's performance on unseen data (which can be evaluated using the `prediction.py` script). This could be achieved by:
1. **Using a different optimizer**, which controls the learning rate.
2. **Increasing the number of neurons** in the dense layer to capture more complex patterns.

In the future, a **Convolutional Neural Network (CNN)** could be used for more accurate image classification, as CNNs are better suited for image data. However, this project provides a solid foundation for building more advanced models.
