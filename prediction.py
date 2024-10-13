import numpy as np
from PIL import Image, ImageOps
import tkinter as tk
from tkinter.filedialog import askopenfilename
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model1.keras")

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)               # Invert colors (MNIST uses black digits on white background)
    img = img.resize((28, 28))               # Resize to 28x28 pixels
    imgPixel = np.array(img) / 255.0        # Normalize pixel values (0-1) as it is a grayscale image

    imgPixelReshaped = imgPixel.reshape(1, 28, 28, 1)  # Reshape the image for model input
    return imgPixelReshaped, imgPixel  # Return the processed array and 2D original pixel array

# Function to predict the digit
def predict_digit(img_path):
    img_array, processed_image = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)  # Get the digit with the highest probability
    return predicted_digit, processed_image

# Function to open a file dialog and predict the selected image
def prediction():
    filePath = askopenfilename(
        title="Select the Image to predict on",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if filePath:
        print(f"Selected file: {filePath}")
        
        # Preprocess the image and predict the digit
        predictedDigit, pixelImage = predict_digit(filePath)
        
        # Display the prediction in the GUI
        predictionLabel.config(text = f"Predicted Digit: {predictedDigit}")
        
        # Update the plot with the processed image using the canvas made earlier
        ax.clear()
        ax.imshow(pixelImage, cmap='gray')
        ax.set_title("Pixel Graph of the Image")
        ax.axis('on')
        selectImageButton.config(text="Upload New Image")
        canvas.draw()  # Update the canvas to show the new plot
    else:
        print("No file selected.")

# Create the GUI window
root = tk.Tk()
root.title("Prediction of Handwritten Digits")
root.geometry("600x600") #size of the window

# Create a button to open file dialog
selectImageButton = tk.Button(root, text="Upload Your Image to Predict", command=prediction) #prediction function set to run on click (even handle)
selectImageButton.pack(pady=20)

# Label to display the prediction
predictionLabel = tk.Label(root, text="Prediction : No Image Currently", font=('Helvetica', 20))
predictionLabel.pack(pady=20)

# Create a Matplotlib figure for the processed image
fig = Figure(figsize=(4, 4))
ax = fig.add_subplot(111)

# Create a canvas to embed the Matplotlib figure in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root) # it wraps the figure created in matplotlib into a widget for Tkinter
canvas_widget = canvas.get_tk_widget() #it extracts the widget into the GUI layout
canvas_widget.pack(pady=10) #vertical padding of 10 pixel


root.mainloop() #keeps the window running, even after finishing the script
