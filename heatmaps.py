import keras
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models
from PIL import Image
import numpy as np

# Load the model
model = load_model("new_skin_disease.keras")  # path to your .keras file
model.summary()  # optional: check model architecture

# Load and preprocess image
img_path = '../../skin_disease_dataset/Skin Cancer Dataset/Seborrheic Keratosis/ISIC_9811367.jpg'  # change this to your image path

pil_image = Image.open(img_path)
opencv_image = np.array(pil_image)
image = opencv_image
if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
elif image.shape[-1] == 1:
    image = np.repeat(image, 3, axis=-1)
image = cv2.resize(image, (224, 224))
original_img = image.copy()
image = np.expand_dims(image, axis=0)
print(f"\n\n\n the image shape is {image.shape} \n\n\n")  # should be (1, 224, 224, 3)
img_array = model.predict(image)
predicted_class = np.argmax(img_array[0])
# Grad-CAM heatmap
grad_model = models.Model([model.inputs], [model.get_layer(index=-4).output, model.output])
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(image)
    loss = predictions[:, predicted_class]

print(f"Shape of conv_outputs: {conv_outputs.shape}")

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()[0]

# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
img_size = (224, 224)  # size of the input image for the model
# Load original image for overlay

# Resize heatmap to match image
heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap
superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# Show the result
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f'Predicted Class: {predicted_class}')
plt.axis('off')
plt.show()
