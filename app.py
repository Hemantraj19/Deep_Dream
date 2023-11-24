from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Load your deepdream model
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]
deepdream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


def calc_loss(image,model):
  img_batch = tf.expand_dims(image, axis=0) # Convert into batch format
  layer_activations = model(img_batch) # Run the model

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act) #calculate mean of each activations
    losses.append(loss)

  return tf.reduce_sum(losses) #calculate sum


def deepdream(model, image, step_size):
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert NumPy array to TensorFlow tensor

    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = calc_loss(image, model)

    gradients = tape.gradient(loss, image)

    gradients /= tf.math.reduce_std(gradients)
    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)

    return loss, image



def deprocess(image):
    image = 255 * (image + 1.0) / 2.0
    image = tf.cast(image, tf.uint8)
    return image.numpy()  # Convert EagerTensor to NumPy array


def run_deep_dream_simple(model,image,steps,step_size=0.01):
  image = tf.keras.applications.inception_v3.preprocess_input(image)

  for step in range(steps):
    loss, image = deepdream(model,image,step_size)

    print('Step {}, loss {}'.format(step,loss))

  return deprocess(image)

# Function to perform deep dream on the image
def run_deep_dream(image,steps):
    # Convert RGBA image to RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image_array = np.array(image)
    # you can run the algorithm at various sizes of the image
    OCTAVE_SCALE = 1.3

    base_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

    for n in range(5):
        new_shape = tf.cast(base_shape*(OCTAVE_SCALE**n), tf.int32)
        image = tf.image.resize(image, new_shape).numpy()

    image = run_deep_dream_simple(model=deepdream_model, image=image_array, steps=steps, step_size=0.001)

    return image

# Function to convert image to base64 format
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    steps = request.form.get('steps')
    steps = int(steps)
    if 'input_image' not in request.files:
        return redirect(request.url)

    file = request.files['input_image']

    if file.filename == '':
        return redirect(request.url)

    if file:
        image = Image.open(file)
        deep_dream_image = run_deep_dream(image,steps)
        base64_image = image_to_base64(Image.fromarray(deep_dream_image))

        return render_template('result.html', image_base64=base64_image)

if __name__ == '__main__':
    app.run(debug=True)