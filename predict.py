import pickle
import numpy as np

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Load saved model and files
# -----------------------------

# paths
MODEL_PATH = "model/best_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
MAX_LENGTH_PATH = "model/max_length.pkl"


# load trained model
model = load_model(MODEL_PATH, compile = False)


# load tokenizer
with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)


# load max length
with open(MAX_LENGTH_PATH, "rb") as file:
    max_length = pickle.load(file)


# feature extraction model (VGG16)
vgg_base = VGG16(
    weights="imagenet",
    include_top=True
)

vgg_model = Model(
    inputs=vgg_base.input,
    outputs=vgg_base.get_layer("fc2").output
)


# Convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None



# Extract image features

def extract_features(image_path):

    # Load and resize image according to VGG16 input size
    image = load_img(image_path, target_size=(224, 224))

    # Convert image into numerical array
    image = img_to_array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Apply VGG16 preprocessing
    image = preprocess_input(image)

    # Extract image features
    feature = vgg_model.predict(image, verbose=0)

    return feature


# Generate caption
def predict_caption(feature):

    # Starting word for caption generation
    caption = "start"

    # Generate words one by one until maximum length
    for _ in range(max_length):

        # Convert current caption into sequence of numbers
        sequence = tokenizer.texts_to_sequences([caption])[0]

        # Pad sequence to match training length
        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post"
        )

        # Predict next word
        prediction = model.predict(
            [feature, sequence],
            verbose=0
        )

        # Select word with highest probability
        predicted_index = np.argmax(prediction)

        # Convert index back to word
        word = idx_to_word(predicted_index, tokenizer)

        # Stop if no word is found
        if word is None:
            break

        # Add predicted word to caption
        caption += " " + word

        # Stop generation after end token
        if word == "end":
            break

    # Remove start and end tokens before displaying
    caption = caption.replace("start", "")
    caption = caption.replace("end", "")

    return caption.strip()