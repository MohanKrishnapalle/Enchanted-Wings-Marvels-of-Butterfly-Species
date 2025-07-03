from flask import Flask, request, jsonify, render_template
import os
import logging
from keras.models import load_model
from keras. preprocessing. image import img_to_array, load_img
import numpy as np
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
try:
    model = load_model("vg16_model.h5")
    logging.info("Model loaded successfully")
except Exception:
    logging.error("Error loading model", exc_info=True)
    model = None
butterfly_names = {
     0: "ADONIS",
     1: "AFRICAN GIANT SWALLOWTAIL",
     2: "AMERICAN SNOOT",
     3: "AN 88",
     4: "APPOLLO",
     5: "ATALA",
     6: "BANDED ORANGE HELICONIAN",
     7: "BANDED PEACOCK",
     8: "BECKERS WHITE",
     9: "BLACK HAIRSTREAK",
    10: "BLUE MORPHO",
    11: "BLUE SPOTTED CROW",
    12: "BROWN SIPROETA",
    13: "CABBAGE WHITE",
    14: "CAIRNS BIRDWING",
    15: "CHECQUERED SKIPPER",
    16: "CHESTNUT",
    17: "CLEOPATRA",
    18: "CLODIUS PARNASSIAN",
    19: "CLOUDED SULPHUR",
    20: "COMMON BANDED AWL",
    21: "COMMON WOOD‑NYMPH",
    22: "COPPER TAIL",
    23: "CRECENT",
    24: "CRIMSON PATCH",
    25: "DANAID EGGFLY",
    26: "EASTERN COMA",
    27: "EASTERN DAPPLE WHITE",
    28: "EASTERN PINE ELFIN",
    29: "ELBOWED PIERROT",
    30: "GOLD BANDED",
    31: "GREAT EGGFLY",
    32: "GREAT JAY",
    33: "GREEN CELLED CATTLEHEART",
    34: "GREY HAIRSTREAK",
    35: "INDRA SWALLOW",
    36: "IPHICLUS SISTER",
    37: "JULIA",
    38: "LARGE MARBLE",
    39: "MALACHITE",
    40: "MANGROVE SKIPPER",
    41: "MESTRA",
    42: "METALMARK",
    43: "MILBERTS TORTOISESHELL",
    44: "MONARCH",
    45: "MOURNING CLOAK",
    46: "ORANGE OAKLEAF",
    47: "ORANGE TIP",
    48: "ORCHARD SWALLOW",
    49: "PAINTED LADY",
    50: "PAPER KITE",
    51: "PEACOCK",
    52: "PINE WHITE",
    53: "PIPEVINE SWALLOW",
    54: "POPINJAY",
    55: "PURPLE HAIRSTREAK",
    56: "PURPLISH COPPER",
    57: "QUESTION MARK",
    58: "RED ADMIRAL",
    59: "RED CRACKER",
    60: "RED POSTMAN",
    61: "RED SPOTTED PURPLE",
    62: "SCARCE SWALLOW",
    63: "SILVER SPOT SKIPPER",
    64: "SLEEPY ORANGE",
    65: "SOOTYWING",
    66: "SOUTHERN DOGFACE",
    67: "STRAITED QUEEN",
    68: "TROPICAL LEAFWING",
    69: "TWO BARRED FLASHER",
    70: "ULYSES",
    71: "VICEROY",
    72: "WOOD SATYR",
    73: "YELLOW SWALLOW TAIL",
    74: "ZEBRA LONG WING",
}
target_img = os.path.join(os.getcwd(), 'static/images')
@app.route('/')
def main_index():
    return render_template('index.html')
@app.route('/input', methods=['GET', 'POST'])

def input_page():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('input.html', prediction="No file part", image_url=None)

            file = request.files['file']
            if file.filename == '':
                return render_template('input.html', prediction="No selected file", image_url=None)

            if file:
                # Save the file
                file_path = os.path.join(target_img, file.filename)
                file.save(file_path)

                # Preprocess image
                image = load_img(file_path, target_size=(224, 224))
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = image / 255.0

                # Predict
                predictions = model.predict(image)
                predicted_class = np.argmax(predictions, axis=1)[0]
                butterfly_name = butterfly_names[predicted_class]

                image_url = os.path.join('static', 'images', file.filename)
                return render_template('input.html', prediction=butterfly_name, image_url=image_url)

        except Exception as e:
            logging.error("Error during prediction", exc_info=True)
            return jsonify({'error': str(e)}), 500

    return render_template('input.html')

if __name__ == '__main__':
    if not os.path.exists(target_img):
        os.makedirs(target_img)
    app.run(debug=True)
