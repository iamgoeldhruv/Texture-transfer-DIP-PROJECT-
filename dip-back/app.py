from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from skimage import io
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from texture_transfer import texture_transfer


app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'input_image' not in request.files or 'target_image' not in request.files:
        return jsonify({"error": "Please upload both input and target images."}), 400

    input_file = request.files['input_image']
    target_file = request.files['target_image']

    print(f"Input file: {input_file.filename}, Type: {input_file.content_type}")
    print(f"Target file: {target_file.filename}, Type: {target_file.content_type}")

    # try:
    #     # Read the images as NumPy arrays
    #     # input_image = io.imread(input_file) 
    #     # target_image = io.imread(target_file) 
        
    #     # Resize input image to match the dimensions of the target image
    #     # input_image = resize(input_image, (target_image.shape[0], target_image.shape[1]), anti_aliasing=True)
    # except Exception as e:
    #     return jsonify({"error": f"Error reading images: {str(e)}"}), 500

    try:
         result_image=texture_transfer(input_file, target_file)
    except Exception as e:
        return jsonify({"error": f"Texture transfer error: {str(e)}"}), 500

    fig, ax = plt.subplots()
    ax.imshow(result_image)
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
