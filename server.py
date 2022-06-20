import multiprocessing
import os
import time
import uuid
from multiprocessing import Queue, Pool

import cv2
import numpy as np

from src.BTTR_Model.BTTR_Model import bttr_model_predict
from src.ABM_Model.ABM_Model import abm_model_predict
import base64
from io import BytesIO

from PIL.ImagePath import Path
from flask_cors import CORS

from PIL import Image
from flask import *
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app)


@app.route("/corrected-data", methods=['POST'])
def save_corrected_data():
    image = request.form['image']
    case_id = uuid.uuid4()

    image_path = os.path.join("./output/", "images/", f'{case_id}.png')
    code = BytesIO(base64.b64decode(image.split(',')[1]))
    image_decoded = Image.open(code)

    image_decoded.save(image_path)

    with open(f'./output/json/{case_id}.json', 'w') as f:
        f.write(json.dumps(json.loads(
           request.form['data']
        ), indent=4))
        f.close()

    response = jsonify({'success': True})
    response.headers.add('Access-Control-Allow-Origin', '*')
    print(response)
    return response


# @socketio.on('getLatexPrediction')
@app.route("/latex", methods=['POST'])
def get_latex_representation():
    # start latex evaluation
    # return latex math code as string
    base64_img = request.form['image']
    code = BytesIO(base64.b64decode(base64_img.split(',')[1]))
    image_decoded = Image.open(code)

    # image_decoded.save(Path(app.config['UPLOAD_FOLDER']) / 'image.png')
    result = predict_latex_representation(image_decoded)

    # except:
    #     result = None
    #     print("No image data available!")

    response = jsonify({'latex': result})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


def prepare_image(image):
    numpy_image = np.array(image)

    height, width, _ = numpy_image.shape
    gray_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2GRAY)
    (ret, thresh) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    thickness = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    img_sharped = cv2.dilate(thresh, kernel, iterations=1)

    # gaussian_blur_img = cv2.GaussianBlur(thresh, (3, 3), 1)
    # img_sharped = np.where(gaussian_blur_img > 0, 255, gaussian_blur_img)

    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width * 0.25), int(height * 0.25)))
    # dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
    #                                        cv2.CHAIN_APPROX_NONE)
    #
    # images = []
    # # DebugStart
    # count = 0
    # # DebugEnd
    #
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #
    #     # DebugStart
    #     rect = cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     # DebugEnd
    #
    #     cropped = thresh[y:y + h, x:x + w]
    #     max_size = max(int(h * 0.75), int(w * 0.75))
    #
    #     if w > max_size or h > max_size:
    #         ratio = calculate_ratio(w, h, max_size, max_size)
    #         cropped_resized = cv2.resize(cropped, dsize=(int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_BITS)
    #         cropped_resized = cv2.GaussianBlur(cropped_resized, (3, 3), 1)
    #         cropped_resized = np.where(cropped_resized > 0, 255, cropped_resized)
    #         images.append(cropped_resized)
    #     else:
    #         images.append(cropped)
    #
    #     # DebugStart
    #     count += 1
    #     # DebugEnd
    #
    # max_size = 0
    # max_size_img = []
    #
    # for img in images:
    #     w, h = img.shape
    #     size = w * h
    #     if size > max_size:
    #         max_size = size
    #         max_size_img = img
    #
    # return max_size_img
    return img_sharped


def calculate_ratio(src_width, src_height, max_width, max_height):
    return min(max_width / src_width, max_height / src_height)


def predict_latex_representation(image):
    configured_image = prepare_image(image)

    # ------------- multiprocessing process start ---------------
    # q = Queue()
    # jobs = []
    #
    # worker = lambda fn, args: q.put(fn(args))
    #
    # for fn in [bttr_model_predict, abm_model_predict]:
    #     process = multiprocessing.Process(
    #         target=worker,
    #         args=(fn, configured_image)
    #     )
    #     jobs.append(process)
    #
    # start_eval = time.time()
    # for j in jobs:
    #     j.start()
    #     print(q.get())
    #
    # for j in jobs:
    #     j.join()
    #
    # end_eval = time.time()
    # print(f"Predictions took {end_eval - start_eval} to process.")

    # ------------- multiprocessing process end ---------------

    # ------------- multiprocessing pool start ---------------
    # pool = Pool(8)
    # jobs = []
    #
    # def worker(args):
    #     print(args[0](args[1]))
    #
    # start_eval = time.time()
    # print(pool.map(worker, (
    #     [bttr_model_predict, configured_image],
    #     [abm_model_predict, configured_image]
    # )))
    #
    # end_eval = time.time()
    # print(f"Predictions took {end_eval - start_eval} to process.")

    # ------------- multiprocessing pool end ---------------

    start_eval = time.time()
    # bttr model prediction
    bttr_output = bttr_model_predict(configured_image)
    # abm model prediction
    abm_output = abm_model_predict(configured_image)

    end_eval = time.time()
    print(f"Predictions took {end_eval - start_eval} to process.")

    predictions = [bttr_output, abm_output]
    _, buffer = cv2.imencode('.png', configured_image)
    base64_img = base64.b64encode(buffer).decode("ascii")

    return {
        "bttr": predictions[0],
        "abm": predictions[1],
        "preprocessedImage": base64_img
    }
