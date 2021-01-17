import time
from absl import app
import cv2
import numpy as np
import tensorflow as tf
from trainee.models import Yolov
from trainee.dataset import transform_images, load_tfrecord_dataset
from trainee.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template
import os
from flask_socketio import SocketIO, emit
import io
from io import StringIO, BytesIO
import base64
from PIL import Image

resize_to = 416                      
output_path = './detections/'   
labels_count = 80                

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = Yolov(classes=labels_count)
yolo.load_weights('./weights/yolov.tf').expect_partial()
class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('camera.html')

@app.route('/detectImages')
def images():
    return render_template('image.html')

@app.route('/index')
def camera():
    return render_template('index.html')

# @socketio.on('image')
# def image(data_image):
#     sbuf = StringIO()
#     sbuf.write(data_image)
#     print(data_image)
#     # decode and convert into image
#     b = io.BytesIO(base64.b64decode(data_image))
#     pimg = Image.open(b)

#     ## converting RGB to BGR, as opencv standards
#     frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

#     # Process the image frame
#     frame = imutils.resize(frame, width=700)
#     frame = cv2.flip(frame, 1)
#     imgencode = cv2.imencode('.jpg', frame)[1]

#     # base64 encode
#     stringData = base64.b64encode(imgencode).decode('utf-8')
#     b64_src = 'data:image/jpg;base64,'
#     stringData = b64_src + stringData

#     # emit the frame back
#     emit('response_back', stringData)

@socketio.on('image')
def image(data_image):
    print(data_image)
    # image = open('image.png', 'wb')
    # image.write(data_image)
    # image.close()
    data = data_image.replace('data:image/png;base64,', '')
    imgdata = base64.b64decode(data)  # I assume you have a way of picking unique filenames
    with open('filename.jpg', 'wb') as f:
        f.write(imgdata)
    # image = open('filename.jpg', 'rb')
    # print(image)
    image_name = 'filename.jpg'
    # image_name = test.jpg
    # image.save(os.path.join(os.getcwd(), image_name))
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    # print(img_raw)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, resize_to)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + 'detection.jpg', img)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()
    
    #remove temporary image
    # os.remove(image_name)

    emit('response_back', response)

@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)
        
    num = 0
    
    response = []

    for j in range(len(raw_images)):
        responses = []
        raw_img = raw_images[j]
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, resize_to)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404)

@app.route('/image', methods= ['POST'])
def get_image():
    image = request.files["images"]
    print(image)
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, resize_to)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + 'detection.jpg', img)
    print('output saved to: {}'.format(output_path + 'detection.jpg'))
    
    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()
    
    #remove temporary image
    os.remove(image_name)

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

# @app.route('/video', methods= ['POST'])
# def video():
#     video = request.files['video']
#     video_name = video.filename
#     video.save(os.path.join(os.getcwd(), video_name))
#     vid = cv2.video
#     return 'Winner!'

if __name__ == '__main__':
    socketio.run(app)