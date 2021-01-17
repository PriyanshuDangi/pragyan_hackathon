import time
from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from trainee.models import Yolov
from trainee.dataset import transform_images, load_tfrecord_dataset
from trainee.utils import draw_outputs

resize_to=416
labels_count=80
flags.DEFINE_list('images', '/data/images/dog.jpg', 'list with paths to input images')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './detections/', 'path to output folder')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    yolo = Yolov(classes=labels_count)

    yolo.load_weights('./weights/yolov.tf').expect_partial()

    class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, resize_to)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        raw_images = []
        images = FLAGS.images
        for image in images:
            img_raw = tf.image.decode_image(
                open(image, 'rb').read(), channels=3)
            raw_images.append(img_raw)
    num = 0    
    for raw_img in raw_images:
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, resize_to)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()

        print('detections:')
        for i in range(nums[0]):
            print('\n{}'.format(class_names[int(classes[0][i])]))

        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(FLAGS.output + 'detection' + str(num) + '.jpg', img)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass