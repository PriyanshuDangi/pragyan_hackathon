from absl import app
import numpy as np
from trainee.models import Yolov
from trainee.utils import weights_inject

labels_count = 80


def main(_argv):
    yolo = Yolov(classes=labels_count)

    yolo.summary()

    weights_inject(yolo, 'weights/yolov.weights')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)

    yolo.save_weights('weights/yolov.tf')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
