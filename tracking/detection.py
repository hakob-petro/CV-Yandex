import os

import cv2
import numpy as np
import torch
from config import VOC_CLASSES, bbox_util, model
from PIL import Image
from skimage import io
from skimage.transform import resize
from utils import get_color


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)


def image2tensor(image):
    # Write code here
    image = image.astype(np.float32)  # convert frame to float
    image = cv2.resize(image, (300, 300))  # resize image to 300x300
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)  # convert RGB to BGR
    image = image - IMAGENET_MEAN  # center with respect to imagenet means
    image = image.transpose([2, 0, 1])  # torch works with CxHxW images
    tensor = torch.tensor(image.copy()).unsqueeze(0)
    tensor = tensor.to(torch.float32)
    # tensor.shape == (1, channels, height, width)
    return tensor


@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    # Write code here
    # First, convert the input image to tensor
    input_tensor = image2tensor(image=frame)

    # Then use model(input_tensor),
    # convert output to numpy
    # and bbox_util.detection_out
    results = model(input_tensor).detach().numpy()

    # Select detections with confidence > min_confidence
    # hint: see confidence_threshold argument of bbox_util.detection_out
    results = bbox_util.detection_out(results, confidence_threshold=min_confidence)

    # If label set is known, use it
    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indices = [
            index
            for index, label in enumerate(result_labels)
            if VOC_CLASSES[label - 1] in labels
        ]
        results = results[indices]

    # Remove confidence column from result
    results = results[0]  # because we have just one frame

    if len(results) > 0:
        results = np.delete(results, 1, axis=1)

        # Resize detection coords to the original image shape.
        results[:, 1::2] *= frame.shape[1]
        results[:, 2::2] *= frame.shape[0]
    # Return result
    return detection_cast(results)


def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """
    frame = frame.copy()

    # Write code here
    for label, xmin, ymin, xmax, ymax in detections:
        rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
        frame[rr, cc] = get_color(label)

    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, "data", "000004.jpg"))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == "__main__":
    main()
