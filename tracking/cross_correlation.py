import os

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template

from detection import detection_cast, draw_detections, extract_detections
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):
    """Return gaussian for tracking.

    shape: [width, height]
    x, y: gaussian center
    dx, dy: std by x and y axes

    return: numpy array (width x height) with gauss function, center (x, y) and std (dx, dy)
    """
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-((X - x) ** 2) / dx**2 - (Y - y) ** 2 / dy**2)


class CorrelationTracker(Tracker):
    """Generate detections and building tracklets."""

    def __init__(self, detection_rate=5, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate  # Detection rate
        self.prev_frame = None  # Previous frame (used in cross correlation algorithm)

    def build_tracklet(self, frame):
        """Between CNN execution uses normalized cross-correlation algorithm (match_template)."""
        detections = []
        # Write code here
        # Apply rgb2gray to frame and previous frame
        frame = rgb2gray(frame)
        prev_frame = rgb2gray(self.prev_frame)

        # For every previous detection
        # Use match_template + gaussian to extract detection on current frame
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            # Step 0: Extract prev_bbox from prev_frame
            prev_bbox = frame[ymin: ymax + 1, xmin: xmax + 1]

            # Step 1: Extract new_bbox from current frame with the same coordinates
            new_bbox = frame[ymin: ymax + 1, xmin: xmax + 1]

            # Step 2: Calc match_template between previous and new bbox
            # Use padding
            matching = match_template(new_bbox, prev_bbox, pad_input=True)

            # Step 3: Then multiply matching by gauss function
            # Find argmax(matching * gauss)
            gauss = gaussian(
                new_bbox.shape,
                new_bbox.shape[1] / 2,
                new_bbox.shape[0] / 2,
                new_bbox.shape[1] / 3,  # new_bbox.std(axis=1).mean(),
                new_bbox.shape[0] / 3,  # new_bbox.std(axis=0).mean()
            )
            result = np.multiply(matching, gauss)
            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]

            # Step 4: Append to detection list
            dx = x - new_bbox.shape[1] / 2
            dy = y - new_bbox.shape[0] / 2
            xmin = np.clip(xmin + dx, 0, frame.shape[1] - 1)
            ymin = np.clip(ymin + dy, 0, frame.shape[0] - 1)
            xmax = np.clip(xmax + dx, 0, frame.shape[1] - 1)
            ymin = np.clip(ymax + dy, 0, frame.shape[0] - 1)
            detections.append([label, xmin, ymin, xmax, ymax])

        return detection_cast(detections)

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
