import depthai as dai
import numpy as np

class Camera:
    def __init__(self):
        self.pipeline = dai.Pipeline()

        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(640, 480)
        cam.setInterleaved(False)

        xout = self.pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)

        self.device = dai.Device(self.pipeline)
        self.q = self.device.getOutputQueue("rgb")

        self.K = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0,   0,   1]
        ])

    def read(self):
        frame = self.q.get().getCvFrame()
        depth = np.ones((frame.shape[0], frame.shape[1])) * 0.6
        return frame, depth, self.K