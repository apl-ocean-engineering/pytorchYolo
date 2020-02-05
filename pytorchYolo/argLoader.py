#!/usr/bin/env python3
from os.path import dirname, abspath
import argparse


class ArgLoader:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='YOLO v3 \
                                              Detection Module')

        self.parser.add_argument("--run_style",
                                 help="How to run the network. 0 for loading \
                                 images. 1 for streaming video. \
                                 2 for all others (ROS wrappers, etc.)",
                                 default=2, type=int)
        self.parser.add_argument("--use_batches",
                                 help="How to run the network. 0 for loading \
                                 images. 1 for streaming video. 2 for all \
                                 others (ROS wrappers, etc.)",
                                 default=0, type=int)
        self.parser.add_argument("--data_cfg",
                                 help="Path to the data cfg file",
                                 default="cfg/YOLO.data", type=str)
        self.parser.add_argument("--verbose",
                                 help="Display detections",
                                 default=False, type=bool)
        self.parser.add_argument("--images",
                                 help="Image / Directory containing images to \
                                 perform detection upon. \
                                 Only needed if reading IMAGES from disk",
                                 default="/data/test_images", type=str)
        self.parser.add_argument("--video",
                                 help="Video path containing video to \
                                 perform detection upon. \
                                 Only needed if reading VIDEO from disk",
                                 default="/data/video.avi", type=str)
        self.parser.add_argument("--output",
                                 help="Image / Directory to store \
                                 detections to",
                                 default="data/output", type=str)
        self.parser.add_argument("--batch_size",
                                 help="Batch size",
                                 default=1, type=int)
        self.parser.add_argument("--conf_thresh",
                                 help="Object Confidence to \
                                 filter predictions",
                                 default=0.75, type=float)
        self.parser.add_argument("--nms_thresh",
                                 help="IOU threshold for non-max suppression",
                                 default=0.4, type=float)
        self.parser.add_argument("--img_size",
                                 help="Input resolution of the network. \
                                 Increase to increase accuracy. "
                                 "Decrease to increase speed",
                                 default=416, type=int)
        self.parser.add_argument("--display_classification",
                                 help="Display classification type",
                                 default=False, type=bool)
        self.parser.add_argument("--save_predictions",
                                 help="Save network predictions",
                                 default=False, type=bool)
        self.parser.add_argument("--save_images",
                                 help="Save labled images",
                                 default=False, type=bool)
        self.parser.add_argument("--save_predictions_path",
                                 help="Path to save network predictions",
                                 default=dirname(dirname(abspath(__file__)))
                                 + "/data/detections/", type=str)
        self.parser.add_argument("--save_predictions_name",
                                 help="Sub-folder to save detections",
                                 default=" ", type=str)
        self.parser.add_argument("--use_dual_manta",
                                 help="To evalue both Manta 1 and Manta 2. \
                                 If true, point images to base folder",
                                 default=False, type=bool)

    @property
    def args(self):
        """
        Parse command line arguments passed to the detector module.

        """
        return self.parser.parse_args()
