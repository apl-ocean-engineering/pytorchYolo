#!/usr/bin/env python3

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""

from pytorchYolo.detector import YoloLiveVideoStream
from pytorchYolo.argLoader import ArgLoader
import cv2
import datetime
import glob

from stereoProcessing.intrinsic_extrinsic import Loader, ExtrinsicIntrnsicLoaderSaver
import sys
from shapely.geometry import Polygon
import copy
import time
import signal

class CvSquare:
    """
    Simple square object which will 'move', for AMP rectified image stereo
    fish detection
    """

    def __init__(self, img_size):
        """
        Input:
            img_size(tuple (x,y)): Size of image to verify in bounds status
        """
        self.lower_x = 0
        self.lower_y = 0
        self.upper_x = 0
        self.upper_y = 0

        self.img_size = img_size

    def set_square(self, lower_x, lower_y, upper_x, upper_y):
        """
        Define square values
        """
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.upper_y = upper_y

    def set_from_YOLO_square(self, sq):
        """
        Set from YOLO detection square
        """
        self.lower_x = sq.lower_x
        self.lower_y = sq.lower_y
        self.upper_x = sq.upper_x
        self.upper_y = sq.upper_y

    def move_square(self, motion=1):
        """
        Move square in forward horizontal direction by motion (default 1)
        """
        if motion > 0:
            if (self.upper_x < self.img_size[0]):
                self.lower_x += motion
                self.upper_x += motion
                return True
        elif motion < 0:
            if (self.upper_x > 0):
                self.lower_x += motion
                self.upper_x += motion
                return True

        return False

    def extend_square(self, x_extend, y_extend):
        """
        Increase a square by (x,y) in both directions
        """
        self.lower_x = max(0, self.lower_x - x_extend)
        self.lower_y = max(0, self.lower_y - y_extend)
        self.upper_x = min(self.img_size[0], self.upper_x + x_extend)
        self.upper_y = min(self.img_size[1], self.upper_y + y_extend)


class StereoProcessing:
    OVERLAP_THRESHOLD = 0.2

    MIN_DET_OBJS = 6
    MIN_SHARED_OBJS = 3

    SQ_X_EXTEND = 0
    SQ_Y_EXTEND = 50

    def __init__(self, args, detector):
        self.detector = detector

        self.loader = Loader(args.base_path)
        self.loader.load_params_from_file(args.calibration_yaml)
        self.EI_loader = ExtrinsicIntrnsicLoaderSaver(self.loader)
        # Load rectification maps

        im_size = (
            int(self.EI_loader.paramaters.im_size[0]),
            int(self.EI_loader.paramaters.im_size[1]))
        self.map1_1, self.map1_2 = cv2.initUndistortRectifyMap(
                self.EI_loader.paramaters.K1, self.EI_loader.paramaters.d1,
                self.EI_loader.paramaters.R1,
                self.EI_loader.paramaters.P1[0:3, 0:3], im_size,
                cv2.CV_32FC1)

        self.map2_1, self.map2_2 = cv2.initUndistortRectifyMap(
                self.EI_loader.paramaters.K2, self.EI_loader.paramaters.d2,
                self.EI_loader.paramaters.R2,
                self.EI_loader.paramaters.P2[0:3, 0:3], im_size,
                cv2.CV_32FC1)


    def sigint_handler(self, signum, frame):
        """
        Exit system if SIGINT
        """
        sys.exit()

    def parse_waitKey(self, k):
        if k == 113:  # q
            sys.exit()

    def draw_images(self, img1, img2, name1="frame1", name2="frame2", wait=0):
        cv2.imshow(name1, img1)
        cv2.imshow(name2, img2)

        k = cv2.waitKey(wait)
        self.parse_waitKey(k)

    def test_square_overlap(self, sq2, square1_list, img1=None, img2=None):
        # draw_circles(img1, sq2, color=(0,0,255))
        for sq1 in square1_list:
            # draw_circles(img1, sq1)
            # draw_images(img1, img2, wait=1)
            p1 = self.construct_polygon(sq1)
            p2 = self.construct_polygon(sq2)

            if p1.intersects(p2):

                max_area = max(p1.area, p2.area)
                overlap_over_area = p1.intersection(p2).area / max_area
                # print(overlap_over_area)
                if overlap_over_area > self.OVERLAP_THRESHOLD:
                    return True, sq1

        return False, None

    def deconstruct_square(self, sq):
        lower_left = (sq.lower_x, sq.lower_y)
        upper_left = (sq.lower_x, sq.upper_y)
        upper_right = (sq.upper_x, sq.upper_y)
        lower_right = (sq.upper_x, sq.lower_y)
        corners = [lower_left, upper_left, upper_right, lower_right]

        return corners

    def construct_polygon(self, square):
        corners = self.deconstruct_square(square)

        return Polygon([corners[0], corners[1], corners[2], corners[3]])

    def draw_circles(self, img, sq, radius=10, color=(255, 0, 0)):
        circles = self.deconstruct_square(sq)
        for circle in circles:
            cv2.circle(img, circle, radius, color, thickness=-1)

    def build_squares(self, im_size, yolo_squares):
        squares = []
        for ysq in yolo_squares:
            if ysq is not None:
                sq = CvSquare(im_size)
                sq.set_from_YOLO_square(ysq)
                sq.extend_square(self.SQ_X_EXTEND, self.SQ_Y_EXTEND)
                squares.append(sq)

        return squares

    def get_detection_squares(self, img):
        time_init = time.time()
        detection, yolo_squares = self.detector.stream_img(
            img, wait_key=1)


        squares = [] #self.build_squares(img.shape[0:2], yolo_squares)

        return detection, squares

    def check_stereo_detection(self, squares1, squares2, img1=None, img2=None,
                               show_YOLO=False, args=None):
        for sq2 in squares2:
            time_init = time.time()
            sq2_init = copy.deepcopy(sq2)
            # draw_circles(img2, sq2)
            # Search forward
            valid = sq2.move_square(motion=1)
            while valid:
                overlap, sq1 = self.test_square_overlap(
                    sq2, squares1,
                    img1=img1, img2=img2)
                if overlap:
                    if show_YOLO:
                        if args is not None:
                            self.draw_circles(img1, sq1)
                            self.draw_circles(img2, sq2_init)
                            self.draw_images(img1, img2, wait=0)
                        else:
                            print("NO ARGS PASSED")
                    return True
                valid = sq2.move_square()

            # Search forward
            valid = sq2.move_square(motion=-1)
            while valid:
                overlap, sq1 = self.test_square_overlap(
                    sq2, squares1,
                    img1=img1, img2=img2)
                if overlap:
                    if show_YOLO:
                        if args is not None:
                            self.draw_circles(img1, sq1)
                            self.draw_circles(img2, sq2_init)
                            self.draw_images(img1, img2, wait=0)
                        else:
                            print("NO ARGS PASSED")
                    return True
                valid = sq2.move_square()

            print("time elapsed", time.time() - time_init)

        return False

    def load_undistort_rectify_image(self, img, K1, d1, map1, map2):
        # raw_img = copy.deepcopy(img)

        # undistort images
        # img = cv2.undistort(img, K1, d1)

        # rectify images
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        return img, None

    def detection(self, detection1, detection2, squares1, squares2, img1, img2):

        if len(squares1) > self.MIN_DET_OBJS or len(squares2) > self.MIN_DET_OBJS:
            return True

        if detection1 and detection2:
            if len(squares1) > self.MIN_SHARED_OBJS and len(squares2) > self.MIN_SHARED_OBJS:
                return True

            stereo_detection = self.check_stereo_detection(
                squares1, squares2, img1=img1, img2=img2,
                show_YOLO=False)

            if stereo_detection:
                return True

        return False



    def run_images(self, img1, img2 = None):
        if img2 is None:
            return False

        img1, raw_img1 = self.load_undistort_rectify_image(
            img1, self.EI_loader.paramaters.K1,
            self.EI_loader.paramaters.d1,
            self.map1_1, self.map1_2)
        #
        #
        img2, raw_img2 = self.load_undistort_rectify_image(
            img2, self.EI_loader.paramaters.K2,
            self.EI_loader.paramaters.d2,
            self.map2_1, self.map2_2)

        if img1 is None or img2 is None:
            return False

        detection1, squares1 = self.get_detection_squares(
                img1)

        detection2, squares2 = self.get_detection_squares(
                img2)

        obj_found = self.detection(
            detection1, detection2, squares1, squares2,
            img1, img2)


        return obj_found
