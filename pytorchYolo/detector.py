#!/usr/bin/env python3
from time import time, sleep
import torch
from torch.autograd import Variable
import cv2
from pytorchYolo import utils
from pytorchYolo import constants 
import os
from pytorchYolo import darknet
import pickle as pkl
import pandas as pd
import random
import numpy as np

import glob

class Detector():
    """
    Base YOLO 'detector' class to process image and video streams
    Attributes:        
        gpu: Wether to use CUDA acceleration or not
        classes: List of classes
        num_classes: Number of classes
        
    Methods:
        -write: Draw bounding boxes on images
        -inp_dim (property): verifies the img_size is proper
        -get_mode: Return the model     
    """
    def __init__(self, args):
        #Load and parse args
        self._images = args.images
        self._videofile = args.video
        self._batch_size = args.batch_size
        self._conf_thresh = args.conf_thresh
        self._nms_thresh = args.nms_thresh
        self._data_cfg_file = args.data_cfg
        self._output_dir = args.output
        self._img_size = args.img_size
        
        self.save_predictions = args.save_predictions
        print(1, args.save_predictions)
        self._parse_data_init()
                
        #Use GPU, if possible
        self.gpu = torch.cuda.is_available()
        #self.gpu = False
        
        self._create_model()
        
        self._imlist_len = 0
        
        #Initalize time stats
        self._start_time_det_loop = time()
        self._start_time_read_dir = time()
        self._end_time_read_dir = time()
        self._start_time_load_batch = time()
        self._end_time_load_batch = time()
        self._end_time_det_loop = time()
        self._start_time_draw_box = time()
        self._end_time_draw_box = time()
        
        self.save_detection_path = "/home/mitchell/YOLO_data/full_trial_analysis/true_negaitves/detections/"

    
    def _parse_data_init(self):
        data_config = utils.parse_data_cfg(self._data_cfg_file) #parse the data config file
        base_path = data_config[constants.BASE_PATH].rstrip()
        if data_config[constants.CONF_NAMES][0] == '/':
            names_path = data_config[constants.CONF_NAMES].rstrip()
        else:
            names_path = base_path + "/" +  data_config[
                    constants.CONF_NAMES].rstrip()
        if data_config[constants.CFG][0] == '/':
            cfg_path = data_config[constants.CFG].rstrip()
        else:
            cfg_path = base_path + "/" + data_config[
                    constants.CFG].rstrip()
        if data_config[constants.CONF_WEIGHTS][0] == '/':
            weights_path = data_config[constants.CONF_WEIGHTS].rstrip()
        else:
            weights_path = base_path + "/" + data_config[
                    constants.CONF_WEIGHTS].rstrip()       
            
            
        self.classes = utils.load_classes(names_path) #load the list of classes
        self.num_classes = len(self.classes)
        self._cfg_file = cfg_path.rstrip()
        self._weights_file = weights_path.rstrip()
            
    def write(self, x, results):
        """
        Draws the bounding box in x over the image in results with a random color.
        
        Input:
            x: Bounding box to be drawn.
            results: Images over which the bounding box needs to be drawn
        
        Return: 
            img: The final image containing the bounding boxes drawn.
        """
        colors = self._get_colors()
        class_color_pair = [(class_name, random.choice(colors)) 
            for class_name in self.classes]  # assign a color to each class
        c1 = tuple(x[1:3].int())  # top-left coordinates
        c2 = tuple(x[3:5].int())  # bottom-right coordinates
        
        img = results[int(x[0])]  # get the image corresponding to the bounding box
        cls = int(x[-1])  # get the class index
        label = "{0}".format(class_color_pair[cls][0])
        color = class_color_pair[cls][1]
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), 
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img
    
    @property
    def inp_dim(self):
        """
        Check the value of the imag_size and verify it's divisable by 32
        
        Return:
            Proper img_size
        """
        inp_dim = int(self._img_size)
        if not inp_dim % 32 == 0:
            inp_dim = int(inp_dim / 32.0)*32
        if inp_dim < 32:
            inp_dim = 32
        
        return int(inp_dim)   

    
    def get_model(self):
        """
        Get the YOLO model
        """
        return self.model     
        
    def _create_model(self):
        # Initialize the network and load the weights from the file
        print('Loading the network...')
        model = darknet.Darknet(self._cfg_file)
        model.load_weights(self._weights_file)
        model.set_input_dimension(self._img_size)
    
        if self.gpu:
            model.cuda()
    
        print('Network loaded successfully.')
    
        # Set the model in evaluation mode
        model.eval()
        
        self.model = model
        
    def _get_colors(self):
        dir_path = str(os.path.dirname(os.path.realpath(__file__)))
        return pkl.load(open(dir_path+ "/pallete.pickle", "rb"))
        
        
    


class YoloImgRun(Detector):
    """
    Child class of Detector. Class to help loop over all images within a 
    specified image folder
    """
    
    def load_images(self):
        """
        Load the images from the image directory
        
        Input:
            None
            
        Return:
            imlist: List of image names
            im_dim_list: List of image dimensions
            loaded_images: List of Cv images
            im_batches: List of prepared cv images for YOLO
        """
        self._start_time_read_dir = time()
        try:
            imlist = [os.path.join(os.path.realpath('.'), self._images, image) 
                                for image in os.listdir(self._images)]
        except NotADirectoryError:
            imlist = []
            imlist.append(os.path.join(os.path.realpath('.'), self._images))
        except FileNotFoundError:
            print("No file or directory found with the name {}"
                                              .format(self._images))
            exit()
        self._end_time_read_dir = time()
    
        # Create the directory to save the output if it doesn't already exist
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
    
        # Load the images from the paths
        self._start_time_load_batch = time()
        loaded_images = [cv2.imread(impath) for impath in imlist if not cv2.imread(impath) is None]
        # Move images to PyTorch variables
        im_batches = list(map(
                utils.prepare_image, loaded_images, 
                [(self._img_size, self._img_size) for x in range(len(imlist))]))
    
        # Save the list of dimensions or original images for remapping the bounding boxes later
        im_dim_list = [(image.shape[1], image.shape[0]) 
                for image in loaded_images]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    
        if self.gpu:
            im_dim_list = im_dim_list.cuda()  
            # Create the batches
        leftover = 0
        if len(im_dim_list) % self._batch_size:
            leftover = 1
    
        if self._batch_size != 1:
            num_batches = len(imlist) // self._batch_size + leftover
            im_batches = [torch.cat((im_batches[i * 
                    self._batch_size: min((i + 1) * 
                    self._batch_size, len(im_batches))])) 
                    for i in range(num_batches)]
        self._end_time_load_batch = time()
            
        self._imlist_len = len(imlist)
        
        return imlist, im_dim_list, loaded_images, im_batches
    
    
    def run(self):     
        """
        Main function. Load and loop over all images. 
        Run back end YOLO network in batches. Display and save
        """
        #Load the images and image batches
        imlist, im_dim_list, loaded_images, im_batches = self.load_images()
        
        # Perform the detections
        write = 0
        self._start_time_det_loop = time()
        for idx, batch in enumerate(im_batches):
            start_time_batch = time()
            if self.gpu:
                batch = batch.cuda()  # move the batch to the GPU
            with torch.no_grad():
                prediction = self.model.forward(Variable(batch), self.gpu)
            prediction, all_ious, full_class_scores = utils.filter_transform_predictions(prediction, 
                        self.num_classes, self._conf_thresh, self._nms_thresh)
            end_time_batch = time()
    
            if type(prediction) == int:
                for im_num, image in enumerate(imlist[idx * 
                        self._batch_size: min((idx + 1) * 
                        self._batch_size, len(imlist))]):
                    im_id = idx * self._batch_size + im_num
                    print("{0:20s} predicted in {1:6.3f} seconds"
                          .format(image.split("/")[-1], 
                        (end_time_batch - start_time_batch) / self._batch_size))
                    print("{0:20s} {1:s}".format("Objects Detected:", ""))
                    print("----------------------------------------------------------")
                continue
    
            # transform the image reference index from batch level to imlist level
            prediction[:, 0] += idx * self._batch_size  
    
            if not write:
                output = prediction
                
                write = 1
            else:
                output = torch.cat((output, prediction))
                
    
    
            for im_num, image in enumerate(imlist[idx * self._batch_size: min((idx + 1) * self._batch_size, len(imlist))]):
                im_id = idx * self._batch_size + im_num
                objs = [self.classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                
                print("{0:20s} predicted in {1:6.3f} seconds".
                      format(image.split("/")[-1], 
                      (end_time_batch - start_time_batch) / self._batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
                print("----------------------------------------------------------")
    
            if self.gpu:
                torch.cuda.synchronize()
    
        try:
            output
        except NameError:
            exit()
            
            
            

        # Translate the dimensions of the bounding box from the input size of 
        #the network to the original size of the image
        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
        scaling_factor = torch.min(self._img_size / im_dim_list, 1)[0].view(-1, 1)
    
        output[:, [1, 3]] -= (self._img_size - scaling_factor * 
                                  im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self._img_size - scaling_factor * 
                                  im_dim_list[:, 1].view(-1, 1)) / 2
    
        # Also undo the scaling introduced by the image_to_letterbox function
        output[:, 1:5] /= scaling_factor
    
        # Clip the bounding boxes which have boundaries outside the image
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, 
                                      im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, 
                                      im_dim_list[i, 1])
            
            
        self._end_time_det_loop = time()
    
        # Draw the boxes and save the images
        self._start_time_draw_box = time()
        
        # draw the boxes on the images
        list(map(lambda x: self.write(x, loaded_images), output))  
        
        # Generate output file names
        detection_names = pd.Series(imlist).apply(lambda x: 
            "{}/{}".format(self._output_dir, x.split("/")[-1]))
            
         # write the files
        list(map(cv2.imwrite, detection_names, loaded_images)) 
        self._end_time_draw_box = time()
        
        self._print_end_stats()
        
        
        torch.cuda.empty_cache()
        
    def _print_end_stats(self):
        print("SUMMARY")
        print("----------------------------------------------------------")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print()
        print("{:25s}: {:2.3f}".format("Loading batch", self._end_time_load_batch - self._start_time_load_batch))
        print("{:25s}: {:2.3f}".format("Detection (" + str(self._imlist_len) + " images)",
                                       self._end_time_det_loop - self._start_time_det_loop))
        print("{:25s}: {:2.3f}".format("Drawing Boxes", self._end_time_draw_box - self._start_time_draw_box))
        print("{:25s}: {:2.3f}".format("Average time_per_img", (self._end_time_draw_box - self._start_time_load_batch) / self._imlist_len))
        print("----------------------------------------------------------")        
        
class YoloLiveVideoStream(Detector):   
    """
    Child class of Detector. Class to run backend YOLO network from input 
    stream of images
    """     
    def stream_img(self, img, fname = ' '):
        """
        Main function. Accepts a cv_image and runs the back end YOLO network
        
        Inputs:
            img (np array): Input opencv image
            
        Returns:
            None
        """
        mean_iou = -1
        orig_im = img
        img_shape = img.shape
        start_img_time = time()
        img = utils.prepare_image(img, (self._img_size,self._img_size))
        im_dim = torch.FloatTensor((orig_im.shape[1], orig_im.shape[0])).repeat(1,2)
        
        if self.gpu:
            im_dim = im_dim.cuda()
            img = img.cuda()
            
        with torch.no_grad():   
            prediction = self.model.forward(Variable(img), self.gpu)
        prediction, all_ious, full_class_scores = utils.filter_transform_predictions(prediction, self.num_classes, self._conf_thresh, self._nms_thresh)
        
        #sm = torch.nn.Softmax()
        #probs = sm(prediction)
        #print(probs)
        
        end_img_time = time()
        if type(prediction) == int:
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            phrase = "img predicted in %f seconds" % (end_img_time - start_img_time)
            print(phrase)
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
            print(self.save_predictions)
            if self.save_predictions:
                f = open("//home/mitchell/YOLO_data/data/AMP_test_detectionLabels/" + fname.replace('.jpg', '.txt'), 'w+')
                f.close()
            return False, mean_iou
        
        total_iou = 0
        total_count = 0
        #print(all_ious)
        for iou in all_ious:
            #print(iou.item())
            total_iou += iou
            total_count += 1
        if total_count > 0:
            mean_iou = total_iou/total_count
                    
                
        
        self.output = prediction
        im_dim = im_dim.repeat(self.output.size(0), 1)
        scaling_factor = torch.min(self._img_size/im_dim,1)[0].view(-1,1)
        
        self.output[:,[1,3]] -= (self._img_size - scaling_factor*im_dim[:,0].view(-1,1))/2
        self.output[:,[2,4]] -= (self._img_size - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        self.output[:,1:5] /= scaling_factor
    

        list(map(lambda x: self.write(x, orig_im), self.output))
        pose_list = []
        print(self.save_predictions)
        if self.save_predictions:
            f = open(self.save_detection_path + fname.replace('.jpg', '.txt'), 'w+')
            for x in self.output:
                c1 = tuple(x[1:3])  # top-left coordinates
                c2 = tuple(x[3:5])  # bottom-right coordinates
            
                
                width_x = abs(c1[0].item() - c2[0].item())
                width_y = abs(c1[1].item() - c2[1].item())
                
                center_x = c1[0].item() + width_x/2
                center_y = c1[1].item() + width_y/2
                pose_list.append([center_x/img_shape[1], center_y/img_shape[0], width_x/img_shape[1], width_y/img_shape[0]])
                #output_write = str(0) + ' ' +  str(center_x/img_shape[0]) + ' ' + str(center_y/img_shape[1]) + ' ' +  str(width_x/img_shape[0]) + ' ' + str(width_y/img_shape[1]) 
                #f.write(output_write + '\n')
            #f.close()
            sorted_output = sorted(pose_list)
            for i, write_list in enumerate(sorted_output):
                if i < len(full_class_scores):
                    output_write = str(0) + ' ' + str(1.0) + ' ' + str(write_list[0]) + ' ' + str(write_list[1]) + ' ' +  str(write_list[2]) + ' ' + str(write_list[3]) 
                else:
                    output_write = str(0) + ' ' + str(1.0) + ' ' + str(write_list[0]) + ' ' + str(write_list[1]) + ' ' +  str(write_list[2]) + ' ' + str(write_list[3]) 
                f.write(output_write + '\n')
            #print(output_write)
            f.close()        
        cv2.imshow("frame", orig_im)
        cv2.waitKey(1)
        
        
        objs = [self.classes[int(x[-1])] for x in prediction]
        phrase = "img predicted in %f seconds" % (end_img_time - start_img_time)
        print(phrase)
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return True, mean_iou
        
        return True, mean_iou
        
        
        
    def write(self, x, img):
        """
        Draws the bounding box in x over a single image with a random color.
        Overwritten from inherited Detector.write() function
        
        Input:
            x: Bounding box to be drawn.
            img: Img over which the bounding box(es) need to be drawn
        
        Return: 
            img: The final image containing the bounding boxes drawn.
        """
        #print(type(img))
        c1 = tuple(x[1:3].int())  # top-left coordinates
        c2 = tuple(x[3:5].int())  # bottom-right coordinates
        
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])

        colors = self._get_colors()
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img
        
class YoloVideoRun(YoloLiveVideoStream):
    """
    Child class of YoloLiveVideoStream. Class to run video stream on YOLO network
    """ 
    def run(self):
        """
        Main function. Load the video file, sends images to YOLO network
        """
        self.model
        cap = cv2.VideoCapture(self._videofile)
        
        assert cap.isOpened(), 'Cannot capture source'
        
        frames = 0
        start = time()    
        while cap.isOpened():   
            ret, frame = cap.read()
            if ret:    
                print("FPS of the video is {:5.2f}".format( frames / (time() - start)))
         
            else:
                break
            
class YoloImageStream(YoloLiveVideoStream):
    """
    Child class of YoloLiveVideoStream. Class to stream a series of images one
    at a time to the YOLO network
    """ 
    def run(self, pause = 0.1):
        """
        Main function. Find all images in a folder, send to YOLO network individually
        """
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        full_images = sorted(glob.glob(self._images + "/*"))
        count = 0
        total_iou = 0
        for frame in full_images: 
            img = cv2.imread(frame)
            print(frame)
            if img is None:
                continue
            
            detection, mean_iou = self.stream_img(img, frame.split('/')[-1])  
            if mean_iou != -1:
                total_iou+=mean_iou
                count += 1
                #print(mean_iou)
                
                
            
            sleep(pause)
        print("Mean iou:", total_iou/count)
    

    
    

    