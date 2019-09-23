# pytorchYolo
Inital work done by:

1. https://github.com/ayooshkathuria/pytorch-yolo-v3 (blog post: https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/).  
2. Darshan Mehta (https://github.com/darshanmehta17/3G-AMP) re-purposed for specific AMP work.

## Introduction

This repository, which builds off the above-mentioned previous work, is a general purpose YOLO network detection platform. A ROS wrapper to stream images from cameras for this platform [exists here](https://github.com/apl-ocean-engineering/ros_yolo).   
  
This repository will perform the forward pass of the YOLO network using a PyTorch framework. It does *not* contain a training platform, probably best to train on the [dark net platform](https://pjreddie.com/darknet/)  

## Installation and weights
It's easiest if both this package and the data/weights folder ([example](https://drive.google.com/drive/folders/1VOEoOOTOrzb-vwieegfKXBICpTeckB2F)) are installed to the same base direcotry (i.e. home) and your data/weights folder structure retains the same structure as the example data structure. All structures supported, but it will require more manual pointing to your weights and classes.  
### Clone and install the package  
1. Clone this repository to directory of your choice  
	- $ git clone https://github.com/apl-ocean-engineering/pytorchYolo.git  
2. Install the package  
	- $ cd <path_to_pytorchYolo>  
	- $ pip3 install .  

### Data and weights
Example data and weights can be [found here](https://drive.google.com/drive/folders/1VOEoOOTOrzb-vwieegfKXBICpTeckB2F). Download to same directory as pytorchYolo. The structure is:  
* YOLO_Example_data  
	- classesYOLO.txt (list of trained classes)  
	- imgs (directory with a few examples image)  
	- weights (directory containing example weights)  

It is recomended to put other images into this folder (or a similar folder), and new training weights into the weights folder.   

## Running
The main module 'detector.py' uses a constant arg parser, specified in argLoader.py. The full argument options can be shown with --help. The most pressing are:  
* --run_style. For the example script run_network.py, this specifies if you would like to run images (--run_style 0) or video(--run_style 1). --run_style 2 is default/for other options  
* --use_batches. For the example script run_network.py, wether to use image batches or stream individually  
* --data_cfg. Points to a .data file which specifies where the class names are, the YOLO cfg file is, and the trained YOLO weights. See YOLO.data for an example  
* --images. Points to a directory containing images to loop over. *Only necessary if --run_style 0 is specified*  
* --video. Points to a directory containing the location of a video to loop through. *Only neccessary if --run_style 1 is specified*  

'detector.py' contains several processing funcionalities, including:  
1. Loading images from a folder and running through YOLO in batches. Class: YoloImgRun (requires specifying --images)  
2. Individually streaming images through YOLO, either from a folder or another input wrapper (e.g., [ros_yolo](https://github.com/apl-ocean-engineering/ros_yolo)). Class:   YoloLiveVideoStream and  YoloImageStream(requires specifying --images)  
3. Loading a video and running individual images through. Class: YoloVideoRun (requires specifying --video)   

run_network.py offers a very simple run example for the above three mentioned options.

## Examples: 
Download the example YOLO set to the same base directory as this repo. That is:  
* <base_path>
	- pytorchYOLO  
	- YOLO_ExampleData  

Edit the YOLO.data script to point to the <base_path> (currently set to my home directory)

### Example 1, run example images from folder with batch processing
1. $ cd <path_to_pytorchYolo_folder>
2. $ ./run_network.py --data_cfg <path_to_YOLO.data> --run_style 0 --use_batches 1 --images <path_to_YOLO_example_data>/imgs

The network should run, print the run statics, and save the detection data to data/output

### Example 2, run example images from folder with no batch processing
1. $ cd <path_to_pytorchYolo_folder>
2. $ ./run_network.py --data_cfg <path_to_YOLO.data> --run_style 0 --use_batches 0 --images <path_to_YOLO_example_data>/imgs

The network should display images with their bounding boxes 

### Example 3, run video
1. $ cd <path_to_pytorchYolo_folder>
2. $ ./run_network.py --data_cfg <path_to_YOLO.data> --run_style 1 --use_batches 0 --video <path_to_VIDEO> 

The network should now be displaying the video with any detected objects showing
