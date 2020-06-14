from __future__ import division

import cv2
import numpy as np
import torch
import torch.nn as nn

from pytorchYolo.constants import *
from pytorchYolo.layers import EmptyLayer, DetectionLayer


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    Input:
    ------
    cfgfile (str): Path to the configuration file.

    Output:
    -------
    blocks (list): List of blocks in the neural network.

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:

        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block[LPROP_TYPE] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_network(blocks):
    """
    Takes a list of blocks as returned by the parse_cfg function and
    creates a network using it.

    :param blocks: list of blocks of the neural network
    :return net_info: dictionary consisting of network and training parameters
    :return module_list: nn.ModuleList object consisting of the layers of the network
    """
    net_info = blocks[0]  # Capture the block containing network parameters

    # check if the net_info captured is correct
    assert net_info[LPROP_TYPE] == LAYER_NET, "Could not find the locate the network info layer. " \
                                              "Please check the cfg file."

    module_list = nn.ModuleList()
    prev_filters = 3  # For the 3 channels of an RGB image
    output_filters = []  # to keep track of the output dimensions of each layer
    filters = None

    # Begin the processing of the remaining blocks of the network.
    # Check the type of the module and create a pytorch implementation of it.
    # Append the layer to the module_list
    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()  # for grouping layers such as conv, activation and batch norm

        if block[LPROP_TYPE] == LAYER_CONVOLUTIONAL:
            activation_type = block[LPROP_ACTIVATION]
            filters = int(block[LPROP_FILTERS])
            padding = int(block[LPROP_PAD])
            kernel_size = int(block[LPROP_SIZE])
            stride = int(block[LPROP_STRIDE])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            try:
                batch_normalize = int(block[LPROP_BATCH_NORMALIZE])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            # Add the convolutional layer to module
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{}".format(index), conv)

            # Add the Batch Norm Layer to module if needed
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)

            # Check the activation.
            # It is either RELU or a Leaky ReLU for YOLO
            if activation_type == LAYER_LEAKY:
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
            elif activation_type == LAYER_RELU:
                activn = nn.ReLU(inplace=True)
                module.add_module("relu_{0}".format(index), activn)

        elif block[LPROP_TYPE] == LAYER_UPSAMPLE:
            stride = int(block[LPROP_STRIDE])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")  # TODO: Fix the constant scale value
            module.add_module("upsample_{}".format(index), upsample)

        elif block[LPROP_TYPE] == LAYER_ROUTE:
            layers = block[LPROP_LAYERS].split(',')

            # Start  of a route
            start = int(layers[0])

            # end, if there exists one.
            try:
                end = int(layers[1])
            except:
                end = 0

            # Positive notation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif block[LPROP_TYPE] == LAYER_SHORTCUT:
            # shortcut corresponds to skip connection
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif block[LPROP_TYPE] == LAYER_YOLO:
            mask = block[LPROP_MASK].split(",")
            mask = [int(x) for x in mask]

            anchors = block[LPROP_ANCHORS].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]  # pick the anchors listed in the mask

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        else:
            assert False, "Unknown layer encountered"

        module_list.append(module)
        prev_filters = filters  # TODO: Improve the tracking of filters value
        output_filters.append(filters)
    return net_info, module_list


def transform_predictions(predictions, input_dim, anchors, num_classes, gpu=False):
    """
    Since the outputs produced by the YOLO layer are tedious to transform, we first flatten
    the predictions into a 2D array where each 1D array corresponds to a bounding box.

    Next, we offset the center coordinates by the grid offsets and log transform the
    dimensions of the bounding box by the size of the anchor. We also apply sigmoid
    to the confidence score and the class scores.

    :param predictions: The raw predictions from the YOLO layer.
    :param input_dim: The dimension of the input image. Assumes that the image is square.
    :param anchors: A list of anchors used by the detection layer
    :param num_classes: Number of output classes to be predicted
    :param gpu: Boolean flag to denote if compute must be on CPU (False) or GPU(True). Default is False.
    :return predictions: 2D array consisting of transformed bounding box predictions.
    """
    # TODO: Optimize this function, understand the anchors part and improve documentation.
    batch_size = predictions.size(0)
    stride = input_dim // predictions.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes  # (x, y, w, h, conf.)
    num_anchors = len(anchors)

    # The anchors are expressed as a factor of the stride, so we get the original dimensions
    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]

    predictions = predictions.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    predictions = predictions.transpose(1, 2).contiguous()  # the dimensions 1 & 2 are swapped
    predictions = predictions.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Now apply sigmoid to the center coordinates, confidence score and the class scores
    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])  # x value
    predictions[:, :, 1] = torch.sigmoid(predictions[:, :, 1])  # y value
    predictions[:, :, 4] = torch.sigmoid(predictions[:, :, 4])  # confidence score

    predictions[:, :, 5:(5 + num_classes)] = torch.sigmoid(predictions[:, :, 5:(5 + num_classes)])  # class scores

    # Create and add the grid offsets to each of the center coordinates
    # to reveal the global position of the bounding boxes.
    grid = np.arange(grid_size)
    x_s, y_s = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(x_s).view(-1, 1)
    y_offset = torch.FloatTensor(y_s).view(-1, 1)

    if gpu:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    predictions[:, :, :2] += x_y_offset  # offset the coordinates

    # Log transform the height and width of the bounding box using the formula:
    # new_dimension = anchor_dimension * exp(old_dimension)
    anchors = torch.FloatTensor(anchors)

    if gpu:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    predictions[:, :, 2:4] = torch.exp(predictions[:, :, 2:4]) * anchors  # log transform the dimensions

    # Now scale back everything to the size of the input image
    predictions[:, :, :4] *= stride

    return predictions


def bbox_iou(box1, box2):
    """
    Calculates the accuracy of bounding box 1 with the bounding box 2. If there are more than one in
    the box2 tensor, then it calculates the IOU with all the bounding boxes in box2.
    :param box1: Tensor containing coordinates of box 1.
    :param box2: Tensor containing coordinates of all the box for which we need to calculate IOU.
    :return iou: Tensor containing the IOUs of every bounding box in box2 with box1.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def filter_transform_predictions(predictions, num_classes, confidence_threshold=0, nms_threshold=0):
    """
    Filters existing prediction by applying confidence thresholding and non-max suppression to
    bounding boxes. The functions also flattens all the predictions for all the images into a
    single D x 8 tensor where D is the total number of true detections across all images and the
    8 parameters for each bounding box are as follows:
    1) Index of which image in the batch does the bounding box belong to.
    2) Top-left X coordinate.
    3) Top-left Y coordinate.
    4) Bottom-right X coordinate.
    5) Bottom-right Y coordinate.
    6) Objectness (confidence) score.
    7) Score of the class with the maximum confidence.
    8) Index representing the class in 7th parameter.

    :param predictions: Tensor consisting of all the bounding box predictions of all the images.
    :param num_classes: Number of object classes possible.
    :param confidence_threshold: The threshold value to filter boxes with low confidence.
    :param nms_threshold: The threshold value for filtering bounding boxes based on IOU in the non-max suppression part.
    :return output: Tensor of shape D x 8 containing true detections. Returns 0 if not true detections were found.
    """

    # Create a mask after applying threshold to confidence scores to identify confident bounding boxes
    conf_mask = (predictions[:, :, 4] > confidence_threshold).float().unsqueeze(2)
    predictions *= conf_mask  # suppress the invalid bounding boxes by making their confidence scores equal to zero

    # Since it is easier to calculate IOU when the coordinates are in the form:
    # <top-left corner x, top-left corner y, bottom-right corner x, bottom-right corner y>
    # instead of the form:
    # <center x, center y, height, width>
    # So we transform the coordinates of the bounding boxes accordingly.
    box_corners = predictions.new(predictions.shape)
    box_corners[:, :, 0] = (predictions[:, :, 0] - predictions[:, :, 2] / 2)  # cx - (width) / 2
    box_corners[:, :, 1] = (predictions[:, :, 1] - predictions[:, :, 3] / 2)  # cy - (height) / 2
    box_corners[:, :, 2] = (predictions[:, :, 0] + predictions[:, :, 2] / 2)  # cx + (width) / 2
    box_corners[:, :, 3] = (predictions[:, :, 1] + predictions[:, :, 3] / 2)  # cy + (height) / 2
    predictions[:, :, :4] = box_corners[:, :, :4]  # copy the new coordinates back to the original predictions tensor

    # Since the number of final predictions after filtering is different
    # for each image in the batch, they need to processed individually.
    batch_size = predictions.size(0)

    write = False
    full_ious = []
    full_class_scores = []

    count = 0
    for batch_index in range(batch_size):

        image_predictions = predictions[batch_index]  # get all bounding boxes for this image

        # Since the number of classes to be predicted could be quite high and we only
        # care about the one with the highest class score, so we remove all the classes
        # and instead just hold an index representing the class with the highest score
        # and the score of the class at that index.
        max_conf_class_score, max_conf_class_index = torch.max(image_predictions[:, 5: (5 + num_classes)], dim=1)
        max_conf_class_score = max_conf_class_score.float().unsqueeze(1)
        max_conf_class_index = max_conf_class_index.float().unsqueeze(1)

        image_predictions = torch.cat((image_predictions[:, :5], max_conf_class_score, max_conf_class_index), 1)
        # Next, we filter out all the boxes which had 0 in their confidence score
        non_zero_indices = torch.nonzero(image_predictions[:, 4])  # 4 is the index holding the confidence score

        if non_zero_indices.size(0) == 0:  # if there are no valid bounding boxes for this image, then skip
            #full_ious.append(0)
            continue
        # The number 7 below refers to the following 7 attributes:
        # 4 coordinates + 1 confidence score + 1 class score + 1 class index
        confident_image_predictions = image_predictions[non_zero_indices.squeeze(), :].view(-1, 7)

        # Get a list of unique classes which have been predicted by the bounding boxes.
        # The class indices lie in the last index of the bounding box.
        img_classes = torch.unique(confident_image_predictions[:, -1])


        # We now perform non-max suppression class-wise.
        for class_ in img_classes:
            #print("cls", class_)
            # First we gather the boxes which predict the above class
            image_predictions_class = confident_image_predictions[confident_image_predictions[:, -1] ==
                                                                  class_].view(-1, 7)
            # Next we sort the predictions in the descending order of the confidence score
            sorted_indices = torch.sort(image_predictions_class[:, 4], descending=True)[1]
            sorted_predictions = image_predictions_class[sorted_indices]
            num_detections = sorted_predictions.size(0)  # get the number of detections of this class
            if num_detections == 0:
                break
            for idx in range(num_detections):
                # We calculate the IOU of the current bounding box with all those present after this one
                # and remove the ones with an IOU > nms_threshold
                #print(confident_image_predictions[idx][5])
                try:
                    ious = bbox_iou(sorted_predictions[idx].unsqueeze(0), sorted_predictions[idx+1:])

                    #full_ious.append(ious)
                except (ValueError, IndexError):
                    #full_ious.append(0)
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_threshold).float().unsqueeze(1)

                sorted_predictions[idx+1:] *= iou_mask
                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(sorted_predictions[:, 4]).squeeze()
                sorted_predictions = sorted_predictions[non_zero_ind].view(-1, 7)

            # Now since we flatten all the bounding boxes into a single array, we need to add a number
            # to each bounding box denoting which image it belongs to.
            batch_idx_tensor = sorted_predictions.new(sorted_predictions.size(0), 1).fill_(batch_index)
            seq = batch_idx_tensor, sorted_predictions
            #print('bidx', batch_idx_tensor)


            for i in range(idx):
               full_class_scores.append(sorted_predictions[i][5])

            if len(ious) > 0:
                for idx in batch_idx_tensor:
                    #print(ious[int(idx.item())])
                    full_ious.append(ious[int(idx.item())].item())

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    # TODO: Optimize the flow of output and nms calculation.
    try:
        #print(full_ious)
        return output, full_class_scores
    except:
        return 0, []


def load_classes(names_file):
    """
    Loads the names of the classes from the input file.
    :param names_file: Path to the file containing the list of class names.
    :return class_names: List of class names.
    """
    print(names_file)
    with open(names_file, 'r') as file_names:
        class_names = file_names.readlines()
    class_names = list(filter(len, map(str.strip, class_names)))
    return class_names


def parse_data_cfg(data_cfg_file):
    """
    Parses the data cfg file and returns a dictionary containing the configuration details.
    :param data_cfg_file: Path to the data cfg file.
    :return config: Dictionary containing the configuration specified in the input file.
    """

    with open(data_cfg_file, 'r') as file_names:
        config_lines = file_names.readlines()
    config = {}
    for config_line in config_lines:
        if config_line[0] == "#":
            continue
        key, value = config_line.split('=')
        # print(key, value)

        config[key.rstrip()] = value.lstrip()

    return config


def image_to_letterbox(image, input_dim):
    """
    Resize the image without changing the aspect ratio. Pads the extra region with
    (128, 128, 128) and gives it a letterbox effect.
    :param image: The image which needs to be resized.
    :param input_dim: Specifies the dimensions to which the image must be resized.
    :return final_image: Final resized image.
    """
    old_w, old_h = image.shape[1], image.shape[0]
    new_w, new_h = input_dim
    scale_ratio = min(new_w / old_w, new_h / old_h)
    final_w = int(old_w * scale_ratio)
    final_h = int(old_h * scale_ratio)
    resized_image = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_CUBIC)

    final_image = np.full((new_w, new_h, 3), 128)
    w_start, h_start = (new_w - final_w) // 2, (new_h - final_h) // 2
    final_image[h_start:h_start + final_h, w_start:w_start + final_w, :] = resized_image
    return final_image


def prepare_image(image, input_dim):
    """
    Prepares the image for input to the neural network. Resizes the image and converts to PyTorch Variable.
    :param image: The image which needs to be prepared.
    :param input_dim: Specifies the dimensions to which the image must be resized.
    :return image: PyTorch Variable containing the final resized image.
    """
    image = image_to_letterbox(image, input_dim)
    image = image[:, :, ::-1].transpose((2, 0, 1)).copy()
    image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    return image


# Test code
if __name__ == '__main__':
    blocks = parse_cfg("./cfg/yolov3-amp.cfg")
    network_info, module_list = create_network(blocks)
    print(len(blocks), len(module_list))
    print(module_list)
