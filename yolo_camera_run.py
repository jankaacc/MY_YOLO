import cv2
import freenect
import numpy as np
import PIL
import tensorflow as tf

from keras import backend as K
from keras.models import load_model

from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    """

    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    filtering_mask = (box_class_scores > threshold)
    scores = tf.boolean_mask(tensor=box_class_scores, mask=filtering_mask)
    boxes = tf.boolean_mask(tensor=boxes, mask=filtering_mask)
    classes = tf.boolean_mask(tensor=box_classes, mask=filtering_mask)

    return scores, boxes, classes



def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """
    xi1 = np.max([box1[0], box2[0]])
    yi1 = np.max([box1[1], box2[1]])
    xi2 = np.min([box1[2], box2[2]])
    yi2 = np.min([box1[2], box2[2]])
    inter_area = np.abs(xi1 - xi2) * np.abs(yi1 - yi2)
    box1_area = np.abs(box1[0] - box1[2]) * np.abs(box1[1] - box1[3])
    box2_area = np.abs(box2[0] - box2[2]) * np.abs(box2[1] - box2[3])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / float(union_area)

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=iou_threshold)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes



def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.4, iou_threshold=.5):#
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[:]
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    return scores, boxes, classes


sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (480., 640.)
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """
    # np.savetxt("output.txt", dept_array, fmt="%i")
    # print(dept_array)

    if image_file.__class__ == str:
        image, image_data = preprocess_image("images/" + image_file, model_image_size=(416, 416))
    else:
        image = PIL.Image.fromarray(image_file)
        image, image_data = preprocess_image(image,model_image_size=(416, 416))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    # print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    labels = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    return out_scores, out_boxes, out_classes, image, labels

#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
    array_mm,_ = freenect.sync_get_depth()
    array = array_mm.astype(np.uint8)
    return array, array_mm

#function to get object distance
def get_object_distance(out_boxes, depth_mm, labels):
    out_boxes[out_boxes<0] = 0
    out_boxes = out_boxes.astype(int)
    items = [depth_mm[box[1]:box[3], box[0]:box[2]] for box in out_boxes]
    print("test run: \n\n")
    for index, i in enumerate(items[::-1]):
        # np.savetxt("{}".format(labels[index]), i, fmt="%i")
        print("{}, distance: {}, shape: {}".format(labels[index], np.median(i), i.shape ))


# cap = cv2.VideoCapture(0)


flag = True

while(True):

    #ret, frame = cap.read()
    # get a frame from RGB camera
    frame = get_video()
    # get a frame from depth sensor
    depth, depth_mm = get_depth()
    cv2.imshow('frame',frame)
    cv2.imshow('Depth image', depth)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord(' '):
        out_scores, out_boxes, out_classes, image, labels = predict(sess, frame)
        get_object_distance(out_boxes, depth_mm, labels)
        #convert PIL image to opencv format
        open_cv_image = np.array(image)
        cv2.imshow("Image", open_cv_image)
        

# cap.release()
cv2.destroyAllWindows()
freenect.sync_stop()



#tensorflow Object-detection API
