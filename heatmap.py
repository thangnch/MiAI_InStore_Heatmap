import cv2
import numpy as np
from imutils.video import VideoStream

# Parameters
video_file = "video.mp4"
classnames_file = "classnames.txt"
weights_file = "yolov4-tiny.weights"
config_file = "yolov4-tiny.cfg"
conf_threshold = 0.5
nms_threshold = 0.4
detect_class = "person"

frame_width = 1280
frame_height = 720
cell_size = 40 # 40x40 pixel
n_cols = frame_width // cell_size
n_rows = frame_height // cell_size
alpha = 0.4

heat_matrix = np.zeros((n_rows, n_cols))
scale = 0.00392 # Chua tim ra nguon goc

yolo_net = cv2.dnn.readNet(weights_file, config_file)

# Doc ten cac class
classes = None
with open(classnames_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Ham tra ve dong va cot tu toa do x, y
def get_row_col(x, y):
    row = y// cell_size
    col = x //cell_size
    return row, col

def draw_grid(image):
    for i in range(n_rows):
        start_point = (0, (i + 1) * cell_size)
        end_point = (frame_width, (i + 1) * cell_size)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    for i in range(n_cols):
        start_point = ((i + 1) * cell_size, 0)
        end_point = ((i + 1) * cell_size, frame_height)
        color = (255, 255, 255)
        thickness = 1
        image = cv2.line(image, start_point, end_point, color, thickness)

    return image

# Ham tra ve output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    global heat_matrix
    r, c = get_row_col( (x_plus_w + x)//2, (y_plus_h + y)//2)
    heat_matrix[r,c] +=1

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

video = VideoStream(video_file).start()

while True:
    frame = video.read()
    # Nhan dien vat the trong khung hinh
    blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(get_output_layers(yolo_net))

    # Loc cac object trong khung hinh
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence > conf_threshold) and (classes[class_id]==detect_class):
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Ve cac khung chu nhat quanh doi tuong
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], round(x), round(y), round(x + w), round(y + h))

    from skimage.transform import resize
    temp_heat_matrix = heat_matrix.copy()
    temp_heat_matrix = resize(temp_heat_matrix, (frame_height, frame_width))
    temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
    temp_heat_matrix = np.uint8(temp_heat_matrix*255)

    # Tao heat map
    image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

    frame = draw_grid(frame)
    # Chong hinh
    cv2.addWeighted(image_heat, alpha, frame, 1-alpha, 0, frame)



    cv2.imshow("Video", frame)
    # cv2.imshow("Heatmap", image_heat)
    if cv2.waitKey(1)==ord('q'):
        break


cv2.destroyAllWindows()
