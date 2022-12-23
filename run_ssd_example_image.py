from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import argparse

parser = argparse.ArgumentParser(
    description='SSD Video Example')
parser.add_argument("--input-source-path",  default="Test.jpg", type=str,
                    help='path of the video')
args = parser.parse_args()
net_type = 'mb2-ssd-lite'
model_path = "models\model_exoglove_07262022.pth"
label_path = "models\labels.txt"
image_path = args.input_source_path

class_names = [name.strip() for name in open(label_path).readlines()]


net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,  # font scale
                (255, 0, 255),
                10,
                cv2.LINE_AA)  # line type
path = f"{image_path}_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")