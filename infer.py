import cv2
import torch

from models.yolo import Model
from utils.datasets import letterbox
import numpy as np

from utils.general import non_max_suppression, scale_coords, clip_coords
from utils.plots import plot_images, plot_one_box
from utils.torch_utils import intersect_dicts


def load_image(path, img_size):
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

if __name__ == '__main__':
    conf_thres = 0.15
    iou_thres = 0.55
    path = '/media/panda/0CFC3A54FC3A37F2/dataset/yolor_dataset/images/valid/0-9679.jpg'
    img_size = 1280
    img0, (h0, w0), (h, w) = load_image(path, img_size)
    img = letterbox(img0, new_shape=img_size, auto_size=64)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    device = torch.device('cuda', 0)
    half = True  # half precision only supported on CUDA

    ckpt = torch.load('/home/panda/Hobby/yolor/yolor_bs12v2/best_ap50.pt', map_location=device)  # load FP32 model
    model = Model(ckpt['model'].yaml, ch=3, nc=1).to(device)  # create
    exclude = ['anchor']  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    if half:
        model.half()  # to FP16

    model.eval()

    img_test = torch.zeros((1, 3, 768, 1280), device=device)  # init img
    _ = model(img_test.half() if half else img) if device.type != 'cpu' else None  # run once

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()  # uint8 to fp16/32
    img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    nb, _, height, width = img_tensor.shape  # batch size, channels, height, width

    with torch.no_grad():
        # Inference
        inf_out, train_out = model(img_tensor, augment=False)
        pred = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)

    predictions = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = 'cors'
                print((xyxy[2] - xyxy[0])*(xyxy[3] - xyxy[1]))
                plot_one_box(xyxy, img0, label=label, line_thickness=3)
                predictions.append('{:.3f} {} {} {} {}'.format(float(conf), int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])))


            cv2.imshow('fig', img0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise Exception

            cv2.imwrite('test.jpg', img0)

    prediction_str = ' '.join(predictions)
    print(prediction_str)