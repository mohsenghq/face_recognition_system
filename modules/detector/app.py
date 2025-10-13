# app.py 
import cv2
import torch
import numpy as np
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  

# ---------- settings----------
conf_threshold = 0.6
nms_threshold = 0.4
vis_thres = 0.6
resize = 1

# ---------- loading model----------
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

net = RetinaFace(cfg=cfg_mnet, phase='test')
net = net.to(device)
state_dict = torch.load("weights/mobilenet0.25_Final.pth", map_location=device)
net.load_state_dict(state_dict, strict=False)
net.eval()
print("✅ RetinaFace MobileNet model loaded.")

# ---------- function ----------
def detect_faces(frame):
    img = np.float32(frame)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])

    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(), torch.no_grad():
        loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    boxes = decode(loc.data.squeeze(0), priors, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.data.squeeze(0)[:, 1].cpu().numpy()
    landms = decode_landm(landms.data.squeeze(0), priors, cfg_mnet['variance'])
    scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height])
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # filtering
    inds = np.where(scores > conf_threshold)[0]
    boxes, landms, scores = boxes[inds], landms[inds], scores[inds]
    order = scores.argsort()[::-1]
    boxes, landms, scores = boxes[order], landms[order], scores[order]

    # NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets, landms = dets[keep, :], landms[keep]

    return dets, landms

# ---------- Real-time ----------
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets, landms = detect_faces(frame)

    # bounding box
    for b, l in zip(dets, landms):
        if b[4] < vis_thres:
            continue
        text = f"{b[4]:.2f}"
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = x1, y1 + 12
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
        for i in range(5):
            cv2.circle(frame, (int(l[2*i]), int(l[2*i+1])), 2, (0,0,255), -1)

    cv2.imshow("RetinaFace Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
