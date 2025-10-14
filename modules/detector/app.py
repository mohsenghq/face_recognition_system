import cv2
import torch
import numpy as np
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

# --- GPU/CPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# --- parameters---
CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
RESIZE = 1  

# ---Retinaface
CFG_MNET = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'in_channel': 32,
    'out_channel': 64
}

# ---loading model---
def load_retinaface(weight_path: str, cfg=CFG_MNET):
    net = RetinaFace(cfg=cfg, phase='test')
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    net.to(device).eval()
    return net



def detect_faces(net, frame, conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD):
    """
    input:
        net: loading RetinaFace
        frame: image BGR (numpy array)
    output:
        dets: numpy array shape (N,5) with [x1,y1,x2,y2,score]
        landms: numpy array shape (N,10) landmarks (x1,y1,...,x5,y5)
    """
    img = np.float32(frame)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])

    # mean subtraction (BGR)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loc, conf, landms = net(img)

    priorbox = PriorBox(CFG_MNET, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)

    boxes = decode(loc.data.squeeze(0), priors, CFG_MNET['variance'])
    boxes = boxes * scale / RESIZE
    boxes = boxes.cpu().numpy()

    scores = conf.data.squeeze(0)[:, 1].cpu().numpy()
    landms = decode_landm(landms.data.squeeze(0), priors, CFG_MNET['variance'])
    scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height])
    landms = landms * scale1 / RESIZE
    landms = landms.cpu().numpy()

    # filter
    inds = np.where(scores > conf_threshold)[0]
    if inds.size == 0:
        return np.empty((0,5)), np.empty((0,10))
    boxes, landms, scores = boxes[inds], landms[inds], scores[inds]
    order = scores.argsort()[::-1]
    boxes, landms, scores = boxes[order], landms[order], scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets, landms = dets[keep, :], landms[keep]

    return dets, landms
