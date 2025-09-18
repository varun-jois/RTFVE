
import time
import cv2 as cv
import mediapipe as mp
import numpy as np
import copy
import torch
from glob import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as F
from utils.utils import cfg_from_log
from models.RTFVE import RTFVE as Net


mp_face_detection = mp.solutions.face_detection
fpath = ''  # path to low quality face video file
rpath = ''  # path to folder with high quality reference images
expnum = 0  #  23  34
device = 'cpu'
features_bool = True
refs_num = 1

# torch.set_num_threads(1)  # Set to number of physical cores
# Disable automatic mixed precision if enabled
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 3. Disable gradients
torch.set_grad_enabled(False)

# torch.set_num_threads(1)
WHITE = [255, 255, 255]
RED = [0, 0, 255]
GREEN = [0, 166, 28]

cap = cv.VideoCapture(fpath)

# get the model hyperparams from the log
cfg = cfg_from_log(f'checkpoints/exp_{expnum}/exp_{expnum}.log')
mdl = cfg['model']

# load torch model
model = Net(**mdl).to(device)
model.load_state_dict(torch.load(f'checkpoints/exp_{expnum}/best_mdl.pth', map_location=device))
model.eval()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# load refs
refs = [to_tensor(Image.open(i)).unsqueeze(0).to(device) for n, i in enumerate(glob(f'{rpath}/*')) if n < refs_num]

# extract ref features
if features_bool:
    refs = model.extract_ref_features(refs)

model = torch.jit.script(model).to(device)


with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.25) as face_detector:
        im_counter = 0
        frame_counter = 2000
        fonts = cv.FONT_HERSHEY_PLAIN
        start_time = time.time()
        times = []
        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            st = time.time()
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = face_detector.process(rgb_frame)
            frame_height, frame_width, c = frame.shape
            if results.detections:
                face = results.detections[0]
                face_rect = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height,
                    ],
                    [frame_width, frame_height, frame_width, frame_height]).astype(int)
                    
                x, y, w, h = face_rect
                mid_x, mid_y = (x + w // 2), (y + h // 2)
                size = 256 // 2
                sr = copy.deepcopy(frame)
                crop = frame[mid_y-size:mid_y+size, mid_x-size:mid_x+size]
                if crop.shape != (256, 256, 3):
                    im_counter += 1
                    continue

                with torch.inference_mode():
                    lq = F.gaussian_blur(to_tensor(cv.cvtColor(crop, cv.COLOR_BGR2RGB)).to(device), 3, 0.8).unsqueeze(0)
                    if features_bool:
                        pr = model(lq, refs, features=features_bool)
                    else:
                        pr = model(lq, refs)
                    times.append(time.time() - st)
                    print(times[-1])
                    res = pr.mul(255).squeeze(0).permute(1, 2, 0).int().detach().cpu().numpy().astype(np.uint8)
                
                # insert the face back into the frame
                cv.rectangle(frame, np.array([mid_x-size, mid_y-size, 256, 256]), color=(255, 255, 255), thickness=2)
                sr[mid_y-size:mid_y+size, mid_x-size:mid_x+size] = cv.cvtColor(res, cv.COLOR_RGB2BGR)
                cv.rectangle(sr, np.array([mid_x-size, mid_y-size, 256, 256]), color=(255, 255, 255), thickness=2)
                ct = np.concatenate([frame, sr], 1)

            cv.imshow("frame", ct)
            key = cv.waitKey(1)
            if key == ord("q"):
                break

        print(f'total time {time.time() - start}')
        cap.release()
        cv.destroyAllWindows()

import statistics
mean = statistics.mean(times)
median = statistics.median(times)
minv = min(times)
maxv = max(times)
stdv = statistics.pstdev(times, mean)
print(f'mean: {mean}')
print(f'median: {median}')
print(f'min: {minv}')
print(f'max: {maxv}')
print(f'stdv: {stdv}')

print(f'mean fps: {1/mean}')
print(f'median fps: {1/median}')
print(f'min fps: {1/maxv}')
print(f'max fps: {1/minv}')
print(f'stdv fps: {1/stdv}')
print(f'total frames: {len(times)}')
print(device)

