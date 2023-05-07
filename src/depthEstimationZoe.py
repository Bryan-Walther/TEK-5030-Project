import torch
import cv2
import numpy as np
from PIL import Image
'''
Uses a pre-trained model to estimate the depth of a monocular image.
Might be useful depending on how we want to try to estimate the depth.
It is fairly slow without a GPU, but it is possible to use in real time with a GPU.
'''
class DepthEstimatorZoe:
    def __init__(self, model_type='NK', device='cpu'):
        self.device = device # cpu or cuda
        self.model_type = model_type

        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        repo = "isl-org/ZoeDepth"

        if self.model_type == 'NK':
            self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True, force_reload=True, config_mode='eval').to(self.device)
        elif self.model_type == 'K':
            self.model = torch.hub.load(repo, "ZoeD_N", pretrained=True, force_reload=True, config_mode='eval').to(self.device)
        self.model.eval()

    def predict_depth(self, img):
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth_numpy = self.model.infer_pil(inp)
        return depth_numpy

    def inv_depth_to_depth(self, inv_map):
        return 1 / (inv_map + 1e-6) # add small epsilon to avoid division by zero

    # Ground truth is a numpy array of shape [n, 3] of n pixels with known depths.
    def align(ground_truth, depth_map):
        '''
        y = ground truth
        x = Midas depth
        s = scale param.
        t = shift param.

        solve for s and t
        '''
        y = ground_truth[:, 2]
        # Index into depth map at x, y
        x_idx = ground_truth[:, 0]
        y_idx = ground_truth[:, 1]
        x = detph_maps[x_idx, y_idx, :]
        A = np.vstack([x, np.ones(len(x))]).T
        s, t = np.linalg.lstsq(A, y, rcond=None)[0]
        aligned_depth = s * self.depth_maps + t # Do we multiply by s or 1/s?
        
        return aligned_depth






