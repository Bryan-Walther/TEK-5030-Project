import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
'''
Uses a pre-trained model to estimate the depth of a monocular image.
Might be useful depending on how we want to try to estimate the depth.
It is fairly slow without a GPU, but it is possible to use in real time with a GPU.
'''
class DepthEstimator:
    def __init__(self, model_path=None, model_type='DPT_Large', device='cpu', depth_map_folder='None'):
        self.model_type = model_type
        self.device = device # cpu or cuda
        self.depth_map_folder = depth_map_folder

        if model_path is None:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device)
        else:
            self.midas = torch.jit.load(model_path).to(device)
        self.midas.eval()

        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

        self.input_batch = []
        self.depth_maps = []

    # Load images from folder
    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)

        self.input_batch = images

    def load_images_from_array(self, images):
        self.input_batch = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

    def predict_depth(self):
        result = []
        for i in range(len(self.input_batch)):
            img = self.input_batch[i]
            inp = self.transform(img)
            with torch.no_grad():
                prediction = self.midas(inp.to(self.device))
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            result.append(self.inv_depth_to_depth(prediction.numpy()))
        self.depth_maps = result
        return result

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






