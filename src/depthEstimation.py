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
    def __init__(self, model_path=None, model_type='DPT_Large', device='cpu'):
        self.model_type = model_type
        self.device = device # cpu or cuda

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

        self.depth_map = None

    def predict_depth(self, img):
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = self.transform(img)
        with torch.no_grad():
            prediction = self.midas(inp.to(self.device))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        result = self.inv_depth_to_depth(prediction.cpu().numpy())
        self.depth_maps = result
        return result

    def inv_depth_to_depth(self, inv_map):
        return 1 / (inv_map + 1e-6) # add small epsilon to avoid division by zero

    def updateDepthEstimates(depthMap, knownMeasurements):
        '''
        Depth map estimate update

        Updates a depth map based on known measurements across the depth range.

            Parameters:
                depthMap (MxNx1 numpy array) : numpy matrix of estimated depth map.
                knownMeasurements: Nx3 numpy matrix, N points with [x, y, z] values.
                degree (int) : number of degrees for the polynomial regression model

            Returns:
                updatedDepthMap (MxNx1 numpy array) : numpy matrix of estimated depth map.
        '''
        if knownMeasurements is None:
            return depthMap
        knownMeasurements_idx = knownMeasurements[:, 0:2].astype(int)
        knownMeasurements = knownMeasurements[:, 2]

        if(len(knownMeasurements) < 1):
            print("No correction points")
            return depthMap
        elif(len(knownMeasurements) == 1):
            depthValue = depthMap[knownMeasurements_idx[0, 0], knownMeasurements_idx[0, 1]]
            scalingFactor = knownMeasurements[0]/depthValue
            print("scaling factor =", knownMeasurements[0], "/", depthValue, "=", scalingFactor)
            #print("original(",knownMeasurements[0, 0])
            updatedDepthMap = depthMap * scalingFactor

        else:
            #get known measurements (normally retrived through object detection)
            true_depths = knownMeasurements.reshape(-1, 1)

            #find the corresponding points in the depth map estimate
            corresponding_depths = depthMap[knownMeasurements_idx[:, 0], knownMeasurements_idx[:, 1]].reshape(-1, 1)

            x, _, _, _ = np.linalg.lstsq(corresponding_depths, true_depths, rcond=None)

            print("scaling factor =", x[0])
            updatedDepthMap = depthMap * x[0]

        return updatedDepthMap


