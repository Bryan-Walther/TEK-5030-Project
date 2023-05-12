import torch
import cv2
import numpy as np
from PIL import Image
'''
Uses a pre-trained model to estimate the depth of a monocular image.
Might be useful depending on how we want to try to estimate the depth.
It is fairly slow without a GPU, but it is possible to use in real time with a GPU.
'''
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DepthEstimatorZoe:
    def __init__(self, model_type='NK', device='cpu'):
        self.device = device # cpu or cuda
        self.model_type = model_type

        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        repo = "isl-org/ZoeDepth"

        if self.model_type == 'NK':
            self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True, force_reload=True, config_mode='eval').to(self.device)
        elif self.model_type == 'K':
            self.model = torch.hub.load(repo, "ZoeD_K", pretrained=True, force_reload=True, config_mode='eval').to(self.device)
        elif self.model_type == 'N':
            self.model = torch.hub.load(repo, "ZoeD_N", pretrained=True, force_reload=True, config_mode='eval').to(self.device)
        self.model.eval()
        self.scale_factors = []

    def predict_depth(self, img):
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth_numpy = self.model.infer_pil(inp)
        return depth_numpy

    def inv_depth_to_depth(self, inv_map):
        return 1 / (inv_map + 1e-6) # add small epsilon to avoid division by zero

    def updateDepthEstimates(self, depthMap, knownMeasurements):
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
            return depthMap * self.scale_factors[-1] if len(self.scale_factors) != 0 else depthMap
        knownMeasurements_idx = knownMeasurements[:, 0:2].astype(int)
        knownMeasurements = knownMeasurements[:, 2]

        if(len(knownMeasurements) < 1):
            print("No correction points")
            return depthMap * self.scale_factors[-1] if len(self.scale_factors) != 0 else depthMap

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
            self.scale_factors.append(x[0])
            print("Mean scaling factor =", np.mean(self.scale_factors))
            # weighted running average
            #weights = np.arange(1, len(self.scale_factors) + 1)
            #weighted_avg = np.sum(self.scale_factors * weights) / np.sum(weights)
            #updatedDepthMap = depthMap * x[0]
            updatedDepthMap = depthMap * np.mean(self.scale_factors)
            #updatedDepthMap = depthMap * weighted_avg

        return updatedDepthMap

