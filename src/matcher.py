import cv2
import numpy as np

class Matcher:
    def __init__(self, follower_extractor, lead_extractor, RANSAC=True):
        self.follower_extractor = follower_extractor
        self.lead_extractor = lead_extractor

        self.follower_keypoints, self.follower_descriptors, self.follower_frames = self.follower_extractor.get_params()
        self.lead_keypoints, self.lead_descriptors, self.lead_frames = self.lead_extractor.get_params()
        self.RANSAC = RANSAC

        self.matches = None
        self.batch = False

    def match_features(self, matcher_type='bf', ratio_thresh=0.75):
        if type(self.follower_keypoints) is list:
            self.matches = self.match_feature_batch(matcher_type=matcher_type, ratio_thresh=ratio_thresh)
            self.batch = True
        else:
            self.matches = self.match_feature(self.follower_keypoints, self.follower_descriptors, self.lead_keypoints, self.lead_descriptors, matcher_type=matcher_type, ratio_thresh=ratio_thresh)

# Match features between two frames or two sets of frames
    def match_feature(self, follower_kp, follower_desc, lead_kp, lead_desc, matcher_type='bf', ratio_thresh=0.75):
        if matcher_type == 'bf':
            # initialize a Brute-Force Matcher
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher_type == 'flann':
            # initialize a FLANN Matcher
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Invalid matcher type: {matcher_type}")

        # match descriptors of the two images
        matches = matcher.match(follower_desc.astype(np.uint8), lead_desc.astype(np.uint8))

        # filter matches using Lowe's ratio test
        good_matches = []

        # Lowe's ratio test sucks
        '''
        for m in matches:
            if len(matches) > 1:
                if m.distance < ratio_thresh * matches[1].distance:
                    good_matches.append(m)
            else:
                if m.distance < ratio_thresh * 500:
                    good_matches.append(m)
        '''

        if self.RANSAC: 
            follower_pts = np.float32([follower_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            lead_pts = np.float32([lead_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

            # Filter using RANSAC
            if len(matches) >= 4:
                homography, mask = cv2.findHomography(follower_pts, lead_pts, cv2.RANSAC, 7.0)
                matches_mask = mask.ravel().tolist()
                good_matches = [m for i, m in enumerate(matches) if matches_mask[i]]

        return good_matches

# Does feature matching for a batch of frames
    def match_feature_batch(self, matcher_type='bf', ratio_thresh=0.75):
        batch_match = []
        for follower_kp, follower_desc, lead_kp, lead_desc in zip(self.follower_keypoints, self.follower_descriptors, self.lead_keypoints, self.lead_descriptors):
            match = self.match_feature(follower_kp, follower_desc, lead_kp, lead_desc, matcher_type=matcher_type, ratio_thresh=ratio_thresh)
            batch_match.append(match)
        return batch_match

    def visualize_match_single(self, img1, keypoints1, img2, keypoints2, matches, show=False):
        vis_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)

        if show:
            cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Matches", (960, 480))
            cv2.imshow("Matches", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_img

    def vizualiize_match_batch(self, img1, keypoints1, img2, keypoints2, matches, show=False):
        vis_imgs = [self.visualize_match_single(img1, keypoints1, img2, keypoints2, matches, show=show) for img1, keypoints1, img2, keypoints2, matches in zip(img1, keypoints1, img2, keypoints2, matches)]

        return vis_imgs

    def visualize_matches(self, show=False):
        if self.batch:
            return self.vizualiize_match_batch(self.follower_frames, self.follower_keypoints, self.lead_frames, self.lead_keypoints, self.matches, show=show)
        else:
            return self.visualize_match_single(self.follower_frames, self.follower_keypoints, self.lead_frames, self.lead_keypoints, self.matches, show=show)
