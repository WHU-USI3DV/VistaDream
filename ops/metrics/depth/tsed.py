import cv2,os
import numpy as np
from tqdm import tqdm

def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[:2]
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()
    
    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r_line[2]/r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0]*c)/r_line[1]])
        img1_color = cv2.line(img1_color, (x0,y0), (x1,y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(pt1), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(pt2), 5, color, -1)
    return img1_color, img2_color

class TSED_Tool():
    def __init__(self):
        pass
    
    def _extract_matches_(self,simg,timg,K1,K2):
        simg = cv2.cvtColor((simg*255.).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        timg = cv2.cvtColor((timg*255.).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # 1. 提取SIFT特征和关键点
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(simg, None)
        keypoints2, descriptors2 = sift.detectAndCompute(timg, None)
        # 2. 基于Mutual NN进行特征匹配
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        # 使用Ratio Test进行筛选
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # 获取匹配的关键点
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        if len(pts1)<10: return None,None,None
        # 使用对应点计算本质矩阵E（Essential Matrix）
        E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K1)

        # 获取基础矩阵F（Fundamental Matrix）
        F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)

        return pts1,pts2,F
    
    def _solve_F_(self,K1,E1,K2,E2):
        R1,T1 = E1[0:3,0:3],E1[0:3,-1]
        R2,T2 = E2[0:3,0:3],E2[0:3,-1]
        # 计算平移向量的叉乘矩阵
        T_diff = T2 - T1
        T_skew = np.array([
            [0, -T_diff[2], T_diff[1]],
            [T_diff[2], 0, -T_diff[0]],
            [-T_diff[1], T_diff[0], 0]
        ])
        # 计算本质矩阵 E
        E = T_skew @ R2.T @ R1
        # 计算基础矩阵 F
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        return F
    
    def _calculate_error_(self,pts1,pts2,F):
        # 4. 计算极线差距
        epipolar_errors = []
        for pt1, pt2 in zip(pts1, pts2):
            x1, y1 = pt1
            x2, y2 = pt2

            # 计算点1的极线（通过基础矩阵）
            epiline1 = np.dot(F, np.array([x1, y1, 1]))
            # 计算点2的极线（通过基础矩阵）
            epiline2 = np.dot(F.T, np.array([x2, y2, 1]))

            # 计算每个对应点的极线差距
            error1 = np.abs(epiline1[0]*x2 + epiline1[1]*y2 + epiline1[2]) / np.sqrt(epiline1[0]**2 + epiline1[1]**2)
            error2 = np.abs(epiline2[0]*x1 + epiline2[1]*y1 + epiline2[2]) / np.sqrt(epiline2[0]**2 + epiline2[1]**2)

            epipolar_errors.append((error1 + error2) / 2)

        # 5. 计算平均极线差距
        median_epipolar_error = np.percentile(np.array(epipolar_errors),60)
        return median_epipolar_error
    
    def _pair_process_(self,IMG1,K1,IMG2,K2):
        try:
            pts1,pts2,F = self._extract_matches_(IMG1,IMG2,K1,K2)
        except:
            return 1000
        if pts1 is None: return 1000
        err = self._calculate_error_(pts1,pts2,F)
        return err
    
    def __call__(self,video_fn,nframes=50,K=None,thres=0.7):
        cap = cv2.VideoCapture(video_fn)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break 
            frame = frame[...,[2,1,0]]
            frames.append(frame)
        sample = np.linspace(0,len(frames)-1,nframes).astype(np.int32)
        frames = [frames[i] for i in sample]
        if K is None:
            raise ValueError('We need intrinsic')
        else:
            intrinsics = K[None].repeat(nframes,axis=0)
            cap = cv2.VideoCapture(video_fn)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break 
                frame = frame[...,[2,1,0]]
                frames.append(frame)
            sample = np.linspace(0,len(frames)-1,nframes).astype(np.int32)
            imgs = [frames[i] for i in sample]
            # reshape to (512,512)
            K[0] = K[0]/imgs[0].shape[1]*512
            K[1] = K[1]/imgs[0].shape[0]*512
            imgs = [cv2.resize(img,(512,512)) for img in imgs]
        errors = []
        for i in tqdm(range(len(imgs)-1)):
            s,t = i,i+1
            err = self._pair_process_(imgs[s],intrinsics[s],
                                      imgs[t],intrinsics[t])
            errors.append(err)
        errors = np.array(errors)
        result = np.mean(errors<thres)
        return result
    