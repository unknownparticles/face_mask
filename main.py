import cv2
import dlib
import numpy as np

class FaceMaskGenerator:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.region_definitions = {
            "face": list(range(0, 27)),
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48)),
            "nose": list(range(27, 36)),
            "mouth": list(range(48, 68)),
            "jaw": list(range(0, 17))
        }
    
    def get_mask(self, image, region="face", mask_dilation=3):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        # 创建空白遮罩
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for face in faces:
            # 获取关键点
            landmarks = self.predictor(gray, face)
            points = [(landmarks.part(n).x, landmarks.part(n).y) for n in self.region_definitions[region]]
            
            # 生成凸包或多边形
            if region in ["face", "jaw"]:
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
            else:
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
            
            # 形态学膨胀
            kernel = np.ones((mask_dilation, mask_dilation), np.uint8)
            mask = cv2.dilate(mask, kernel)
        
        return mask

    def visualize_mask(self, image, mask):
        # 创建带透明度的遮罩层
        overlay = image.copy()
        overlay[mask == 255] = (0, 255, 0)
        alpha = 0.3
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# 使用示例
if __name__ == "__main__":
    generator = FaceMaskGenerator()
    image = cv2.imread("test.jpg")
    
    # 生成各区域遮罩
    regions = ["face", "left_eye", "right_eye", "nose", "mouth"]
    for region in regions:
        mask = generator.get_mask(image, region)
        masked_img = generator.visualize_mask(image, mask)
        cv2.imwrite(f"{region}_mask.jpg", mask)
        cv2.imwrite(f"{region}_overlay.jpg", masked_img)