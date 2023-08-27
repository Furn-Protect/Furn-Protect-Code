from mmpose.apis import MMPoseInferencer
#from mmpose.apis import inference_top_down_pose_model, init_pose_model
#from mmcv import Config
import numpy as np
import cv2
import numpy as np

data = {
    0: dict(name='L_Eye', id=0, color=[0, 255, 0], type='upper', swap='R_Eye'),
    1: dict(name='R_Eye', id=1, color=[255, 128, 0], type='upper', swap='L_Eye'),
    2: dict(name='L_EarBase', id=2, color=[0, 255, 0], type='upper', swap='R_EarBase'),
    3: dict(name='R_EarBase', id=3, color=[255, 128, 0], type='upper', swap='L_EarBase'),
    4: dict(name='Nose', id=4, color=[51, 153, 255], type='upper', swap=''),
    5: dict(name='Throat', id=5, color=[51, 153, 255], type='upper', swap=''),
    6: dict(name='TailBase', id=6, color=[51, 153, 255], type='lower', swap=''),
    7: dict(name='Withers', id=7, color=[51, 153, 255], type='upper', swap=''),
    8: dict(name='L_F_Elbow', id=8, color=[0, 255, 0], type='upper', swap='R_F_Elbow'),
    9: dict(name='R_F_Elbow', id=9, color=[255, 128, 0], type='upper', swap='L_F_Elbow'),
    10: dict(name='L_B_Elbow', id=10, color=[0, 255, 0], type='lower', swap='R_B_Elbow'),
    11: dict(name='R_B_Elbow', id=11, color=[255, 128, 0], type='lower', swap='L_B_Elbow'),
    12: dict(name='L_F_Knee', id=12, color=[0, 255, 0], type='upper', swap='R_F_Knee'),
    13: dict(name='R_F_Knee', id=13, color=[255, 128, 0], type='upper', swap='L_F_Knee'),
    14: dict(name='L_B_Knee', id=14, color=[0, 255, 0], type='lower', swap='R_B_Knee'),
    15: dict(name='R_B_Knee', id=15, color=[255, 128, 0], type='lower', swap='L_B_Knee'),
    16: dict(name='L_F_Paw', id=16, color=[0, 255, 0], type='upper', swap='R_F_Paw'),
    17: dict(name='R_F_Paw', id=17, color=[255, 128, 0], type='upper', swap='L_F_Paw'),
    18: dict(name='L_B_Paw', id=18, color=[0, 255, 0], type='lower', swap='R_B_Paw'),
    19: dict(name='R_B_Paw', id=19, color=[255, 128, 0], type='lower', swap='L_B_Paw')
}

name_list = [data[key]['name'] for key in data]

def calculate_angle(keypoints, point1_idx, point2_idx, point3_idx):
    point1 = np.array(keypoints[point1_idx])
    point2 = np.array(keypoints[point2_idx])
    point3 = np.array(keypoints[point3_idx])
    
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    
    angle_radians = np.arccos(dot_product / norm_product)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees




if __name__ == '__main__':
    
    img_path = 'C:\\Users\\15156\\Desktop\\test\\test\\17.png'   # 将img_path替换给你自己的路径
    
    image = cv2.imread(img_path)

    
    # 使用模型别名创建推断器
    inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_animalpose-256x256')
    # MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
    result_generator = inferencer(img_path, show=False)
    result = next(result_generator)
    predictions = result['predictions']
    dic = predictions[0][0]
    keypoints = dic['keypoints']
    
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint[0], keypoint[1]
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(image, str(i) + ' ' + name_list[i], (int(x) + 10, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    #cv2.imshow('Pose Estimation', image)
    cv2.imwrite('C:\\Users\\15156\\Desktop\\test\\test\\detected\\17.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 示例关键点索引
    point1_idx = 0
    point2_idx = 7
    point3_idx = 9

    # 计算角度
    angle = calculate_angle(keypoints, point1_idx, point2_idx, point3_idx)
    print("Angle between keypoints:", angle)
