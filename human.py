from mmpose.apis import MMPoseInferencer
#from mmpose.apis import inference_top_down_pose_model, init_pose_model
#from mmcv import Config
import numpy as np
import cv2
import numpy as np

data={
    0:dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
    1:dict(name='left_eye',id=1,color=[51, 153, 255],type='upper',swap='right_eye'),
    2:dict(name='right_eye',id=2,color=[51, 153, 255],type='upper',swap='left_eye'),
    3:dict(name='left_ear',id=3,color=[51, 153, 255],type='upper',swap='right_ear'),
    4:dict(name='right_ear',id=4,color=[51, 153, 255],type='upper',swap='left_ear'),
    5:dict(name='left_shoulder',id=5,color=[0, 255, 0],type='upper',swap='right_shoulder'),
    6:dict(name='right_shoulder',id=6,color=[255, 128, 0],type='upper',swap='left_shoulder'),
    7:dict(name='left_elbow',id=7,color=[0, 255, 0],type='upper',swap='right_elbow'),
    8:dict(name='right_elbow',id=8,color=[255, 128, 0],type='upper',swap='left_elbow'),
    9:dict(name='left_wrist',id=9,color=[0, 255, 0],type='upper',swap='right_wrist'),
    10:dict(name='right_wrist',id=10,color=[255, 128, 0],type='upper',swap='left_wrist'),
    11:dict(name='left_hip',id=11,color=[0, 255, 0],type='lower',swap='right_hip'),
    12:dict(name='right_hip',id=12,color=[255, 128, 0],type='lower',swap='left_hip'),
    13:dict(name='left_knee',id=13,color=[0, 255, 0],type='lower',swap='right_knee'),
    14:dict(name='right_knee',id=14,color=[255, 128, 0],type='lower',swap='left_knee'),
    15:dict(name='left_ankle',id=15,color=[0, 255, 0],type='lower',swap='right_ankle'),
    16:dict(name='right_ankle',id=16,color=[255, 128, 0],type='lower',swap='left_ankle'),
    17:dict(name='head_top',id=17,color=[51, 153, 255],type='upper',swap=''),
    18:dict(name='neck', id=18, color=[51, 153, 255], type='upper', swap='')
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
    
    img_path = 'LycorisRecoil (1).jpg'   # 将img_path替换给你自己的路径
    
    image = cv2.imread(img_path)

    
    # 使用模型别名创建推断器
    inferencer = MMPoseInferencer('td-reg_res152_8xb64-210e_mpii-256x256')
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
    
    cv2.imshow('Pose Estimation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 示例关键点索引
    point1_idx = 0
    point2_idx = 7
    point3_idx = 9

    # 计算角度
    angle = calculate_angle(keypoints, point1_idx, point2_idx, point3_idx)
    print("Angle between keypoints:", angle)
    
    score = 100
    
    # if calculate_angle( , , ) > 
