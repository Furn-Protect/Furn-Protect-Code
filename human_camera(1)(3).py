
from mmpose.apis import MMPoseInferencer
import cv2
cap = cv2.VideoCapture(0)
import numpy as np
# 创建MMPoseInferencer对象
inferencer = MMPoseInferencer('human')


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

def calculate_score(angle,angle1):
        if angle < 43.19649400346399 and angle>17.83552164924314:
            if angle1<181 and angle1>160:
                score=100
            else:
                score=95
        elif (angle >=43.19649400346399  and angle < 60) :
                score=85
        elif (angle < 17.83552164924314 and angle >0) :
            if angle1<160:
                score=70
            else:
                score=75

        else:
            score = 59
        return score


if __name__ == '__main__':

    while True:
        
        # 读取摄像头的图像
        ret, frame = cap.read()
        
        frame = cv2.imread('2442.png')

        # 使用MMPose提取每一帧图像的关键点
        result_generator = inferencer(frame, show=False)
        result = next(result_generator)
        predictions = result['predictions']
        dic = predictions[0][0]
        keypoints = dic['keypoints']

        # 实时显示关键点和摄像头拍摄到的画面
        for i, keypoint in enumerate(keypoints):
            x, y = keypoint[0], keypoint[1]
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i) + ' ' + name_list[i], (int(x) + 10, int(y) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Camera', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) == ord('q'):
            break
        point1_idx = 6
        point2_idx = 8
        point3_idx = 10
        point4_idx = 5
        point5_idx = 7
        point6_idx = 9

        #********************************************#
        '''在这里写判断程序，如果计算出来角度不对直接breake'''
        angle=calculate_angle(keypoints, point1_idx, point2_idx, point3_idx)
        angle1= calculate_angle(keypoints, point4_idx, point5_idx, point6_idx)
        score = calculate_score(angle,angle1)

        if score>61:
            print("Score:", score)
            if score==100:
                print('完美')
            else:
                print('提示：')
                if score==95:
                    print('左手臂没伸直')
                elif score==70:
                    print('右手臂抬得过低，左手臂没伸直')
                elif score==85:
                    print('右手臂抬得过高')
                elif score==75:
                    print('右手臂抬得过高')
                elif score==65:
                    print('右手臂抬得过低')
        else:
            print('你不在射箭！')
            
            
            
            
            # 设置文本参数
        text = f"Score: 100" + "  Perfect!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # 文本颜色，这里设置为白色
        thickness = 2  # 文本线条粗细

        # 获取文本框的大小
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # 设置文本位置
        text_position = (10, 10 + text_height)  # 左上角位置，这里设置间距为10像素

        # 在图像上绘制文本
        cv2.putText(frame, text, text_position, font, font_scale, color, thickness)

        # 保存图像
        cv2.imwrite("image_with_score-2.jpg", frame)

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()