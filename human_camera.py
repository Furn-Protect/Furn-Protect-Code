from mmpose.apis import MMPoseInferencer
import cv2
cap = cv2.VideoCapture(0)

# 创建MMPoseInferencer对象
inferencer = MMPoseInferencer('human')

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

def calculate_score(angle):
        if angle < 43.19649400346399 and angle>17.83552164924314:
            score =100
        elif angle >=43.19649400346399  and angle < 60:
            score =85
        elif angle < 17.83552164924314 and angle >0:
            score = 75
        else:
            score = 59
        return score


if __name__ == '__main__':

    while True:
        
        # 读取摄像头的图像
        ret, frame = cap.read()

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
        
        
        #********************************************#
        '''在这里写判断程序，如果计算出来角度不对直接breake'''
        
        

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()