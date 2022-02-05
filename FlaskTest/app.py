from flask import Flask,render_template,Response
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

import torch
import torchvision.transforms as transforms

device = torch.device('cpu')
face_model = torch.load('C:/Users/admin/Desktop/FlaskTest/models/resnet50.pt', map_location=device)

face_model.eval()
class_names = ['female_active', 'female_clean', 'female_cute', 'female_elegant', 'female_natural', 'female_sexy', 
                'male_active', 'male_clean', 'male_cute', 'male_elegant', 'male_natural', 'male_sexy']
                
preprocess=transforms.Compose([ transforms.Resize(size=64),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])


# 질문 문구
font1 = ImageFont.truetype("C:/Users/admin/Desktop/FlaskTest/나눔손글씨 마고체.ttf",50)
font2 = ImageFont.truetype("C:/Users/admin/Desktop/FlaskTest/나눔손글씨 마고체.ttf",45) 
font3 = ImageFont.truetype("C:/Users/admin/Desktop/FlaskTest/나눔손글씨 마고체.ttf",38) 

app=Flask(__name__)
camera=cv2.VideoCapture(1)

Q_flag = 1
tmp = 0
cnt = 0
C_list = []

def generate_frames():
    while True:
        success,frame=camera.read()
        
        if not success:
            break
        
        else:
            # MediaPipe hands model
            mp_hands = mp.solutions.hands
            mp_facedetector = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils
            hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            img = frame
            img = cv2.resize(img, (1280, 820))
            img = cv2.flip(img, 1)

            img = cv2.rectangle(img, (420,200), (810,600), (0,255,0), 2)
            x, y = 425, 205
            w, h = 380, 390
            crop_img = img[y:y+h, x:x+w] #이미지 분류 모델 태울 얼굴위주이미지

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            location_x = 0
            location_y = 0
            
            # hand landmark detecting
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    location_x = int(res.landmark[8].x * img.shape[1])
                    location_y = int(res.landmark[8].x * img.shape[0])               
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(img, f'{(location_x,location_y)}', org=(int(res.landmark[8].x * img.shape[1]), int(res.landmark[8].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            else :
                pass
            
            global Q_flag, tmp, cnt
 
            if Q_flag == 1 :
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)   
                draw.text((30,30),"Q1. 나를 편안하게하고 기분 좋아지게 하는 향기를 골라주세요.",font=font1,fill=(0,0,0), stroke_width = 1)
                draw.text((30,100),"1. 풀, 나무 향    2. 과일, 꽃 향    3. 도시의 향기   4. 시원한 물 향",font=font2,fill=(0,0,0), stroke_width = 1)
                img = np.array(img)
                if location_x > 500 and location_y > 500: #조건 추가될 것
                    tmp = 1
                if tmp == 1:
                    cnt += 1
                if cnt == 20 :
                    tmp = 0
                    cnt = 0
                    
                    # 얼굴 분류 모델
                    with torch.no_grad():
                        crop_img = Image.fromarray(crop_img)
                        crop_img.convert('RGB')
                        inputs = preprocess(crop_img).unsqueeze(0)
                        outputs = face_model(inputs)
                        _, preds = torch.max(outputs, 1)   
                        C_list.append(class_names[preds])
                    
                    Q_flag = 2

            elif Q_flag == 2 :
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)   
                draw.text((30,30),"Q2. 상황 선택",font=font1,fill=(0,0,0), stroke_width = 1)
                draw.text((30,100),"1. 밝은 햇살이 들어오는 상쾌한 숲을 거닐 때     2. 울창한 숲 한가운데 서있을 때",font=font3,fill=(0,0,0), stroke_width = 1)
                draw.text((30,150),"3. 넓은 들판에 누워 낮잠 잘 때     4. 한적한 공간에서 티타임을 즐길 때",font=font3,fill=(0,0,0), stroke_width = 1)
                img = np.array(img)
                if location_x > 500 and location_y > 500: #조건 추가될 것
                    tmp = 1
                if tmp == 1:
                    cnt += 1
                if cnt == 20 :
                    tmp = 0
                    cnt = 0
                    
                    # 얼굴 분류 모델
                    with torch.no_grad():
                        crop_img = Image.fromarray(crop_img)
                        crop_img.convert('RGB')
                        inputs = preprocess(crop_img).unsqueeze(0)
                        outputs = face_model(inputs)
                        _, preds = torch.max(outputs, 1)   
                        C_list.append(class_names[preds])
                    
                    Q_flag = 3
                    
            elif Q_flag == 3 :
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)   
                draw.text((30,30),"Q3. 언제 쓰고 싶은 향수를 추천 받고 싶나요?",font=font1,fill=(0,0,0), stroke_width = 1)
                draw.text((30,100),"1. 봄/여름    2. 가을/겨울",font=font2,fill=(0,0,0), stroke_width = 1)
                img = np.array(img)
                if location_x > 500 and location_y > 500: #조건 추가될 것
                    tmp = 1
                if tmp == 1:
                    cnt += 1
                if cnt == 20 :
                    tmp = 0
                    cnt = 0
                    
                    # 얼굴 분류 모델
                    with torch.no_grad():
                        crop_img = Image.fromarray(crop_img)
                        crop_img.convert('RGB')
                        inputs = preprocess(crop_img).unsqueeze(0)
                        outputs = face_model(inputs)
                        _, preds = torch.max(outputs, 1)   
                        C_list.append(class_names[preds])
                    
                    Q_flag = 4

            elif Q_flag == 4 :
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)   
                draw.text((30,30),"THE END",font=font1,fill=(0,0,0), stroke_width = 1)
                img = np.array(img)    

            frame = img
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

