import os
import cv2
import PySimpleGUI as sg
import numpy as np

sg.popup("설치 성공")
print(dir(sg))

# 설정
INPUT_DIR = './image/hb'         # 원본 이미지 위치
OUTPUT_DIR = './image/labeled'       # 저장 경로
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 이미지 리스트 불러오기
file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')]

# GUI 구성
layout = [
    [sg.Text("이미지 선택:"), sg.Listbox(values=file_list, size=(30, 20), key='-FILE-', enable_events=True)],
    [sg.Image(key='-IMAGE-')],
    [sg.Text("문자 입력 (4자):"), sg.InputText(key='-LABEL-', size=(10,1))],
    [sg.Button("저장"), sg.Button("종료")]
]

window = sg.Window("간단 캡차 라벨링 툴", layout, finalize=True)

current_img = None
current_filename = None

def show_image(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (200, 50))  # GUI용 축소
    img_bytes = cv2.imencode('.png', resized)[1].tobytes()
    return img_bytes

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, '종료'):
        break

    if event == '-FILE-':
        current_filename = values['-FILE-'][0]
        current_img = cv2.imread(os.path.join(INPUT_DIR, current_filename), cv2.IMREAD_GRAYSCALE)
        window['-IMAGE-'].update(data=show_image(os.path.join(INPUT_DIR, current_filename)))

    if event == '저장':
        label = values['-LABEL-'].strip()
        if current_img is None or not current_filename:
            sg.popup("이미지를 먼저 선택하세요.")
            continue
        if len(label) != 4:
            sg.popup("정확히 4자 입력해주세요.")
            continue

        h, w = current_img.shape
        char_w = w // 4

        # 문자 4개 슬라이싱
        chars = []
        for i in range(4):
            x1 = i * char_w
            x2 = (i + 1) * char_w
            char_img = current_img[0:h, x1:x2]
            chars.append(char_img)

        # 이어붙이기 (선택사항, 저장 포맷에 따라)
        final_img = cv2.hconcat(chars)

        # 저장
        save_name = f"{current_filename[:-4]}_{label}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), final_img)
        sg.popup(f"저장 완료: {save_name}")
        window['-LABEL-'].update('')

window.close()

