from flask import Flask, render_template, url_for, request, redirect, session, flash, Response, send_from_directory

# python基本ライブラリ
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib

# ファイル作成
import shutil
import datetime
import cv2
import os

# pytorch
import torch
from PIL import Image
# from function import generation

# パッケージのimport
import sys
import random
import time
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

from models.ssd_model import VOCDataset, DataTransform, Anno_xml2list, od_collate_fn
from models.ssd_model import SSD
from models.ssd_model import MultiBoxLoss
from models.ssd_predict_show import SSDPredictShow
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SECRET_KEY"] = "jfaogiehi2iw8jLD0ejJ"
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'} 

# 画像の保存ディレクトリを設定
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# カメラのステータスを追跡する変数
camera_active = False

@app.route('/', methods=['GET'])
def select_get():
    return render_template('select.html')

@app.route('/image_detect', methods=['GET'])
def image_get():
    return render_template('image_detect.html')

@app.route('/image_detect', methods=['POST'])
def select_post():
    
    # ファイルの作成
    img_dir = "static/images/"
    result_dir = "static/detected_images/"
    dir_list = [img_dir, result_dir]
    
    for item in dir_list:
        if os.path.exists(item):
            shutil.rmtree(item)
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 入力受け取り
    input_image = request.files['image']
    
    # Flaskの設定
    input_flg = True
    
    if not input_image:
        flash("画像を入力してください")
        input_flg = False
        
    if not input_flg:
        return redirect(url_for("select_get"))
    
    # 画像の変換
    stream = input_image.stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    
    # 画像の保存
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = img_dir + dt_now + ".jpg"
    result_path = result_dir + dt_now + ".jpg"
    cv2.imwrite(img_path, img)

    voc_classes = ['can', 'pet bottle']
    # SSDネットワークモデル
    ssd_cfg = {
        'num_classes': 3,  # 背景クラスを含めた合計クラス数
        'input_size': 300,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [21, 45, 99, 153, 207, 261],  # DBOXの大きさを決める
        'max_sizes': [45, 99, 153, 207, 261, 315],  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }
    net = SSD(phase="inference", cfg=ssd_cfg)
    # SSDの学習済みの重みを設定
    radio_p = request.form.get("radio")
    net_weights = torch.load('models/' + radio_p + '.pth', map_location={'cuda:0': 'cpu'})
    net.load_state_dict(net_weights)

    # 予測と、予測結果を画像で描画する
    ssd = SSDPredictShow(eval_categories=voc_classes, net=net)
    c_count, b_count = ssd.show(img_path, result_path, data_confidence_level=0.5)
    
    return render_template('image_detect.html',
                           content=img_path,
                           content2=result_path,
                           c_count=c_count,
                           b_count=b_count)

@app.route('/capture', methods=['GET'])
def photo():
    return render_template('capture.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # カメラからのキャプチャ
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            if cv2.waitKey(1) == ord('c'):
                camera = cv2.VideoCapture(0)
                success, frame = camera.read()
                if success:
                    # ファイル名をセキュアに生成
                    filename = secure_filename('capture.jpg')
                    img_dir = "static/images/"
                    filepath = img_dir + filename
                    cv2.imwrite(filepath, frame)  # 画像を保存
                    camera.release()
                    return redirect(url_for('uploaded_file', filename=filename))
                else:
                    return "画像のキャプチャに失敗しました"
    

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shoto', methods=['POST'])
def capture():
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    if success:
        # ファイル名をセキュアに生成
        filename = secure_filename('capture.jpg')
        img_dir = "static/test/"
        filepath = img_dir + filename
        cv2.imwrite(filepath, frame)  # 画像を保存
        camera.release()
        return redirect(url_for('uploaded_file', filename=filename))
    else:
        return "画像のキャプチャに失敗しました"

# アップロードされたファイルを表示
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('select.html', filename=filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)