import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from new_2_Interpret_script_Main import main

from models.sequence_cmt import Seq_Cross_Transformer_Network  # as Seq_Cross_Transformer_Network
from models.sequence_cmt import Epoch_Cross_Transformer
from models.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward

import json
from ai.sync import sync_vivogpt


app = Flask(__name__)

# 获取app.py文件的绝对路径
basedir = os.path.abspath(os.path.dirname(__file__))

# 设置上传文件夹为app.py同级目录下的data文件夹
UPLOAD_FOLDER = os.path.join(basedir, 'data')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#登录页
@app.route('/')
def login():
    return render_template('login.html')

#脑电监测页
@app.route('/EEG-record', methods=['GET', 'POST'])
def EEG_record():
    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['file']
        # 如果文件存在且文件名不为空
        if file and file.filename != '':
            # 使用secure_filename确保文件名安全
            filename = secure_filename(file.filename)
            # 保存文件到指定的文件夹
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            main(filename)
            # return f'文件 {filename} 已保存到 {UPLOAD_FOLDER}'
    # 如果是GET请求，返回上传表单
    return render_template('EEG-record.html')


@app.route('/patient')
def patient():
    return render_template('patient.html')

@app.route('/sleep-report')
def sleep_report():
    content = sync_vivogpt()
    return render_template('sleep-report.html', content=content)

# 定义一个新的路由来提供 sleep_data.json 的内容
@app.route('/sleep-data')
def sleep_data():
    # 假设 sleep_data.json 在当前工作目录的 sleep_proportion 文件夹内
    with open('sleep_proportion/sleep_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0')
    app.run(debug=True)