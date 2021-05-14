# -*- coding: utf-8 -*-
# !@time: 2021/5/15 上午12:01
# !@author: superMC @email: 18758266469@163.com
# !@fileName: web_flask.py

import os
import time

from flask import Flask, render_template, Response, request, send_from_directory, make_response
from werkzeug.utils import secure_filename
from pyhanlp.main import main

app = Flask(__name__)
dir_path = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(dir_path, 'images/')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        global demo



@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    if request.method == 'GET':
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)





if __name__ == '__main__':
    app.run(host='192.168.2.2', debug=True, port=5020, use_reloader=True)