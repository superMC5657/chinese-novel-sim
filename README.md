
## 代码介绍
### algorithm
+ dtree: 句法依存树
+ idf: tfidf模型 训练与推理
+ sbert: sota 神经网络语义相似度模型
+ utils: 一些工具func

### data
+ datasets: 支持多个txt文档输入
+ de_hdt-ud-dev.conllu: 测试conllu

### static 
+ 输出图片保存位置

### templates
+ index.html 前端静态页面

### web_flask
+ 后端程序



## 安装与使用
python 必须为3.7版本 (ddparser依赖)
pip install -r requirements


所有文件的执行目录都是根目录
python web_flask.py