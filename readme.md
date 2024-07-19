###

### 环境

1. python: 3.9.6
2. pip3 install paddlepaddle
3. pip3 install paddleocr
4. pip3 install requests
5. pip3 install scikit-learn
6. 激活虚拟空间，执行 python_path.py 查看 python 解释器的路径
   > /Users/xmly/Documents/shadow/ocr_cli/.venv/bin/python

### 解压

- ch_PP-OCRv4_det_server_infer.tar
- ch_PP-OCRv4_rec_server_infer.tar

### 使用

> https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/quickstart.md#211

### 下载模型

> https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md

1. 下载：ch_PP-OCRv4_server_det、 ch_PP-OCRv4_server_rec
2. 使用(https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/whl.md#31-%E4%BB%A3%E7%A0%81%E4%BD%BF%E7%94%A8)：
   ```
     ocr = PaddleOCR(
         # det_model_dir="/Users/xmly/Documents/shadow/ocr/ch_PP-OCRv4_det_server_infer",
         # rec_model_dir="/Users/xmly/Documents/shadow/ocr/ch_PP-OCRv4_rec_server_infer",
         use_angle_cls=True,
         lang="ch",
     )  # need to run only once to download and load model into memory
   ```

### 参数详解

> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/inference_args.md

### 过滤非水平的

### 精确文字

有些识别的区域包含了左右的图标
![alt text](result.jpg)

可以按照竖向的空隙，分割图片，然后进行逐字识别，然后将可信率差的剔除

- 竖向切割，竖向都是一个颜色的就进行切割
