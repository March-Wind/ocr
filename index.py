import paddleocr
from paddleocr import PaddleOCR, draw_ocr
import logging

logging.disable(level=logging.DEBUG)
logging.disable(level=logging.WARNING)

# 初始化PaddleOCR，使用简体中文模型
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 读取图片
img_path = "/Users/xmly/Documents/shadow/img_seg/src/0_4.png"

# 进行OCR识别
# result格式说
# [
#     [
#         [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],   # 文本框的四个顶点坐标
#         ('识别出的文本字符串', 置信度分数)              # 识别出的文本及其置信度
#     ],
#     ...
# ]
result = ocr.ocr(img_path, cls=True)

# 显示结果
# for line in result:
#     print(line)

print(result)
