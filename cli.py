import logging
import json
from paddleocr import PaddleOCR

logging.disable(level=logging.DEBUG)
logging.disable(level=logging.WARNING)


def ocr_image(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = ocr.ocr(image_path, cls=True)
    return result


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1]
    result = ocr_image(image_path)
    print(json.dumps(result, ensure_ascii=False))  # 使用 json.dumps 转换为 JSON 字符串
