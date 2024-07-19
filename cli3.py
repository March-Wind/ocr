import subprocess
import json
import copy

from typing import List, Tuple, Dict, Any
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from collections import Counter
from cv2.typing import MatLike
import os
import logging


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)

logging.disable(level=logging.DEBUG)
logging.disable(level=logging.WARNING)
### 经过左右过滤的文本区域，更加精确


def filter_credibility(ocr_results: List[List[List[Any]]]) -> List[List[List[Any]]]:
    """
    过滤掉可信度小于0.9的文本框。

    :param ocr_results: 包含文本框和对应文本信息的OCR结果列表。
    :return: 过滤后的OCR结果列表。
    """
    filtered_results: List[List[List[Any]]] = []
    for line in ocr_results:
        filtered_line = [
            [box, textInfo] for box, textInfo in line if textInfo[1] >= 0.9
        ]
        filtered_results.append(filtered_line)
    return filtered_results


def filter_horizontal_text(ocr_results: List[List[List[Any]]]) -> List[List[List[Any]]]:
    """
    过滤掉非水平的文本框。

    :param ocr_results: 包含文本框和对应文本的OCR结果列表。
    :return: 过滤后的水平文本框OCR结果列表。
    """
    horizontal_results: List[List[List[Any]]] = []
    for line in ocr_results:
        horizontal_line = [
            [box, textInfo]
            for box, textInfo in line
            if abs(box[0][1] - box[1][1]) < 5 and abs(box[2][1] - box[3][1]) < 5
        ]
        horizontal_results.append(horizontal_line)
    return horizontal_results


# 裁剪文字区域
def crop_text_region(
    image: MatLike, points: List[Tuple[int, int]]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]
    x_min = int(min(x1, x2, x3, x4))
    x_max = int(max(x1, x2, x3, x4))
    y_min = int(min(y1, y2, y3, y4))
    y_max = int(max(y1, y2, y3, y4))
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def get_top_colors(
    cropped_img: np.ndarray,
    top_n: int = 3,
    exclude_colors: List[Tuple[int, int, int]] = [],
) -> List[Tuple[Tuple[int, int, int], int]]:
    """
    统计 cropped_img 中的所有颜色，并找出最多的 top_n 个颜色，排除指定的颜色。

    :param cropped_img: 裁剪后的图像。
    :param top_n: 要找出的最多颜色数，默认为3。
    :param exclude_colors: 要排除的颜色列表，默认为白色和接近白色。
    :return: 包含最多的 top_n 个颜色及其出现次数的列表。
    """
    # 将图像从 BGR 转换为 RGB
    cropped_img_rgb = cropped_img[..., ::-1]
    # 将图像数据展平为二维数组，每行表示一个像素的颜色值
    data = np.reshape(cropped_img_rgb, (-1, 3))
    # 统计每个颜色出现的次数
    counter = Counter(map(tuple, data))
    # 过滤掉要排除的颜色
    for color in exclude_colors:
        counter.pop(color, None)
    # 找出出现次数最多的 top_n 个颜色
    top_colors = counter.most_common(top_n)
    # 将 BGR 格式转换为 RGB 格式
    # top_colors = [
    #     ((color[2], color[1], color[0]), count) for color, count in top_colors
    # ]
    return top_colors


# 计算颜色直方图并返回主色
def get_dominant_color(cropped_img: np.ndarray) -> Tuple[int, int, int]:
    data = np.reshape(cropped_img, (-1, 3))
    counter = Counter(map(tuple, data))
    most_common_color = counter.most_common(1)[0][0]
    # 将BGR格式转换为RGB格式
    return (most_common_color[2], most_common_color[1], most_common_color[0])


def get_dominant_color(image: np.ndarray) -> Tuple[int, int, int]:
    # 计算颜色的直方图，并获取出现次数最多的颜色
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color = palette[np.argmax(counts)]
    return tuple(map(int, dominant_color))


# 绘制矩形块并统计竖向同颜色的矩形块
# 这些矩形块，期望是文字间隙，但是也会被文字竖向的笔画干扰，暂时忽略不计
# 这样得出来的颜色，就算是背景色
def draw_and_count_colors(
    cropped_img: np.ndarray, bounds: Tuple[int, int, int, int], image: np.ndarray
) -> Tuple[Counter, List[Tuple[int, int, int, int]]]:
    # 提取边界框的坐标
    x_min, y_min, x_max, y_max = bounds
    # 获取裁剪区域的高、宽和通道数
    height, width, _ = cropped_img.shape

    # 用于存储连续垂直颜色块的列索引
    vertical_blocks: List[int] = []
    # 用于存储前一列的颜色
    prev_col_colors: np.ndarray = None
    # 用于统计颜色出现次数
    color_counter: Counter = Counter()
    # 用于存储所有矩形区域
    rectangles: List[Tuple[int, int, int, int]] = []

    # 遍历裁剪图像的每一列
    for col in range(width):
        # 获取当前列的所有颜色
        column_colors = cropped_img[:, col, :]

        # 判断当前列是否为单一颜色列
        if np.all(column_colors == column_colors[0]):
            # 如果是单一颜色列，并且与前一列颜色相同，继续收集垂直块
            if prev_col_colors is None or np.all(column_colors[0] == prev_col_colors):
                vertical_blocks.append(col)
                prev_col_colors = column_colors[0]
            else:
                # 如果当前单一颜色列与前一列不同，处理之前收集的垂直块
                if vertical_blocks:
                    start_col = vertical_blocks[0]
                    end_col = vertical_blocks[-1]
                    # 在原图上绘制矩形框
                    # cv2.rectangle(
                    #     image,
                    #     (x_min + start_col, y_min),
                    #     (x_min + end_col, y_max),
                    #     (0, 255, 0),
                    #     2,
                    # )
                    # 存储矩形区域
                    rectangles.append(
                        (x_min + start_col, y_min, x_min + end_col, y_max)
                    )
                    # 裁剪当前垂直块图像
                    block_img = cropped_img[:, start_col : end_col + 1]
                    # 获取垂直块的主色
                    dominant_color = get_dominant_color(block_img)
                    # 更新颜色计数器
                    color_counter[dominant_color] += (end_col - start_col + 1) * height
                    vertical_blocks = []
                prev_col_colors = column_colors[0]
                vertical_blocks.append(col)
        else:
            # 当前列不是单一颜色列，处理之前收集的垂直块
            if vertical_blocks:
                start_col = vertical_blocks[0]
                end_col = vertical_blocks[-1]
                # 在原图上绘制矩形框
                # cv2.rectangle(
                #     image,
                #     (x_min + start_col, y_min),
                #     (x_min + end_col, y_max),
                #     (0, 255, 0),
                #     2,
                # )
                # 存储矩形区域
                rectangles.append((x_min + start_col, y_min, x_min + end_col, y_max))
                # 裁剪当前垂直块图像
                block_img = cropped_img[:, start_col : end_col + 1]
                # 获取垂直块的主色
                dominant_color = get_dominant_color(block_img)
                # 更新颜色计数器
                color_counter[dominant_color] += (end_col - start_col + 1) * height
                vertical_blocks = []
            prev_col_colors = None

    # 处理最后收集的垂直块
    if vertical_blocks:
        start_col = vertical_blocks[0]
        end_col = vertical_blocks[-1]
        # 在原图上绘制矩形框
        # cv2.rectangle(
        #     image, (x_min + start_col, y_min), (x_min + end_col, y_max), (0, 255, 0), 2
        # )
        # 存储矩形区域
        rectangles.append((x_min + start_col, y_min, x_min + end_col, y_max))
        # 裁剪当前垂直块图像
        block_img = cropped_img[:, start_col : end_col + 1]
        # 获取垂直块的主色
        dominant_color = get_dominant_color(block_img)
        # 更新颜色计数器
        color_counter[dominant_color] += (end_col - start_col + 1) * height

    # 返回颜色计数器和矩形区域列表
    return color_counter, rectangles


# 读取图像并进行 OCR 识别
def perform_ocr(
    image: MatLike,
) -> List[List[List[Any]]]:
    # 初始化 OCR 模型
    ocr = PaddleOCR(
        # det_model_dir="./ch_PP-OCRv4_det_server_infer",
        # rec_model_dir="./ch_PP-OCRv4_rec_server_infer",
        det_model_dir=f"{current_directory}/ch_PP-OCRv4_det_server_infer",
        rec_model_dir=f"{current_directory}/ch_PP-OCRv4_rec_server_infer",
        use_angle_cls=True,
        lang="ch",
    )
    return ocr.ocr(image, cls=False)


def two_part_area(
    image: np.ndarray, points: List[Tuple[int, int]], rect: Tuple[int, int, int, int]
) -> List[Tuple[np.ndarray, List[Tuple[int, int]]]]:
    """
    裁剪图像中指定区域外的部分。

    根据给定的四个点和矩形区域，裁剪图像中更长的外部区域（左或右），并返回裁剪后的图像及其矩形区域的四个顶点。

    :param image: 输入的图像。
    :param points: 包含四个坐标点的列表，这些点定义了一个四边形区域。
    :param rect: 包含四个整数的元组，表示矩形的左上角和右下角的坐标。
    :return: 裁剪后的图像及其矩形区域的四个顶点。
    """
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]
    x_min = int(min(x1, x2, x3, x4))
    x_max = int(max(x1, x2, x3, x4))
    y_min = int(min(y1, y2, y3, y4))
    y_max = int(max(y1, y2, y3, y4))
    left_rect_x1, _, right_rect_x2, _ = rect

    # # 计算左侧区域的长度
    # left_region_length = left_rect_x1 - x_min
    # # 计算右侧区域的长度
    # right_region_length = x_max - right_rect_x2
    # cropped_image = None
    # cropped_points = []
    # 裁剪左侧区域的图片
    cropped_image_left = image[y_min:y_max, int(x_min) : int(left_rect_x1)]
    cropped_points_left = [
        (x_min, y_min),
        (left_rect_x1, y_min),
        (left_rect_x1, y_max),
        (x_min, y_max),
    ]
    # 裁剪右侧区域的图片
    cropped_image_right = image[y_min:y_max, int(right_rect_x2) : int(x_max)]
    cropped_points_right = [
        (right_rect_x2, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (right_rect_x2, y_max),
    ]

    return [
        (cropped_image_left, cropped_points_left),
        (cropped_image_right, cropped_points_right),
    ]


def is_valid_image_part(image_part: np.ndarray) -> bool:
    """判断图像部分是否有效."""
    return image_part.size > 0 and image_part.shape[1] > 0 and image_part.shape[0] > 0


def search_shortest_text_consistent(
    image: np.ndarray,
    # 被切割的区域
    whole_area_points: List[Tuple[int, int]],
    # 分割器，是个矩形
    rect: Tuple[int, int, int, int],
    # 所有的分割器
    rectangles: List[Tuple[int, int, int, int]],
    origin_text: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]]] or None:
    # 找到当前rect的索引
    index = rectangles.index(rect)
    # 以当前rect为分界点分成两段
    left_rectangles = rectangles[:index]
    right_rectangles = rectangles[index + 1 :]

    left_image_part, right_image_part = two_part_area(image, whole_area_points, rect)
    # if origin_text == "用户" or origin_text == "已收获":
    #     # 显示左侧图像部分
    #     if is_valid_image_part(left_image_part[0]):
    #         cv2.imshow("Left Image Part", left_image_part[0])
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    #     # 显示右侧图像部分
    #     if is_valid_image_part(right_image_part[0]):
    #         cv2.imshow("Right Image Part", right_image_part[0])
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    if not is_valid_image_part(left_image_part[0]):
        # left_part_ocr = [[["", ("")]]]
        left_text = ""
    else:
        left_part_ocr = perform_ocr(left_image_part[0])
        if not left_part_ocr[0]:
            left_text = ""
        else:
            left_text = left_part_ocr[0][0][1][0]

    # 检查right_image_part[0]是否为空数组
    if not is_valid_image_part(right_image_part[0]):
        # right_part_ocr = [[["", ("")]]]
        right_text = ""
    else:
        right_part_ocr = perform_ocr(right_image_part[0])
        if not right_part_ocr[0]:
            right_text = ""
        else:
            right_text = right_part_ocr[0][0][1][0]

    # left_text = left_part_ocr[0][0][1][0]
    # right_text = right_part_ocr[0][0][1][0]

    if left_text == origin_text:
        if len(left_rectangles) == 0:
            return left_image_part
        # 这种写法是丢掉右边
        result = search_shortest_text_consistent(
            image,
            left_image_part[1],
            left_rectangles[-1],
            left_rectangles,
            origin_text,
        )
        if result:
            return result
        else:
            result = search_shortest_text_consistent(
                image,
                left_image_part[1],
                left_rectangles[len(left_rectangles) - 1],
                left_rectangles,
                origin_text,
            )
            if result:
                return result
            else:
                return left_image_part

    if right_text == origin_text:
        if len(right_rectangles) == 0:
            return right_image_part
        # 这种写法是丢掉左边
        result = search_shortest_text_consistent(
            image,
            right_image_part[1],
            right_rectangles[0],
            right_rectangles,
            origin_text,
        )
        if result:
            return result
        else:
            # 这种写法是丢掉右边
            result = search_shortest_text_consistent(
                image,
                right_image_part[1],
                right_rectangles[len(right_rectangles) - 1],
                right_rectangles,
                origin_text,
            )
            if result:
                return result
            else:
                return right_image_part

    return None


# 找出每个文字区域中颜色最多的颜色
def process_ocr_results(
    image: MatLike,
    ocr_results: List[List[List[Any]]],
    _image: MatLike,
) -> None:
    for line in ocr_results:
        for word_info in line:
            points = word_info[0]
            cropped_img, bounds = crop_text_region(image, points)
            color_counter, rectangles = draw_and_count_colors(
                cropped_img, bounds, _image
            )
            # 重新绘制矩形块
            # for rect in rectangles:
            #     # for x1, y1, x2, y2 in rectangles:
            #     #     cv2.rectangle(_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #     cropped_image2, cropped_image2_points = two_part_area(image, points, rect)
            #     # # cv2.imshow("Cropped Image", cropped_image2)
            #     # # cv2.waitKey(0)  # 等待用户按键
            #     # # cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            #     part_ocr = perform_ocr(cropped_image2)
            #     originText = word_info[1]

            #     print(part_ocr)
            if rectangles:
                shotest = search_shortest_text_consistent(
                    image, points, rectangles[0], rectangles, word_info[1][0]
                )
                if shotest:
                    word_info[0] = shotest[1]
                    # print(word_info[1][0], shotest[1])
            # 确保 word_info 至少有 3 个元素
            while len(word_info) < 3:
                word_info.append(None)

            if color_counter:
                # color_counter.most_common(1)是最多的一项
                # color_counter.most_common(1)就是[((255,255,255), 5)]
                # color_counter.most_common(1)[0][0])就是(255,255,255)
                bg = color_counter.most_common(1)[0][0]
                text_color = get_top_colors(cropped_img, 1, [bg])
                word_info[2] = {"bg": bg, "text_color": text_color}


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64, np.uint8)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    import sys

    img_path = sys.argv[1]
    image = cv2.imread(img_path)
    ocr_results = perform_ocr(image)
    filter_ocr_results = copy.deepcopy(ocr_results)
    filter_ocr_results = filter_credibility(filter_ocr_results)
    filter_ocr_results = filter_horizontal_text(filter_ocr_results)

    _image = image.copy()
    _process_ocr_results = copy.deepcopy(filter_ocr_results)
    process_ocr_results(image, _process_ocr_results, _image)
    # 过滤掉第三项是None的数据
    _process_ocr_results = [
        item for item in _process_ocr_results[0] if item[2] is not None
    ]
    ocr_results_serializable = convert_to_serializable(_process_ocr_results)
    print(json.dumps(ocr_results_serializable, ensure_ascii=False))
