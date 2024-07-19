import cv2

# 读取图片
image = cv2.imread("/Users/xmly/Documents/shadow/img_seg/src/tools/2.png")

# 定义矩形的四个点
points = [(1045, 305), (1150, 305), (1150, 366), (1045, 366)]
# points = [(215, 306), (378, 306), (378, 366), (215, 366)]

# 绘制矩形
cv2.line(image, points[0], points[1], (0, 255, 0), 2)
cv2.line(image, points[1], points[2], (0, 255, 0), 2)
cv2.line(image, points[2], points[3], (0, 255, 0), 2)
cv2.line(image, points[3], points[0], (0, 255, 0), 2)

# 显示图片
cv2.imshow("Image with Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图片
cv2.imwrite("output_image.jpg", image)
