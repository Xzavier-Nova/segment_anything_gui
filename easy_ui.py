import cv2

# 读取图像
image = cv2.imread('origin.webp')
original_height, original_width = image.shape[:2]

# 创建全局变量来存储点击位置和缩放比例
point_loc = []
point_label = []

scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global scale, point_loc, point_label

    # 获取窗口大小
    window_width = cv2.getWindowImageRect('Image Window')[2]
    window_height = cv2.getWindowImageRect('Image Window')[3]

    # 计算缩放比例
    scale_width = window_width / original_width
    scale_height = window_height / original_height
    scale = min(scale_width, scale_height)

    # 计算图像在窗口中的位置
    img_display_width = int(original_width * scale)
    img_display_height = int(original_height * scale)
    x_offset = (window_width - img_display_width) // 2
    y_offset = (window_height - img_display_height) // 2

    if event == cv2.EVENT_LBUTTONDOWN:
        # 计算在原图中的位置
        original_x = int((x - x_offset) / scale)
        original_y = int((y - y_offset) / scale)
        point_loc.append((original_x, original_y))
        point_label.append(1)
        print(f"Left Click at: {original_x}, {original_y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 计算在原图中的位置
        original_x = int((x - x_offset) / scale)
        original_y = int((y - y_offset) / scale)
        point_loc.append((original_x, original_y))
        point_label.append(0)
        print(f"Right Click at: {original_x}, {original_y}")


# 创建一个窗口
cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image Window', mouse_callback)

# 记录窗口大小的初始值
last_window_width = cv2.getWindowImageRect('Image Window')[2]
last_window_height = cv2.getWindowImageRect('Image Window')[3]

while True:
    # 获取当前窗口大小
    window_width = cv2.getWindowImageRect('Image Window')[2]
    window_height = cv2.getWindowImageRect('Image Window')[3]

    # 仅当窗口大小改变时重新计算缩放图像
    if window_width != last_window_width or window_height != last_window_height:
        last_window_width = window_width
        last_window_height = window_height

        # 计算缩放比例
        scale_width = window_width / original_width
        scale_height = window_height / original_height
        scale = min(scale_width, scale_height)

    # 缩放图像
    img_display = cv2.resize(image, (int(original_width * scale), int(original_height * scale)))

    # 绘制点击点
    for ind in range(len(point_loc)):
        point = point_loc[ind]
        label = point_label[ind]
        if label == 1:
            cv2.circle(img_display, (int(point[0] * scale), int(point[1] * scale)), 5, (0, 255, 0), -1)
        elif label == 0:
            cv2.circle(img_display, (int(point[0] * scale), int(point[1] * scale)), 5, (0, 0, 255), -1)

    # 计算边界值，确保非负
    top = max((window_height - img_display.shape[0]) // 2, 0)
    bottom = max((window_height - img_display.shape[0]) // 2, 0)
    left = max((window_width - img_display.shape[1]) // 2, 0)
    right = max((window_width - img_display.shape[1]) // 2, 0)

    # 在窗口居中显示图像
    canvas = cv2.copyMakeBorder(img_display, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 显示图像
    cv2.imshow('Image Window', canvas)

    key = cv2.waitKey(1)
    # 按下'Esc'键退出
    if key == 27:
        break
    elif key == ord("q") and len(point_loc) > 0:
        point_loc.pop(-1)
        point_label.pop(-1)

# 关闭所有窗口
cv2.destroyAllWindows()
