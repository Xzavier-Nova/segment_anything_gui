import cv2
import os
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import time

# 定义输入和输出目录以及是否使用裁剪模式
input_dir = 'input'
output_dir = 'output'
crop_mode = True

# 创建输出目录，如果已存在则不覆盖
os.makedirs(output_dir, exist_ok=True)

# 列出输入目录中所有后缀为.png、.jpg、.jpeg的文件
image_files = [f for f in os.listdir(input_dir) if
               f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'))]

# 初始化SAM模型并转移到CUDA设备
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
_ = sam.to(device="cuda")
# 创建SAM预测器
predictor = SamPredictor(sam)

# 初始化输入点、标签列表和停止标志
input_point = []
input_label = []
input_stop = False

# 定义颜色变化间隔和上一次颜色变化时间
color_change_interval = 0.5  # 颜色变化间隔，单位：秒
last_color_change_time = time.time()
# 初始化当前颜色
current_color = tuple(np.random.randint(0, 256, 3).tolist())


# 鼠标点击事件处理函数
def mouse_click(event, x, y, flags, param):
    global input_point, input_label, input_stop, scale_factor
    if not input_stop:
        # 将点击坐标反映射到原始图像上
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)
        # 左键点击表示选择感兴趣的点
        if event == cv2.EVENT_LBUTTONDOWN:
            input_point.append([orig_x, orig_y])
            input_label.append(1)
        # 右键点击表示选择不感兴趣的点
        elif event == cv2.EVENT_RBUTTONDOWN:
            input_point.append([orig_x, orig_y])
            input_label.append(0)
    else:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            print('此时不能添加点,按w退出mask选择模式')


# 应用遮罩到图像上，可选择是否使用alpha通道
def apply_mask(image, mask, alpha_channel=True):
    if alpha_channel:
        alpha = np.zeros_like(image[..., 0])
        alpha[mask == 1] = 255
        image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))
    else:
        image = np.where(mask[..., None] == 1, image, 0)
    return image

# 为图像的指定区域应用颜色遮罩
def apply_color_mask(image, mask, color, color_dark=0.5):
    # 根据遮罩应用颜色
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - color_dark) + color_dark * color[c], image[:, :, c])
    return image

# 生成下一个可用的文件名
def get_next_filename(base_path, filename):
    name, ext = os.path.splitext(filename)
    for i in range(1, 101):
        new_name = f"{name}_{i}{ext}"
        if not os.path.exists(os.path.join(base_path, new_name)):
            return new_name
    return None

# 保存带遮罩的图像，根据裁剪模式处理图像
def save_masked_image(image, mask, output_dir, filename, crop_mode_):
    if crop_mode_:
        y, x = np.where(mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        masked_image = apply_mask(cropped_image, cropped_mask)
    else:
        masked_image = apply_mask(image, mask)
    filename = filename[:filename.rfind('.')] + '.png'
    new_filename = get_next_filename(output_dir, filename)
    if new_filename:
        if masked_image.shape[-1] == 4:
            cv2.imwrite(os.path.join(output_dir, new_filename), masked_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(os.path.join(output_dir, new_filename), masked_image)
        print(f"Saved as {new_filename}")
    else:
        print("Could not save the image. Too many variations exist.")

# 将图像调整到指定的最大宽度和高度，同时保持宽高比
def resize_image(image, max_side_length=1024):
    height, width = image.shape[:2]
    scaling_factor = min(max_side_length / max(height, width), 1)  # 防止放大图像
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scaling_factor

# 初始化当前图像索引
current_index = 0

# 创建一个名为"image"的窗口，设置为不可调整大小
cv2.namedWindow("image")
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# 设置鼠标回调函数
cv2.setMouseCallback("image", mouse_click)

# 主循环
while True:
    # 获取当前索引处的文件名
    filename = image_files[current_index]
    # 读取原始图像
    image_orign = cv2.imread(os.path.join(input_dir, filename))
    # 将图像等比例缩放至最长边为1024像素
    image_orign, scale_factor = resize_image(image_orign)
    # 复制原始图像以用于裁剪
    image_crop = image_orign.copy()
    # 将原始图像转换为RGB格式
    image = cv2.cvtColor(image_orign.copy(), cv2.COLOR_BGR2RGB)
    # 初始化选定的掩码和逻辑输入
    selected_mask = None
    logit_input = None

    while True:
        # 设置退出标志
        input_stop = False
        # 组合显示信息
        display_info = f'{filename} | save(s) | predict(w) | next(d) | previous(a) | clear(space) | cancel(q)'
        # 设置窗口标题为显示信息
        cv2.setWindowTitle("image", display_info)

        # 复制原始图像以用于显示
        image_display = image_orign.copy()

        # 在图像上绘制已选择的点和标签
        for point, label in zip(input_point, input_label):
            display_point = (int(point[0] * scale_factor), int(point[1] * scale_factor))
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(image_display, display_point, 5, color, -1)

        # 如果已选择掩码，则应用颜色并显示
        if selected_mask is not None:
            current_time = time.time()
            if current_time - last_color_change_time > color_change_interval:
                current_color = tuple(np.random.randint(0, 256, 3).tolist())
                last_color_change_time = current_time
            selected_image = apply_color_mask(image_display, selected_mask, current_color)

        # 显示处理后的图像
        cv2.imshow("image", image_display)
        # 等待按键事件
        key = cv2.waitKey(1)

        # 处理按键事件
        if key == ord(" "):
            input_point = []
            input_label = []
            selected_mask = None
            logit_input = None
        elif key == ord("w"):
            input_stop = True
            if len(input_point) > 0 and len(input_label) > 0:
                predictor.set_image(image)
                input_point_np = np.array(input_point)

                # 进行预测并获取掩码、分数和逻辑
                masks, scores, logits = predictor.predict(
                    point_coords=input_point_np,
                    point_labels=np.array(input_label),
                    mask_input=logit_input[None, :, :] if logit_input is not None else None,
                    multimask_output=True,
                )

                # 遍历预测的掩码，允许用户选择和编辑
                mask_idx = 0
                num_masks = len(masks)
                while True:
                    current_time = time.time()
                    if current_time - last_color_change_time > color_change_interval:
                        current_color = tuple(np.random.randint(0, 256, 3).tolist())
                        last_color_change_time = current_time
                    image_select = image_orign.copy()
                    selected_mask = masks[mask_idx]
                    selected_image = apply_color_mask(image_select, selected_mask, current_color)
                    mask_info = f'total: {num_masks} | index: {mask_idx} | score: {scores[mask_idx]:.2f} | confirm(w) | next(d) | previous(a) | cancel(q) | save(s)'
                    cv2.setWindowTitle("image", mask_info)

                    cv2.imshow("image", selected_image)

                    key = cv2.waitKey(10)
                    # 处理按键事件，包括删除点、保存掩码、选择下一掩码等
                    if key == ord('q') and len(input_point) > 0:
                        input_point.pop(-1)
                        input_label.pop(-1)
                    elif key == ord('s'):
                        save_masked_image(image_crop, selected_mask, output_dir, filename, crop_mode_=crop_mode)
                    elif key == ord('a'):
                        if mask_idx > 0:
                            mask_idx -= 1
                        else:
                            mask_idx = num_masks - 1
                    elif key == ord('d'):
                        if mask_idx < num_masks - 1:
                            mask_idx += 1
                        else:
                            mask_idx = 0
                    elif key == ord('w'):
                        break
                    elif key == ord(" "):
                        input_point = []
                        input_label = []
                        selected_mask = None
                        logit_input = None
                        break
                logit_input = logits[mask_idx, :, :]
                print('max score:', np.argmax(scores), ' select:', mask_idx)

        # 处理其他按键事件，如前进、后退、退出等
        elif key == ord('a'):
            current_index = max(0, current_index - 1)
            input_point = []
            input_label = []
            break
        elif key == ord('d'):
            current_index = min(len(image_files) - 1, current_index + 1)
            input_point = []
            input_label = []
            break
        elif key == 27:
            break
        elif key == ord('q') and len(input_point) > 0:
            input_point.pop(-1)
            input_label.pop(-1)
        elif key == ord('s') and selected_mask is not None:
            save_masked_image(image_crop, selected_mask, output_dir, filename, crop_mode_=crop_mode)

    # 如果按下ESC键，则退出循环
    if key == 27:
        break

# 释放窗口资源
cv2.destroyAllWindows()