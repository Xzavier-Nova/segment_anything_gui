import time

import cv2
import numpy as np

from predictor import get_predictor


def show_ui(image, model_type='vit_b', is_gpu=False):
    original_height, original_width = image.shape[:2]

    model_type = model_type
    predictor = get_predictor(model_type=model_type, is_gpu=is_gpu)  # segment_anything predictor

    # 创建全局变量来存储点击位置和缩放比例
    point_loc = []
    point_label = []
    input_stop = False

    scale = 1.0

    selected_mask = None  # 全局保存一个mask
    logit_input = None

    color_change_interval = 0.5  # 定义颜色更换频率
    last_color_change_time = time.time()
    # 初始化当前颜色
    current_color = tuple(np.random.randint(0, 256, 3).tolist())

    def mouse_callback(event, x, y, flags, param):
        if not input_stop:
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

    def apply_color_mask(image, mask, color, color_dark=0.5):
        # 根据遮罩应用颜色
        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - color_dark) + color_dark * color[c],
                                      image[:, :, c])
        return image

    # 应用遮罩到图像上，可选择是否使用alpha通道
    def create_mask(image, mask, alpha_channel=True):
        if alpha_channel:
            alpha = np.zeros_like(image[..., 0])
            alpha[mask == 1] = 255
            image_ = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))
        else:
            image_ = np.where(mask[..., None] == 1, image, 0)
        return image_

    def get_masked_image(image, mask, crop_mode_=False):
        if crop_mode_:
            y, x = np.where(mask)
            y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
            cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
            cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
            masked_image = create_mask(cropped_image, cropped_mask)
        else:
            masked_image = create_mask(image, mask)
        return masked_image

    # 创建一个窗口
    cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image Window', mouse_callback)

    # 记录窗口大小的初始值
    last_window_width = cv2.getWindowImageRect('Image Window')[2]
    last_window_height = cv2.getWindowImageRect('Image Window')[3]

    while True:
        input_stop = False
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

        selected_image = None
        # 如果已选择掩码，则应用颜色并显示
        if selected_mask is not None:
            current_time = time.time()
            if current_time - last_color_change_time > color_change_interval:
                current_color = tuple(np.random.randint(0, 256, 3).tolist())
                last_color_change_time = current_time
            selected_image = apply_color_mask(image.copy(), selected_mask, current_color)
        if selected_image is not None:
            img_display = cv2.resize(selected_image, (int(original_width * scale), int(original_height * scale)))
        else:
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
        canvas = cv2.copyMakeBorder(img_display, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # 显示图像
        cv2.imshow('Image Window', canvas)

        key = cv2.waitKey(1)
        # 按下'Esc'键退出
        if key == 27:
            break
        elif key == ord('q') and len(point_loc) > 0:
            point_loc.pop(-1)
            point_label.pop(-1)
            print("Removed last point")
        elif key == ord('w'):
            input_stop = True  # 此时禁止添加点
            if point_loc and point_label:
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    point_coords=np.array(point_loc),
                    point_labels=np.array(point_label),
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

                    image_select = image.copy()
                    selected_mask = masks[mask_idx]
                    selected_image = apply_color_mask(image_select, selected_mask, current_color)
                    selected_image = cv2.resize(selected_image,
                                                (int(original_width * scale), int(original_height * scale)))

                    mask_info = f'total: {num_masks} | index: {mask_idx} | score: {scores[mask_idx]:.2f} | confirm(w) | next(d) | previous(a) | cancel(q) | save(s)'
                    cv2.setWindowTitle('Image Window', mask_info)

                    cv2.imshow('Image Window', selected_image)

                    key = cv2.waitKey(10)
                    # 处理按键事件，包括删除点、保存掩码、选择下一掩码等
                    if key == ord('q') and len(point_loc) > 0:
                        point_loc.pop(-1)
                        point_label.pop(-1)
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
                        point_loc = []
                        point_label = []
                        selected_mask = None
                        logit_input = None
                        break


    # 关闭所有窗口
    cv2.destroyAllWindows()
    return get_masked_image(image, selected_mask)


if __name__ == '__main__':
    src_image = cv2.imread('origin.webp')
    masked_image = show_ui(src_image, model_type='vit_h', is_gpu=True)
    if masked_image.shape[-1] == 4:
        cv2.imwrite('./predicted_img.png', masked_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite('./predicted_img.png', masked_image)
