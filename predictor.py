from segment_anything import sam_model_registry, SamPredictor

model_info = {
    'vit_b': 'sam_vit_b_01ec64.pth'
}


def get_predictor(model_type='vit_b', is_gpu=True):
    # 初始化SAM模型并转移到CUDA设备
    sam = sam_model_registry[model_type](checkpoint=model_info[model_type])
    if is_gpu:
        _ = sam.to(device='cuda')
    # 创建SAM预测器
    predictor = SamPredictor(sam)
    return predictor
