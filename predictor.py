import torch
from segment_anything import sam_model_registry, SamPredictor

model_info = {
    'vit_b': 'sam_vit_b_01ec64.pth',
    'vit_l': 'sam_vit_l_0b3195.pth',
    'vit_h': 'sam_vit_h_4b8939.pth'
}


def get_predictor(model_type='vit_b', is_gpu=True):
    # 初始化SAM模型并转移到CUDA设备
    sam = sam_model_registry[model_type](checkpoint=model_info[model_type])
    if is_gpu:
        if not torch.cuda.is_available():
            print(f'cuda is not available!')
        _ = sam.to(device='cuda')
    # 创建SAM预测器
    predictor = SamPredictor(sam)
    return predictor
