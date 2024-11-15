import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything

if __name__ == '__main__':
    #创建 ArgumentParser 对象 parser，并添加多个命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str,default='./assets/examples/demo3.png')
    parser.add_argument('--outdir', type=str, default='./results/vil_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_false', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_false', help='do not apply colorful palette')
    args = parser.parse_args()

    #定义分割宽度、标题高度、字体样式、字体缩放和字体粗细，用于后续的图像拼接和标题添加。
    margin_width = 50
    caption_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    """ 通过DPT_DINOv2直接构建模型. """
    #加载模型 ，用DPT_DINOv2 初始化 depth_anything 模型
    depth_anything = DPT_DINOv2().to(DEVICE).eval()
    #加载预训练权重
    weights = torch.load("./checkpoints/depth_anything_vitb14.pth")
    depth_anything.load_state_dict(weights)
    #计算并输出模型的总参数数量，以百万（M）为单位。为了衡量模型的大小和潜在的计算成本
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

'''
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    encoder = 'vitb'  # or 'vitb', 'vits'
    depth_anything = DepthAnything(model_configs[encoder]).to(DEVICE).eval()
    depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_{encoder}14.pth'))
    # 计算并输出模型的总参数数量，以百万（M）为单位。为了衡量模型的大小和潜在的计算成本
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    """ 构建好模型深度估计模型"""
    #定义图像预处理步骤
    transform = Compose([
        # 调整图像大小为 518x518，保持宽高比，确保尺寸为 14 的倍数。
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        # 归一化图像颜色值，使其符合模型输入规范。
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #将图像准备为网络的输入格式（例如改变通道顺序）。
        PrepareForNet(),
    ])

    args.img_path='./assets/self_data/adv_imge.jpg'
    #加载图像文件列表，检查 img-path 是否为单个图像、文本文件或文件夹，分别读取其中的图像路径。
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()

    #创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    #读取图像
    for filename in tqdm(filenames):
        #使用 cv2.imread 读取图像，并将颜色格式从 BGR 转换为 RGB，同时归一化。
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        """获取图像高度和宽度，应用预处理步骤，并将图像转为 PyTorch 张量格式后发送到设备"""
        #返回原本图像的第一个和第二个维度（即高度和宽度），以便稍后调整生成的深度图与原图的大小一致。
        h, w = image.shape[:2]
        #应用预处理改变image形状
        image = transform({'image': image})['image']#（320，1024，3）-----（3，518，1652）
        #unsqueeze(0) 在第 0 维增加一个维度，将图像变为四维张量 [batch_size, channels, height, width]，适应模型输入格式。
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)#(1,3,518,1652)

        #在禁用梯度计算的上下文中，进行深度估计预测，降低内存消耗。
        with torch.no_grad():
            depth = depth_anything(image)#depth为(1,518,1652)未归一化

        """将深度图还原到原始图像大小，归一化并转换为 8 位整型的图像格式,适合图像保存"""
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]#(320,1024)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        #若设置 grayscale，生成灰度图像，否则将深度图应用伪彩色 COLORMAP_INFERNO。
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        #获取文件的名字，方便以后存储的时候保存名字
        filename = os.path.basename(filename)

        #若设置 pred-only，则保存深度图预测图像，文件名添加 _depth 后缀，否则就拼接图象。
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([raw_image, split_region, depth])
            
            caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + margin_width

            #在标题区域添加文字，显示“Raw image”和“Depth Anything”。
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
            final_result = cv2.vconcat([caption_space, combined_results])
            
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)
        