import gc
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import torch.nn as nn
from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything
from depth_anything.blocks import FeatureFusionBlock, _make_scratch
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesUV,
    materials
)
from data_loader_mde import MyDataset
import sys




# 这里是系统有多卡的时候的配置代码，我们单卡直接运行会导致显卡分配出错
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# fixme
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# 设置环境变量以减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# 深度学习模型包装类。接收一个编码器和解码器，用于处理输入图像，最终输出深度图 (disp)。
class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder  # 编码器，通常用于从输入数据（例如图像）中提取特征。
        self.decoder = decoder  # 解码器，用来处理提取的特征并生成所需的输出（在此例中，是视差图

    # 前向传播，将输入图像传递给编码器，提取特征并通过解码器生成输出（即视差图 disp）
    def forward(self, input_image):
        features = self.encoder(
            input_image)  # input_image 是输入的图像数据，通常是一个张量（tensor），其形状可能为 [batch_size, channels, height, width]。
        outputs = self.decoder(features)  # 将编码器生成的 features 特征图传递给解码器 decoder，生成最终的输出。
        disp = outputs[
            ("disp", 0)]  # 表示从字典中提取尺度为 0 的视差图 disp，视差图是指每个像素的位移量，可以用于计算深度信息，其值通常是通过 sigmoid 函数限制在 [0, 1] 范围内。
        return disp


def disp_to_depth(disp, min_depth, max_depth):
    # """Convert network's sigmoid output into depth prediction
    # The formula for this conversion is given in the 'additional considerations'
    # section of the paper.
    # """
    min_disp = 1 / max_depth  # 最小深度值对应的视差
    max_disp = 1 / min_depth  # 最大深度值对应的视差
    scaled_disp = min_disp + (max_disp - min_disp) * disp  # 将输入的 disp 进行缩放，使其值从 [0, 1] 变换到 [min_disp, max_disp] 之间。
    depth = 1 / scaled_disp  # 根据视差与深度的反比关系，反转视差得到的深度
    return scaled_disp, depth


# 获取平均深度差异
def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask):  # 对抗样本视差图，基准样本视差图，掩码张量（指定需要计算的区域）
    scaler = 5.4  # 用于调整深度值的比例因子。
    # print(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1])
    # print(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1].shape)
    # print(torch.max(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]))
    # print(torch.min(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]))
    # print(torch.max(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]))
    # print(torch.min(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]))
    # print(torch.sum(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask.unsqueeze(0))/torch.sum(scene_car_mask))
    # print(torch.sum(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask.unsqueeze(0))/torch.sum(scene_car_mask))

    # 先将视差图（adv_disp1或ben_disp2）转换为深度图，然后乘以 scene_car_mask（掩码），只保留车辆部分的深度，并使用scaler调整深度值。
    # 再使用 torch.clamp 将深度值限制在最大值 50，以避免异常值影响后续计算。
    dep1_adv = torch.clamp(disp_to_depth(torch.abs(adv_disp1), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler,
                           max=50)
    dep2_ben = torch.clamp(disp_to_depth(torch.abs(ben_disp2), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler,
                           max=50)
    mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben))/torch.sum(scene_car_mask)

    # 计算每个像素之间的深度差，然后求和。将总深度差除以有效像素数，得到平均深度差。
    #mean_depth_diff = torch.sum(dep1_adv - dep2_ben) / torch.sum(scene_car_mask)
    return mean_depth_diff


# 计算样本之间的影响比率
def get_affected_ratio(disp1, disp2, scene_car_mask):
    scaler = 5.4
    dep1 = torch.clamp(disp_to_depth(torch.abs(disp1), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler, max=50)
    dep2 = torch.clamp(disp_to_depth(torch.abs(disp2), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler, max=50)
    # 创建与dep1与dep2相同形状的张量，分别填充全1和全0
    ones = torch.ones_like(dep1)
    zeros = torch.zeros_like(dep1)
    # 计算影响比率，scene_car_mask.unsqueeze(0) 用于确保掩码与深度差异的形状匹配。将影响的像素总数除以有效像素数量（torch.sum(scene_car_mask)），计算出影响比率。
    affected_ratio = torch.sum(scene_car_mask.unsqueeze(0) * torch.where((dep1 - dep2) > 1, ones, zeros)) / torch.sum(scene_car_mask)
    return affected_ratio


def loss_smooth(img):
    # img: [batch_size, h, w, 3]
    b, c, w, h = img.shape  # b: 批次大小，c: 通道数（如 RGB 通道），w: 宽度，h: 高度。
    # 计算相邻像素的平方差
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)  # 水平方向，当前像素和左边相邻像素的差
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)  # 垂直方向，当前像素和上方相邻像素的差
    return torch.square(torch.sum(s1 + s2)) / (b * c * w * h)


def loss_nps(img, color_set):
    # img: [batch_size, h, w, 3]
    # color_set: [color_num, 3]
    _, h, w, c = img.shape  # _:批次大小，h：高，w：宽 c：通道
    color_num, c = color_set.shape
    img1 = img.unsqueeze(1)  # 在img的第二维插入一个新的维度，形状变为[batch_size, 1, height, width, 3]
    color_set1 = color_set.unsqueeze(1).unsqueeze(1).unsqueeze(
        0)  # 对color_set三次插入，形状为 [color_num, 1, 1, 1, 3]，以便进行广播操作。
    # 计算图像与颜色集合之间的差异
    gap = torch.min(torch.sum(torch.abs(img1 - color_set1) / 255, -1), 1).values
    return torch.sum(gap) / h / w


def attack(args):
    # 定义分割宽度、标题高度、字体样式、字体缩放和字体粗细，用于后续的图像拼接和标题添加。
    margin_width = 50
    caption_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    '''
    """ 通过DPT_DINOv2直接构建模型. """
    # 加载模型 ，用DPT_DINOv2 初始化 depth_anything 模型
    depth_anything = DPT_DINOv2().to(args.device0).eval()
    # 加载预训练权重
    weights = torch.load("./checkpoints/depth_anything_vitl14.pth",weights_only=True)
    depth_anything.load_state_dict(weights)
    # 计算并输出模型的总参数数量，以百万（M）为单位。为了衡量模型的大小和潜在的计算成本
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    '''



    """ 构建好模型深度估计模型"""
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    encoder = 'vitb'  # or 'vitb', 'vits'
    depth_anything = DepthAnything(model_configs[encoder]).to(args.device0).eval()
    depth_anything.load_state_dict(torch.load(f'../checkpoints/depth_anything_{encoder}14.pth'))
    # 计算并输出模型的总参数数量，以百万（M）为单位。为了衡量模型的大小和潜在的计算成本
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    # 定义图像预处理步骤
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
        # 将图像准备为网络的输入格式（例如改变通道顺序）。
        PrepareForNet(),
    ])


    """"伪装纹理生成"""
    # 伪装参数设置;根据传入的伪装形状和分辨率计算所需的高度和宽度。
    H, W = args.camou_shape, args.camou_shape#默认都是1024
    resolution = 8  # 伪装图像分辨率
    h, w = int(H / resolution), int(W / resolution)#都是128

    # 扩展卷积核: 创建一个转置卷积层，用于将伪装参数从小尺寸扩展到较大尺寸，并初始化权重和偏置。
    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(args.device1)
    # 初始化卷积核：
    expand_kernel.weight.data.fill_(0)  # 所有权重初始化为0
    expand_kernel.bias.data.fill_(0)  # 所有偏置初始化为0
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)  # 对于每一个通道，将对角线的权重初始化为1

    # 颜色集: 定义一组（10个RGB）颜色，转换为 torch.tensor 并归一化到 [0, 1] 范围，用于生成伪装图像的颜色选择
    color_set = torch.tensor(
        [[0, 0, 0], [255, 255, 255], [0, 18, 79], [5, 80, 214], [71, 178, 243], [178, 159, 211], [77, 58, 0],
         [211, 191, 167], [247, 110, 26], [110, 76, 16]]).to(args.device1).float() / 255

    # 初始化伪装参数: 随机生成一个大小为 [1（b）, h, w, 3] 的三通道伪装参数，表示伪装图像的 RGB 颜色。
    camou_para = torch.rand([1, h, w, 3]).float().to(args.device1)

    camou_para_np = camou_para.detach().cpu().numpy()
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #展示 camou_para
    axes[0].imshow(camou_para_np[0])  # camou_para 也是 [1, H, W, 3]
    axes[0].set_title('Camouflage Parameter')
    axes[0].axis('off')
    camou_para.requires_grad_(True)  # 表示允许伪装参数在优化过程中更新
    print(f"Original shape: {camou_para.shape}")

    # 优化器设置: 使用 Adam 优化器来更新伪装参数。
    optimizer = optim.Adam([camou_para], lr=args.lr)

    # 扩展伪装参数: 将伪装参数通过扩展卷积核进行处理，调整其维度。
    # permute(0, 3, 1, 2): 将伪装参数的维度从 [1, h, w, 3] 转换为 [1, 3, h, w]，以适应卷积操作的输入格式。
    # expand_kernel: 使用转置卷积将伪装参数扩展到原始的图像分辨率。
    # permute(0, 2, 3, 1): 将结果的维度转换回 [1(B), H, W, 3] 格式。
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


    """创建数据集和数据加载器:"""
    # 使用 MyDataset 类加载训练数据集，传入了训练目录、图像大小、目标名称、伪装掩码和设备。
    dataset = MyDataset(args.train_dir, args.img_size, args.obj_name, args.camou_mask, args.device1)
    # DataLoader: 使用 PyTorch 的数据加载器按批次加载数据，batch_size 是批处理大小，shuffle=False 意味着不打乱数据顺序。
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=2,
    )

    # 设置纹理：将初始计算好的伪装纹理camou_para1应用到数据集每张图像中去中
    dataset.set_textures(camou_para1)#(B,H,W,C)的tensor

    # 训练循环: 进行 15 个训练轮次，并在每个批次中迭代数据加载器。
    for epoch in range(15):
        print('-' * 30 + 'epoch begin: ' + str(epoch) + '-' * 30)
        tqdm_loader = tqdm(loader)
        MEAN=0
        G=0
        for i, (index, total_img, total_img0, mask, img, _, _) in enumerate(tqdm_loader):#得到的total（B,C,H,W）
            # 调整被攻击图像到模型输入的大小
            input_image0 = total_img0.squeeze(0)  # 去掉第一个维度，(1(b),3(c),320,1024)得到形状为 (3, 320, 1024)
            # 获取图像的高度和宽度
            h0, w0 = input_image0.shape[1:3]  # (320, 1024)
            # 应用预处理函数
            # 处理之前需要将input_image0(3,320,1024)---变为-----（320，1024，3）
            input_image0 = input_image0.permute(1, 2, 0)  # 变为 (320, 1024, 3)
            # 确保 input_image0 是一个 CUDA 张量,移动到cpu
            if input_image0.is_cuda:
                input_image0 = input_image0.cpu()
            input_image0 = transform({'image': input_image0.detach().numpy()})['image']#(3(c),518,1652)
            # 增加一个维度并将其转为 PyTorch 张量
            input_image0 = torch.from_numpy(input_image0).unsqueeze(0).to(args.device1)#(1,3,518,1652)（B,C,H,W）

            # 调整攻击图像到模型输入的大小
            input_image = total_img.squeeze(0)  # 去掉第一个维度，得到形状为 (3, 320, 1024)
            # 获取图像的高度和宽度
            h1, w1 = input_image.shape[1:3]  # (320, 1024)
            # 应用预处理函数
            #处理之前需要将input_image(3,320,1024)---变为-----（320，1024，3）
            input_image = input_image.permute(1, 2, 0)  # 变为 (320, 1024, 3)
            # 确保 input_image0 是一个 CUDA 张量,移动到cpu
            if input_image.is_cuda:
                input_image = input_image.cpu()
            #numpy()：只有在张量不需要梯度时才能安全调用,否则要加上detach()
            input_image = transform({'image': input_image.detach().numpy()})['image']
            # 增加一个维度并将其转为 PyTorch 张量
            input_image = torch.from_numpy(input_image).unsqueeze(0).to(args.device1)#(1,3,518,1652)（B,C,H,W）


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




            with torch.no_grad():
            # 深度估计模型推理(正向传播)
            # 保证统一设备下计算
                 input_image_a=input_image.to(args.device0)
                 outputs = depth_anything(input_image_a)#outputs为(1(b),518,1652)（0-255）为归一化的
            """将攻击的深度图还原到原始图像大小，归一化并转换为 8 位整型的图像格式,适合图像保存"""
            depth = F.interpolate(outputs[None], (h1, w1), mode='bilinear',align_corners=False)#未归一化的张量 b  # tensor(1,518,1652)----tensor(1,1,320,1024)
            #depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            #depth = depth.cpu().numpy().astype(np.uint8)

            #adv_los = torch.sum(10 * torch.pow(depth * mask, 2)) / torch.sum(mask)

            print(f"显存0使用情况: {torch.cuda.memory_allocated(args.device0) / (1024 ** 2):.2f} MB")
            print(f"显存1使用情况: {torch.cuda.memory_allocated(args.device1) / (1024 ** 2):.2f} MB")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




            with torch.no_grad():
                # 深度估计模型推理(正向传播)
                #保证统一设备下计算
                input_image_b=input_image0.to(args.device0)
                outputs0 = depth_anything(input_image_b)#(1(b),518,1652)（0-255）
            """将没有攻击深度图还原到原始图像大小，归一化并转换为 8 位整型的图像格式,适合图像保存"""
            depth0 = F.interpolate(outputs0[None], (h0, w0), mode='bilinear', align_corners=False)[0, 0]#(tensor(1,518,1652)----tensor(1,1,320,1024)
            depth0 = (depth0 - depth0.min()) / (depth0.max() - depth0.min())
            #depth0 = depth0.cpu().numpy().astype(np.uint8)
            # 删除张量
            del input_image0
            del input_image_b
            del outputs0
            # 强制进行垃圾回收
            gc.collect()
            # 清空缓存
            torch.cuda.empty_cache()
            #print(f"显存0使用情况: {torch.cuda.memory_allocated(args.device0) / (1024 ** 2):.2f} MB")
            #print(f"显存1使用情况: {torch.cuda.memory_allocated(args.device1) / (1024 ** 2):.2f} MB")
            ##########################################################################################################

            if i%199==0:
            ##保存现在所生成的图像
                total_img_np = total_img.data.cpu().numpy()[0] * 255
                total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))
                total_img_np.save(os.path.join(args.outdir, 'test_total.jpg'))
                total_img_np0 = total_img0.data.cpu().numpy()[0] * 255
                total_img_np0 = Image.fromarray(np.transpose(total_img_np0, (1,2,0)).astype('uint8'))
                total_img_np0.save(os.path.join(args.outdir, 'test_total0.jpg'))
                print("保存total成功")
            ##保存当前的掩码
            # Image.fromarray((255 * mask).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(args.log_dir, str(i) + 'mask.png'))

            ###############################################################################################################

            if i%5==0:
                depth_squeezed = depth.squeeze(0).squeeze(0)  # 移除前两个维度(1,1,320,1024)----(320,1024)
                depth_squeezed = (depth_squeezed - depth_squeezed.min()) / (depth_squeezed.max() - depth_squeezed.min()) * 255.0
                depth_squeezed = depth_squeezed.detach().cpu().numpy().astype(np.uint8)
                # 若设置 grayscale，生成灰度图像，否则将深度图应用伪彩色 COLORMAP_INFERNO。

                if args.grayscale:
                    depth_squeezed = np.repeat(depth_squeezed[..., np.newaxis], 3, axis=-1)
                else:
                    depth_squeezed = cv2.applyColorMap(depth_squeezed, cv2.COLORMAP_INFERNO)



                depth0_squeezed = depth0.squeeze(0).squeeze(0)  # 移除前两个维度(1,1,320,1024)----(320,1024)
                depth0_squeezed = (depth0_squeezed - depth0_squeezed.min()) / (depth0_squeezed.max() - depth0_squeezed.min()) * 255.0
                depth0_squeezed = depth0_squeezed.cpu().numpy().astype(np.uint8)
                # 若设置 grayscale，生成灰度图像，否则将深度图应用伪彩色 COLORMAP_INFERNO。
                if args.grayscale:
                    depth0_squeezed = np.repeat(depth0_squeezed[..., np.newaxis], 3, axis=-1)
                else:
                    depth0_squeezed = cv2.applyColorMap(depth0_squeezed, cv2.COLORMAP_INFERNO)



                # 若设置 pred-only，则保存深度图预测图像，文件名添加 adv_depth 后缀，否则就拼接图象。
                if args.pred_only:
                    cv2.imwrite(os.path.join(args.outdir,  f"{i}_adv_depth.png"), depth_squeezed)
                else:
                    split_region = np.ones((depth0_squeezed.shape[0], margin_width, 3), dtype=np.uint8) * 255
                    combined_results = cv2.hconcat([depth0_squeezed, split_region, depth_squeezed])

                    caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
                    captions = ['Raw image', 'Attack image']
                    segment_width = w + margin_width

                    # 在标题区域添加文字，显示“Raw image”和“Attack image”。
                    for j, caption in enumerate(captions):
                        # Calculate text size
                        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                        # Calculate x-coordinate to center the text
                        text_x = int((segment_width * j) + (w - text_size[0]) / 2)
                        # Add text caption
                        cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0),font_thickness)

                    final_result = cv2.vconcat([caption_space, combined_results])

                    cv2.imwrite(os.path.join(args.outdir,  f"combin_depth.png" ),final_result)

            # 调整掩码尺寸,mask 是一个二值掩码，用于标识图像中伪装对象的位置。
            mask = mask.mean(dim=1)  # 将第二个维度求平均,将其去除，(1,3,320,1024)----(1, 320, 1024)
            #将depth0 和 depth 移到device1上去，与mask保持在一个设备下
            depth=depth.to(args.device1)#(未归一化)
            depth0 = depth0.to(args.device1)

            MEAN=get_mean_depth_diff(depth,depth0,mask).item()+MEAN
            print(f"\nMEAN为{MEAN}m")
            G=get_affected_ratio(depth,depth0,mask).item()+G
            print(f"\nG为{G}")
            # 总损失计算

            adv_loss = torch.sum(20 * torch.pow(depth * mask, 2)) / torch.sum(mask)
            print("\nadvloos:",adv_loss)
            tv_loss = loss_smooth(camou_para)
            nps_loss = loss_nps(camou_para, color_set)
            loss = (tv_loss * 1e-1) + adv_loss + (nps_loss * 5)
            print(loss)
            # 反向传播: 清空梯度，计算损失的反向传播，并更新伪装参数。
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # 更新纹理：重新计算扩展后的伪装参数，并将其限制在 [0, 1] 范围内，更新数据集中的纹理。(B,H,W,C)
            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            camou_para1 = torch.clamp(camou_para1, 0, 1)
            dataset.set_textures(camou_para1)

        # 保存结果: 将当前轮次的伪装参数保存为 PNG 图像和 NumPy 数组。
        # 将伪装参数 camou_para1 从张量转换为 NumPy 格式，并将其乘以 255 转换为 8 位图像。
        camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # 使用 cv2.imwrite 将其保存为 .png 图像文件，文件名包含当前的轮次 epoch。
        cv2.imwrite(args.log_dir + str(epoch) + 'camou.png', camou_png)
        # 使用 np.save 将伪装参数的原始数据保存为 .npy 文件，以便后续分析或恢复。
        np.save(args.log_dir + str(epoch) + 'camou.npy', camou_para.detach().cpu().numpy())
        print(f"第{epoch}轮平均误差为{MEAN/8400}")
        print(f"第{epoch}轮平均误差为{G/8400}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camou_mask", type=str, default='../data/car/mask.jpg', help="camouflage texture mask")  # 伪装纹理掩码路径
    parser.add_argument("--camou_shape", type=int, default=1024, help="shape of camouflage texture")  # 伪装纹理的形状，默认1024
    parser.add_argument("--obj_name", type=str, default='../data/car/lexus_hs.obj')  # 3D 模型文件路径,
    parser.add_argument("--device0", type=str, default="cuda:1", help="Device for Model 强壮.")
    parser.add_argument("--device1", type=str, default="cuda:0", help="Device for Model 弱鸡.")
    parser.add_argument("--train_dir", type=str, default='../data/mde_carla_rgb/')  # 训练数据目录，默认地址是dffault
    parser.add_argument("--img_size", type=int, nargs=2, default=(320, 1024))  # 图像大小,nargs允许用户在命令行中传递两个整数作为图像的尺寸（高度和宽度）
    parser.add_argument("--batch_size", type=int,default=1)  # 批量大小，默认值为 4， 这里的default数字是batch的大小（也就是同时处理样本的数量），如果跑的过程中发生了out of memory的报错，把这个调小，只能再减小成2或者1了
    parser.add_argument("--lr", type=float, default=0.01)  # 学习率，默认值为 0.01
    parser.add_argument("--log_dir", type=str, default='./Texture')  # 日志目录地址，默认值地址为(保存生成的纹理)
    parser.add_argument('--outdir', type=str, default='./results/vib_depth')#保存当前生成的深度结果图像
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    args = parser.parse_args()

    #检查可用的GPU
    print("Available devices:", torch.cuda.device_count())
    # 将命令行参数转换为 torch.device 对象
    args.device0 = torch.device(args.device0)
    args.device1 = torch.device(args.device1)
    # 将 --img_size 转换为 tuple
    args.img_size = tuple(args.img_size)  # 确保img_size 会被正确解析为元组。

    attack(args)

