import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import math
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

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

import networks
from utils import download_model_if_doesnt_exist
from data_loader_mde import MyDataset
import sys



# 这里是系统有多卡的时候的配置代码，我们单卡直接运行会导致显卡分配出错
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
# fixme
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


#深度学习模型包装类。接收一个编码器和解码器，用于处理输入图像，最终输出深度图 (disp)。
class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder#编码器，通常用于从输入数据（例如图像）中提取特征。
        self.decoder = decoder#解码器，用来处理提取的特征并生成所需的输出（在此例中，是视差图

    #前向传播，将输入图像传递给编码器，提取特征并通过解码器生成输出（即视差图 disp）
    def forward(self, input_image):
        features = self.encoder(input_image)#input_image 是输入的图像数据，通常是一个张量（tensor），其形状可能为 [batch_size, channels, height, width]。
        outputs = self.decoder(features)#将编码器生成的 features 特征图传递给解码器 decoder，生成最终的输出。
        disp = outputs[("disp", 0)]#表示从字典中提取尺度为 0 的视差图 disp，视差图是指每个像素的位移量，可以用于计算深度信息，其值通常是通过 sigmoid 函数限制在 [0, 1] 范围内。
        return disp


def disp_to_depth(disp,min_depth,max_depth):
# """Convert network's sigmoid output into depth prediction
# The formula for this conversion is given in the 'additional considerations'
# section of the paper.
# """
    min_disp=1/max_depth#最小深度值对应的视差
    max_disp=1/min_depth#最大深度值对应的视差
    scaled_disp=min_disp+(max_disp-min_disp)*disp#将输入的 disp 进行缩放，使其值从 [0, 1] 变换到 [min_disp, max_disp] 之间。
    depth=1/scaled_disp#根据视差与深度的反比关系，反转视差得到的深度
    return scaled_disp,depth

#获取平均深度差异
def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask):#对抗样本视差图，基准样本视差图，掩码张量（指定需要计算的区域）
    scaler=5.4# 用于调整深度值的比例因子。
    # print(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1])
    # print(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1].shape)
    # print(torch.max(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]))
    # print(torch.min(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]))
    # print(torch.max(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]))
    # print(torch.min(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]))
    # print(torch.sum(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask.unsqueeze(0))/torch.sum(scene_car_mask))
    # print(torch.sum(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask.unsqueeze(0))/torch.sum(scene_car_mask))

    #先将视差图（adv_disp1或ben_disp2）转换为深度图，然后乘以 scene_car_mask（掩码），只保留车辆部分的深度，并使用scaler调整深度值。
    #再使用 torch.clamp 将深度值限制在最大值 50，以避免异常值影响后续计算。
    dep1_adv=torch.clamp(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    dep2_ben=torch.clamp(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    # mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben))/torch.sum(scene_car_mask)

    #计算每个像素之间的深度差，然后求和。将总深度差除以有效像素数，得到平均深度差。
    mean_depth_diff = torch.sum(dep1_adv-dep2_ben)/torch.sum(scene_car_mask)
    return mean_depth_diff

#计算样本之间的影响比率
def get_affected_ratio(disp1, disp2, scene_car_mask):
    scaler=5.4
    dep1=torch.clamp(disp_to_depth(torch.abs(disp1),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    dep2=torch.clamp(disp_to_depth(torch.abs(disp2),0.1,100)[1]*scene_car_mask.unsqueeze(0)*scaler,max=50)
    # 创建与dep1与dep2相同形状的张量，分别填充全1和全0
    ones = torch.ones_like(dep1)
    zeros = torch.zeros_like(dep1)
    #计算影响比率，scene_car_mask.unsqueeze(0) 用于确保掩码与深度差异的形状匹配。将影响的像素总数除以有效像素数量（torch.sum(scene_car_mask)），计算出影响比率。
    affected_ratio = torch.sum(scene_car_mask.unsqueeze(0)*torch.where((dep1-dep2)>1, ones, zeros))/torch.sum(scene_car_mask)
    return affected_ratio


def loss_smooth(img):
    # img: [batch_size, h, w, 3]
    b, c, w, h = img.shape#b: 批次大小，c: 通道数（如 RGB 通道），w: 宽度，h: 高度。
    #计算相邻像素的平方差
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)#水平方向，当前像素和左边相邻像素的差
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)#垂直方向，当前像素和上方相邻像素的差
    return torch.square(torch.sum(s1 + s2)) / (b*c*w*h)
    

def loss_nps(img, color_set):
    # img: [batch_size, h, w, 3]
    # color_set: [color_num, 3]
    _, h, w, c = img.shape#_:批次大小，h：高，w：宽 c：通道
    color_num, c = color_set.shape
    img1 = img.unsqueeze(1)#在img的第二维插入一个新的维度，形状变为[batch_size, 1, height, width, 3]
    color_set1 = color_set.unsqueeze(1).unsqueeze(1).unsqueeze(0)#对color_set三次插入，形状为 [color_num, 1, 1, 1, 3]，以便进行广播操作。
    #计算图像与颜色集合之间的差异
    gap = torch.min(torch.sum(torch.abs(img1 - color_set1)/255, -1), 1).values
    return torch.sum(gap)/h/w


def attack(args):
    # 这里需要在github上提供的链接里下载模型，
    model_name = "my_mono+stereo_1024x320"  # 指定用于攻击的模型名称。这表示这是一个使用 Mono+Stereo 方法在 1024x320 分辨率上微调过的深度估计模型，可能是基于 Carla 数据集进行的微调。
    # download_model_if_doesnt_exist(model_name)      # 这个函数是判断是否有模型文件，没有就下载，但是因为网络问题，按照它的逻辑无法直接执行，所以直接注释即可

    # 此处是按照服务器的规则进行包装化的路径书写，在windows系统中无法直接识别，所以直接在这里改写文件绝对路径即可
    # encoder_path = os.path.join("models", model_name, "encoder.pth")
    encoder_path = r"D:\Black Attack\3D2Fool\models\mymono+stereo_1024x320\encoder.pth"

    # depth_decoder_path = os.path.join("models", model_name, "depth.pth")
    depth_decoder_path = r"D:\Black Attack\3D2Fool\models\mymono+stereo_1024x320\depth.pth"

    # LOADING PRETRAINED MODEL加载预训练模型
    #创建 ResnetEncoder 和 DepthDecoder 的实例，这些网络将用于深度估计。
    encoder = networks.ResnetEncoder(18, False)#使用 ResNet-18 架构作为编码器，并且没有使用预训练权重。
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))#基于编码器的输出通道数（num_ch_enc），设置了用于深度估计的解码器，输出四个不同尺度的深度图。

    #加载编码器权重
    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')#从encoder_path加载编码器的权重文件
    #filtered_dict_enc: 通过筛选 loaded_dict_enc 中的键和值，确保只加载与编码器模型状态匹配的部分。
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)#将筛选后的权重加载到编码器模型中。

    #加载深度解码器权重
    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)#将加载的权重应用于深度解码器模型。

    #模型包装
    #创建 DepthModelWrapper，它封装了编码器和解码器，使其能够协同工作，最终形成完整的深度估计模型。
    #to(args.device) 将模型移动到指定设备上（如 CPU 或 GPU）
    depth_model = DepthModelWrapper(encoder, depth_decoder).to(args.device)

    #模型设置: 将模型设置为评估模式；并设置 requires_grad_(False)，确保其参数不计算梯度，以节省内存和计算。
    depth_model.eval()
    for para in depth_model.parameters():
        para.requires_grad_(True)

    #获取输入尺寸: 从编码器权重中提取训练时的输入图像的高度和宽度，用于调整输入图像的尺寸。
    feed_height = loaded_dict_enc['height']
    feed_width  = loaded_dict_enc['width']
    input_resize = transforms.Resize([feed_height, feed_width])#创建一个调整输入图像大小的变换函数，以适应模型的输入要求。
    # keys = [("disp", 0), ("disp", 1), ("disp", 2), ("disp", 3)]
    # disp_size = [[192, 640], [96, 320], [48, 160], [24, 80]]

    #伪装参数设置;根据传入的伪装形状和分辨率计算所需的高度和宽度。
    H, W = args.camou_shape, args.camou_shape
    resolution = 8#伪装图像分辨率
    h, w = int(H/resolution), int(W/resolution)

    #扩展卷积核: 创建一个转置卷积层，用于将伪装参数从小尺寸扩展到较大尺寸，并初始化权重和偏置。
    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(args.device)
    #初始化卷积核：
    expand_kernel.weight.data.fill_(0)#所有权重初始化为0
    expand_kernel.bias.data.fill_(0)#所有偏置初始化为0
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)#对于每一个通道，将对角线的权重初始化为1

    #颜色集: 定义一组（10个RGB）颜色，转换为 torch.tensor 并归一化到 [0, 1] 范围，用于生成伪装图像的颜色选择
    color_set = torch.tensor([[0,0,0],[255,255,255],[0,18,79],[5,80,214],[71,178,243],[178,159,211],[77,58,0],[211,191,167],[247,110,26],[110,76,16]]).to(args.device).float() / 255


    # 初始化伪装参数: 随机生成一个大小为 [1, h, w, 3] 的三通道伪装参数，表示伪装图像的 RGB 颜色。
    camou_para = torch.rand([1, h, w, 3]).float().to(args.device)

    #选定一个纹理作为起始纹理
    #texy= np.load('./1.npy')
    #camou_para = torch.from_numpy(texy).float().to(args.device)
    camou_para_np = camou_para.detach().cpu().numpy()

    # 绘图
    #fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 展示 camou_para
    #axes[0].imshow(camou_para_np[0])  # camou_para 也是 [1, H, W, 3]
    #axes[0].set_title('Camouflage Parameter')
    #axes[0].axis('off')


    camou_para.requires_grad_(True)#表示允许伪装参数在优化过程中更新
    print(f"Original shape: {camou_para.shape}")

    #优化器设置: 使用 Adam 优化器来更新伪装参数。
    optimizer = optim.Adam([camou_para], lr=args.lr)

    #扩展伪装参数: 将伪装参数通过扩展卷积核进行处理，调整其维度。
    #permute(0, 3, 1, 2): 将伪装参数的维度从 [1, h, w, 3] 转换为 [1, 3, h, w]，以适应卷积操作的输入格式。
    #expand_kernel: 使用转置卷积将伪装参数扩展到原始的图像分辨率。
    #permute(0, 2, 3, 1): 将结果的维度转换回 [1, H, W, 3] 格式。
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


    #创建数据集和数据加载器:
    #使用 MyDataset 类加载训练数据集，传入了训练目录、图像大小、目标名称、伪装掩码和设备。
    dataset = MyDataset(args.train_dir, args.img_size, args.obj_name, args.camou_mask, args.device)
    #DataLoader: 使用 PyTorch 的数据加载器按批次加载数据，batch_size 是批处理大小，shuffle=False 意味着不打乱数据顺序。
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=2,
    )


    #设置纹理：将初始计算好的伪装纹理camou_para1应用到数据集每张图像中去中
    dataset.set_textures(camou_para1)


    #训练循环: 进行 15 个训练轮次，并在每个批次中迭代数据加载器。
    for epoch in range(1):
        print('-'*30 + 'epoch begin: ' + str(epoch) + '-'*30)
        tqdm_loader = tqdm(loader)
        for i, (index, total_img, total_img0, mask, img, _, _) in enumerate(tqdm_loader):

            #调整到模型输入的大小
            input_image = input_resize(total_img)
            input_image0 = input_resize(total_img0)
            #深度估计模型推理(正向传播)
            outputs = depth_model(input_image)
            outputs0 = depth_model(input_image0)


##########################################################################################################

        #if i%3==0:
        ##保存现在所生成的图像
            #total_img_np = total_img.data.cpu().numpy()[0] * 255
            #total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))
            #total_img_np.save(os.path.join(args.log_dir, 'test_total.jpg'))
            #total_img_np0 = total_img0.data.cpu().numpy()[0] * 255
            #total_img_np0 = Image.fromarray(np.transpose(total_img_np0, (1,2,0)).astype('uint8'))
            #total_img_np0.save(os.path.join(args.log_dir, 'test_total0.jpg'))
        ##保存当前的掩码

            #Image.fromarray((255 * mask).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(args.log_dir, str(i) + 'mask.png'))



###############################################################################################################
            # 调整掩码尺寸：使用 input_resize 函数调整 mask 的尺寸，并且只保留第一通道（[:, 0, :, :]）。mask 是一个二值掩码，用于标识图像中伪装对象的位置。
            mask = input_resize(mask)[:, 0, :, :]

            #总损失计算
            adv_loss = torch.sum(10 * torch.pow(outputs*mask,2))/torch.sum(mask)
            tv_loss = loss_smooth(camou_para)
            nps_loss = loss_nps(camou_para, color_set)
            loss = (tv_loss* 1e-1)-adv_loss + (nps_loss* 5)

            #反向传播: 清空梯度，计算损失的反向传播，并更新伪装参数。
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            #更新纹理：重新计算扩展后的伪装参数，并将其限制在 [0, 1] 范围内，更新数据集中的纹理。
            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            camou_para1 = torch.clamp(camou_para1, 0, 1)
            dataset.set_textures(camou_para1)

        #保存结果: 将当前轮次的伪装参数保存为 PNG 图像和 NumPy 数组。
        #将伪装参数 camou_para1 从张量转换为 NumPy 格式，并将其乘以 255 转换为 8 位图像。
        camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #使用 cv2.imwrite 将其保存为 .png 图像文件，文件名包含当前的轮次 epoch。
        cv2.imwrite(args.log_dir+str(epoch)+'camou.png', camou_png)
        #使用 np.save 将伪装参数的原始数据保存为 .npy 文件，以便后续分析或恢复。
        np.save(args.log_dir+str(epoch)+'camou.npy', camou_para.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camou_mask", type=str, default='./car/mask.jpg', help="camouflage texture mask")#伪装纹理掩码路径
    parser.add_argument("--camou_shape", type=int, default=1024, help="shape of camouflage texture")#伪装纹理的形状，默认1024
    parser.add_argument("--obj_name", type=str, default='./car/lexus_hs.obj')#3D 模型文件路径,
    parser.add_argument("--device", type=str, default=torch.device("cuda:0"))#计算的设备类型，默认为 cuda:0
    parser.add_argument("--train_dir", type=str, default='/data/zjh/mde_carla/')#训练数据目录，默认地址是dffault
    parser.add_argument("--img_size", type=int, nargs=2,default=(320, 1024))#图像大小,nargs允许用户在命令行中传递两个整数作为图像的尺寸（高度和宽度）
    parser.add_argument("--batch_size", type=int, default=4)#批量大小，默认值为 4， 这里的default数字是batch的大小（也就是同时处理样本的数量），如果跑的过程中发生了out of memory的报错，把这个调小，只能再减小成2或者1了
    parser.add_argument("--lr", type=float, default=0.01)#学习率，默认值为 0.01
    parser.add_argument("--log_dir", type=str, default='./res')#日志目录地址，默认值地址为 ./res/ 这里需要更改要保存的位置

    args = parser.parse_args()#解析命令行参数，并将结果存储在 args 对象中。

    # 将 --device 转换为 torch.device
    args.device = torch.device(args.device)
    # 将 --img_size 转换为 tuple
    args.img_size = tuple(args.img_size)#确保img_size 会被正确解析为元组。
    attack(args)
    
