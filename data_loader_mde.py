import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import math
import pickle
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader,
    materials
)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class MyDataset(Dataset):

    def __init__(self, data_dir, img_size, obj_name, camou_mask, device=torch.device("cuda:0"), tex_trans_flag=True, phy_trans_flag=True):
        self.tex_trans_flag = tex_trans_flag #纹理转换标志，表示是否需要对纹理进行转换。
        self.phy_trans_flag = phy_trans_flag #物理转换标志，表示是否需要物理转换。
        self.data_dir = data_dir + 'rgb/'
        test_path = r".\data\ann.pkl"#用于保存数据集注释的 .pkl 文件路径。通过 pickle.load 加载注释。
        with open(test_path, 'rb') as ann_file:
            self.ann = pickle.load(ann_file)#从文件中加载注释数据，存储在 self.ann 中
        self.files = os.listdir('./data/mde_carla_rgb')
        print('dataset length: ', len(self.files))

        #将图像尺寸 img_size 和设备 device 保存为类的属性，供后续使用。
        self.img_size = img_size
        self.device = device

        self.camou_mask = torch.from_numpy(cv2.imread(camou_mask)/255).to(device).unsqueeze(0).float()

        #加载3D模型：使用 load_obj() 函数从 obj_name 指定的路径中加载 3D 模型文件。
        self.verts, self.faces, self.aux = load_obj(
            obj_name,
            load_textures=True,
            create_texture_atlas=False,
            texture_atlas_size=4,
            texture_wrap='repeat',
            path_manager=None,
        )

        #获取原始纹理：从辅助信息中提取模型的原始表面纹理，并移动到计算设备，扩展维度使其成为 4D 张量（批量维度）。
        self.camou0 = list(self.aux.texture_images.values())[0].to(self.device)[None]  # 汽车原表面纹理

        #加载模型为网格：使用 load_objs_as_meshes() 函数将 .obj 文件加载为 Meshes 对象，方便在后续渲染时使用。
        self.mesh = load_objs_as_meshes([obj_name], device=device)

        #verts_uvs：UV 坐标，用于将纹理映射到模型表面。
        self.verts_uvs = self.aux.verts_uvs.to(device)  # (V, 2)

        #faces_uvs：面与 UV 坐标的对应关系，用于确定每个面上的纹理映射。
        self.faces_uvs = self.faces.textures_idx.to(device)  # (F, 3)

        #光栅化设置
        self.raster_settings = RasterizationSettings(
            image_size=self.img_size,#渲染图象大小
            blur_radius=0.0,#模糊半径，设置为0表示不进行模糊处理
            faces_per_pixel=1,#每个像素最多显示多少个三角面片。
            max_faces_per_bin=250000#每个 bin 内最大的面片数，通常用于优化性能。
        )

        #设置光源，用于模拟光照效果
        self.lights = PointLights(device=device, location=[[100.0, 85, 100.0]])
        #初始化相机设置
        self.cameras=''

        #设置渲染器
        #MeshRasterizer：基于相机和光栅化设置对网格进行渲染。
        #HardPhongShader：使用 Phong 着色器处理光照效果，模拟反射和阴影。
        self.renderer = MeshRenderer(

            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=device,
                cameras=self.cameras,
                lights=self.lights
            )
        )

    #纹理转换模块
    def tex_trans(self, camou):
        # mask=[1, 4096, 4096, 3], camou=[1, 1024, 1024, 3]
        camou_column = []#该列表用于保存每一列的伪装纹理片段。
        for i in range(6):
            camou_row_list = []#该列表用于保存每一行的伪装纹理片段。
            for j in range(6):
                #随机水平翻转：
                camou1 = T.RandomHorizontalFlip(p=0.5)(camou.permute(0, 3, 1, 2)[0]) # 依概率p水平翻转
                #随机垂直翻转
                camou2 = T.RandomVerticalFlip(p=0.5)(camou1)
                #随机旋转
                if np.random.rand(1)>0.5:
                    camou3 = TF.rotate(camou2, 90)
                else:
                    camou3 = camou2
                # temp = camou3.detach().cpu().permute(1,2,0).numpy()*255
                # cv2.imwrite('./assets/tex1.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))

                #每次处理后的纹理片段都添加到 camou_row_list 列表中。
                camou_row_list.append(camou3)

            #拼接行纹理片段：将 camou_row_list 中的 6 个纹理片段沿第 1 维（宽度）进行拼接，形成一整行的纹理。
            camou_row = torch.cat(tuple(camou_row_list), 1)
            # print(camou_row.shape)
            #将完整的一行纹理添加到列列表中：每次处理完一行纹理，将其添加到 camou_column 列表中。
            camou_column.append(camou_row)

        #拼接所有行纹理：将 camou_column 中的 6 行纹理沿第 2 维（高度）进行拼接，形成一个完整的大纹理图像，大小为 [4096, 4096, 3]，然后通过 unsqueeze(0) 扩展出批次维度，变为 [1, 4096, 4096, 3]。
        camou_full = torch.cat(tuple(camou_column), 2).unsqueeze(0)

        # temp = camou_full[0].detach().cpu().permute(1,2,0).numpy()*255
        # cv2.imwrite('./assets/tex2.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))

        #随即裁剪：裁剪大小为[4096,4096],再使用 permute(0, 2, 3, 1) 将维度重新排列，转换为 [1, 4096, 4096, 3] 的格式，适应后续处理。
        camou_crop = T.RandomCrop(4096)(camou_full).permute(0, 2, 3, 1) # 随机裁剪

        # temp = camou_crop[0].detach().cpu().numpy()*255
        # cv2.imwrite('./assets/tex3.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # print(camou_crop.shape)
        return camou_crop


    def tex_trans0(self, camou):
        # mask=[1, 4096, 4096, 3], camou=[1, 1024, 1024, 3]
        camou_column = []
        for i in range(4):
            camou_row_list = []
            for j in range(4):
                camou_row_list.append(camou)
            camou_row = torch.cat(tuple(camou_row_list), 1)
            camou_column.append(camou_row)
        camou_full = torch.cat(tuple(camou_column), 2)
        return camou_full

    #给输入图像加雾操作
    def frog(self, img, A=0.5, beta=0.08):
        # A:亮度, beta:雾化浓度, size:雾化尺寸, center:雾化中心
        (chs, row, col) = img[0].shape#从第一个通道中提取图像的通道数、行数（高度）、列数（宽度）
        img1 = img.clone()
        size = math.sqrt(max(row, col))#计算雾化尺寸
        #生成雾化中心：随机生成中心坐标center，
        center = [row * np.random.rand(1).tolist()[0], col * np.random.rand(1).tolist()[0]]
        #将 center 转换为 PyTorch 张量，并将其移动到指定的计算设备（通常是 GPU）。
        center = torch.tensor(center).to(self.device)
        #生成图像的坐标网络
        coordinates = torch.stack(torch.meshgrid(torch.arange(row), torch.arange(col)), -1).to(self.device)
        #计算每个像素到雾化中心点的距离d
        d = -0.04 * torch.sqrt(torch.sum(torch.pow(coordinates-center, 2), 2)) + size
        #计算雾化衰减因子
        td = torch.exp(-beta * d)
        #应用雾化效果
        img1[0] = img[0] * td + A * (1 - td)
        return img1

    def Blur_trans(self, img1, img0, dist):
        kernel_size_list = [1,3,5,7]#定义模糊核大小列表
        if dist < 5:
            if np.random.rand(1) < 0.3:
                kernel_size = kernel_size_list[np.random.randint(0,2)]#选择较小的核（1/3）
                delta = np.random.rand(1).tolist()[0] * dist / 10#模糊程度，由随机数和距离dist决定，最大不超过2，以避免过度模糊
                if delta > 2:
                    delta = 2
                img1 = T.GaussianBlur(kernel_size, delta)(img1)
                img0 = T.GaussianBlur(kernel_size, delta)(img0)
        elif dist < 8:
            if np.random.rand(1) < 0.6:
                kernel_size = kernel_size_list[np.random.randint(1,3)]
                delta = np.random.rand(1).tolist()[0] * dist / 15
                if delta > 2:
                    delta = 2
                img1 = T.GaussianBlur(kernel_size, delta)(img1)
                img0 = T.GaussianBlur(kernel_size, delta)(img0)
        else:
            if np.random.rand(1) < 0.8:
                kernel_size = kernel_size_list[np.random.randint(2,4)]
                delta = np.random.rand(1).tolist()[0] * dist / 20
                if delta > 2:
                    delta = 2
                img1 = T.GaussianBlur(kernel_size, delta)(img1)
                img0 = T.GaussianBlur(kernel_size, delta)(img0)
        return img1, img0

    def Color_trans(self, img1, img0, brightness=[0.7, 1.3], contrast=[0.9, 1.1], saturation=[0.9, 1.1], hue=[-0.05, 0.05]):
        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))
        def Color_change(img, b, c, s, h):
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h)
            return img
        img1 = Color_change(img1, b, c, s, h)
        img0 = Color_change(img0, b, c, s, h)
        return img1, img0

    def myColor_trans(self, img1, img0, flag=0, brightness=[0.7, 1.3], contrast=[0.9, 1.1], saturation=[0.9, 1.1], hue=[-0.05, 0.05]):
        #b:亮度 c：对比度 s：饱和度 h：色调
        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))
        def Color_change(img, b, c, s, h):
            #该函数对输入图像 img 应用颜色变换，使用 torchvision.transforms.functional (TF) 中的颜色调整函数：
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h)
            return img
        img1 = Color_change(img1, b, c, s, h)
        img0 = Color_change(img0, b, c, s, h)
        if flag == 1:
            img1, img0 = self.add_shadow(img1, img0)
        elif flag == 2:
            img1, img0 = self.add_exposure(img1, img0)
        return img1, img0

    ##添加阴影效果
    def add_shadow(self, img1, img0, shadow_dimension=5):#shadow_dimension为阴影多边形的顶点数量
        x1 = 0
        x2 = self.img_size[1]
        y1 = 0
        y2 = self.img_size[0]
        mask = np.ones([self.img_size[0], self.img_size[1], 3])
        vertex=[]
        for dimensions in range(shadow_dimension): ## Dimensionality of the shadow polygon
            vertex.append((random.randint(x1, x2),random.randint(y1, y2)))
        vertices = np.array([vertex], dtype=np.int32) ## 单个阴影顶点
        b = np.random.rand(1).tolist()[0] * 0.2 + 0.7##随机生成阴影强度（b）：阴影的强度在0.7到0.9之间，阴影区域变暗。
        cv2.fillPoly(mask, vertices, (b, b, b))
        # 对阴影掩码进行模糊处理：应用高斯模糊来平滑阴影边缘。
        mask = torch.from_numpy(mask).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
        mask = T.GaussianBlur(3, 1.5)(mask)
        #生成阴影效果
        img11 = img1 * mask[:,:,:,:]
        img00 = img0 * mask[:,:,:,:]
        return img11, img00

    #添加过曝效果：使图像的某些区域变得更加明亮。
    def add_exposure(self, img1, img0, exposure_dimension=5):#过曝区域的顶点数量。
        x1 = 0
        x2 = self.img_size[1]
        y1 = 0
        y2 = self.img_size[0]
        mask = np.ones([self.img_size[0], self.img_size[1], 3])
        vertex=[]
        for dimensions in range(exposure_dimension): ## Dimensionality of the shadow polygon
            vertex.append((random.randint(x1, x2),random.randint(y1, y2)))
        vertices = np.array([vertex], dtype=np.int32) ## 单个过曝顶点
        b = np.random.rand(1).tolist()[0] * 0.2 + 1.1
        cv2.fillPoly(mask, vertices, (b, b, b))
        mask = torch.from_numpy(mask).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
        mask = T.GaussianBlur(3, 1.5)(mask)
        img11 = torch.clamp(img1 * mask[:,:,:,:], 0, 1)
        img00 = torch.clamp(img0 * mask[:,:,:,:], 0, 1)
        return img11, img00

    #生成随机雨滴：用于生成雨滴的起点
    #no_of_drops：雨滴的数量。slant：雨滴的倾斜角度。drop_length：每个雨滴的长度。drop_width：每个雨滴的宽度。
    def generate_random_lines(self, no_of_drops=10, slant=0, drop_length=100, drop_width=2):
        #生成随机的雨滴坐标：为每个雨滴随机选择一个起点 (x, y)。
        drops = []
        for i in range(no_of_drops): ## 如果想下大雨，就增加no_of_drops即可
            if slant<0:
                x= np.random.randint(slant+drop_width, self.img_size[1]-drop_width)
            else:
                x= np.random.randint(drop_width, self.img_size[1]-slant-drop_width)
            y= np.random.randint(drop_width, self.img_size[0]-drop_length-drop_width)
            drops.append((x,y))
        return drops   ##返回的是雨滴的起始点（x，y）列表。


    ##添加雨滴效果：将雨滴效果应用到图像中。
    def add_rain(self, img1, img0, no_of_drops=10, drop_length=100, drop_width=2):
        slant= np.random.randint(-5,5)#倾斜角度随机-5度到5度之间

        # 对每个雨滴，在掩码上绘制一条白色的线，模拟雨滴的形状。
        rain_drops = self.generate_random_lines(no_of_drops, slant*2, drop_length, drop_width)#雨滴起点列表，表示所有生成的雨滴的坐标
        mask = np.ones([self.img_size[0], self.img_size[1], 3])
        mask_color = (0.85, 0.85, 0.85)#雨滴颜色（白色）
        for rain_drop in rain_drops:
            #绘制雨滴，即在掩码上画线，
            cv2.line(mask, (rain_drop[0],rain_drop[1]), (rain_drop[0]+slant,rain_drop[1]+drop_length), mask_color, drop_width)

        #将掩码转换为张量并进行高斯模糊，雨滴边缘更加柔和。
        mask = torch.from_numpy(mask).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
        mask = T.GaussianBlur(3, 1.5)(mask)
        img11 = img1 * mask[:,:,:,:]
        img00 = img0 * mask[:,:,:,:]
        return img11, img00

    ##增强训练的转换：在训练过程中通过随机变换增强数据的鲁棒性。
    def EoT(self, img1, img0, index):
        # self.files[index] = 'Town04_w2_0l_cam2.jpg'
        map = self.files[index].split('_') [0]#地图信息，根据'Town04_w2_0l_cam2.jpg'来判断
        weather = self.files[index].split('_')[1]  #天气条件，根据'Town04_w2_0l_cam2.jpg'来判断
        eye = self.ann[self.files[index]]['camera_pos'].copy()#从 self.ann 中获取相机的位置信息 eye
        dist = np.sqrt(np.sum(np.power(eye, 2)))#dist 计算的是相机与原点之间的距离，即相机到世界坐标系原点的欧氏距离。
        img1, img0 = self.Blur_trans(img1, img0, dist)  #高斯模糊
        flag = random.randint(0, 2)#用于后续的颜色转换中作为随机性参数。
        if weather == 'w1':  # 晴朗
            img1, img0 = self.Color_trans(img1, img0, brightness=[0.9, 1.3])#晴朗天气下，亮度在 [0.9, 1.3] 之间
        elif weather == 'w2':  # 多云
            img1, img0 = self.Color_trans(img1, img0, brightness=[0.8, 1.2])#多云天气下，亮度在 [0.8, 1.2] 之间
        elif weather == 'w3':  # 阴雨
            img1, img0 = self.Color_trans(img1, img0, brightness=[0.7, 1.1])#阴雨天气下，亮度在 [0.7, 1.1] 之间。
        return img1, img0


    ##物理环境转换
    def phy_trans(self, img1, img0, index):
        # self.files[index] = 'Town04_w2_0l_cam2.jpg'
        map = self.files[index].split('_')[0]
        weather = self.files[index].split('_')[1]
        eye = self.ann[self.files[index]]['camera_pos'].copy()
        dist = np.sqrt(np.sum(np.power(eye, 2)))
        img1, img0 = self.Blur_trans(img1, img0, dist)  #高斯模糊
        flag = random.randint(0, 2)
        if weather == 'w1':  # 晴朗
            # img1, img0 = self.Color_trans(img1, img0, brightness=[0.9, 1.3])
            img1, img0 = self.myColor_trans(img1, img0, flag, brightness=[0.9, 1.3])  #颜色变换
        elif weather == 'w2':  # 多云
            # img1, img0 = self.Color_trans(img1, img0, brightness=[0.8, 1.2])
            img1, img0 = self.myColor_trans(img1, img0, flag, brightness=[0.8, 1.2])  #颜色变换
            if map == 'Town04':  # 加雾
                A = dist / 15 * 0.3 + np.random.rand(1).tolist()[0] * 0.06 - 0.03  #控制雾气浓度参数的扰动
                beta = dist / 15 * 0.04 + np.random.rand(1).tolist()[0] * 0.01 - 0.005  #透明性参数扰动
                img1 = self.frog(img1, A, beta)
                img0 = self.frog(img0, A, beta)
        elif weather == 'w3':  # 阴雨
            # img1, img0 = self.Color_trans(img1, img0, brightness=[0.7, 1.1])
            img1, img0 = self.add_rain(img1, img0)
            img1, img0 = self.myColor_trans(img1, img0, flag, brightness=[0.7, 1.1])  #颜色变换
            if map == 'Town04':  # 加雾
                A = dist / 15 * 0.5 + np.random.rand(1).tolist()[0] * 0.1 - 0.05  #扰动
                beta = dist / 15 * 0.08 + np.random.rand(1).tolist()[0] * 0.02 - 0.01  #扰动
                img1 = self.frog(img1, A, beta)
                img0 = self.frog(img0, A, beta)
        return img1, img0


    ##将伪装纹理（camou）应用于3D网格
    def set_textures(self, camou):
        # temp = self.tex_trans(camou)
        # camou_png = cv2.cvtColor((temp[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./res/camou.png', camou_png)

        #self.camou0：原始伪装纹理。self.camou_mask：决定了纹理的混合比例的掩膜。self.tex_trans(camou)或self.tex_trans0(camou)：根据self.tex_trans_flag的值选择使用的纹理转换方法。
        #这行代码将原始伪装纹理和转换后的伪装纹理按照掩膜进行加权混合，从而生成最终的纹理图像。
        if self.tex_trans_flag:
            image = self.camou0 * (1-self.camou_mask) + self.tex_trans(camou) * self.camou_mask
        else:
            image = self.camou0 * (1-self.camou_mask) + self.tex_trans0(camou) * self.camou_mask
        # image_png = cv2.cvtColor((image[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./res/image.png', image_png)

        #设置网格纹理
        #TexturesUV(...)：构造一个UV纹理对象，接受顶点UV坐标和面UV坐标。
        #self.mesh.textures：将生成的纹理应用到网格上，确保在渲染时能够正确显示。
        self.mesh.textures = TexturesUV(verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=image)


    ##用于索引访问类的实例中的数据项。它主要负责加载相机参数、渲染图像和处理背景图像。
    def __getitem__(self, index):
        # self.files[index] = 'Town02_w2_0l_cam2.jpg'
        # print(self.files[index])

        # 加载相机参数
        eye = self.ann[self.files[index]]['camera_pos'].copy()#得到位置
        # eye = np.array([-4, -2.5, 1.3])
        eye[0] *= -1#翻转X轴坐标
        for i in range(3):
            eye[i] *= 20 #对相机的位置坐标进行缩放，使得相机位置适合场景。
        camera_up = (0, 0, 1)#指定了相机的上方方向为Z轴正方向。
        # caculate R, T matrix for camera

        #R：旋转矩阵  T：位移矩阵  使相机能正确地“看向”场景中的某一点（这里是(0, 0, 10)）。
        R, T = look_at_view_transform(eye=(tuple(eye),), up=(tuple(camera_up),), at=((0, 0, 10),))

        #创建一个透视相机对象，设置近远裁剪面和视野。
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=1.0, zfar=300.0, fov=45.0)
        # 创建光源点（设置为相机的位置，以便物体能得到合理的光照）
        self.renderer.shader.lights=PointLights(device=self.device, location=[eye])
        # 创建一个高光材质
        self.materials = Materials(
            device=self.device,
            specular_color=[[1.0, 1.0, 1.0]],#颜色
            shininess=500.0#光泽度
        )
        # 设置渲染器相机
        self.renderer.rasterizer.cameras=self.cameras  #分给渲染器的光栅化模块
        self.renderer.shader.cameras=self.cameras      #分给渲染器的着色器模块

        # rendering the adversarial vehicle image with white background
        #在白色背景下渲染对抗性车辆图像
        imgs_pred1 = self.renderer(self.mesh, materials=self.materials)[:, ..., :3]
        imgs_pred1 = imgs_pred1.permute(0, 3, 1, 2)   # [1, 3, 320, 1024]

        self.mesh0 = self.mesh.clone() #克隆原始网格
        self.mesh0.textures = TexturesUV(verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=self.camou0)#将新的纹理对象对克隆的网格进行渲染

        # rendering the clean vehicle image with white background
        #在白色背景下渲染干净的车辆图像
        imgs_pred0 = self.renderer(self.mesh0, materials=self.materials)[:, ..., :3]
        imgs_pred0 = imgs_pred0.permute(0, 3, 1, 2)   # [1, 3, 320, 1024]

        # using Physical Augmentation or EoT
        #使用物理增强或者EoT
        if self.phy_trans_flag:
            imgs_pred11, imgs_pred00 = self.phy_trans(imgs_pred1, imgs_pred0, index)
        else:
            imgs_pred11, imgs_pred00 = self.EoT(imgs_pred1, imgs_pred0, index)

        # loading background image sampling from Carla simulator
        #加载背景图像
        file_path = os.path.join(r".\data\mde_carla_rgb", self.files[index])
        # file_path = '/data/zjh/mde_carla/rgb/Town04_w2_0l_cam2.jpg'
        img = cv2.imread(file_path)  # [640, 1600, 3] BGR
        img = img[40:-100, :, ::-1]  # [500, 1600, 3] RGB
        img_cv = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # [320, 1024, 3]
        img = np.transpose(img_cv, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))  # [1, 3, 320, 1024]
        img = torch.from_numpy(img).cuda(device=self.device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        ###obtaining the vehicle mask to get the final adv-img and clean-img
        ###获取车辆掩码以获取最终的 adv-img 和 clean-img
        #这一步的作用是生成一个用于分割车辆区域和背景的掩码，其中背景区域的像素值为 0，车辆区域的像素值为 1。
        contour = torch.where((imgs_pred1 == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))

        #使用掩码 contour 将背景区域和车辆区域区分开来，生成带对抗性纹理的图像 total_img。contour=0,则为img背景，否则为img_pred11对抗车量
        total_img = torch.where((contour == 0.), img, imgs_pred11)  # [1, 3, 320, 1024]
        # 使用掩码 contour 将背景区域和车辆区域区分开来，生成不带带对抗性纹理的图像 total_img0。contour=0,则为img背景，否则为img_pred00干净车量
        total_img0 = torch.where((contour == 0.), img, imgs_pred00) # [1, 3, 320, 1024]

        # return [batch_size, 3, 320, 1024]
        return index, total_img[0], total_img0[0], contour[0], img[0], imgs_pred1[0], imgs_pred0[0]
        # return index, total_img[0], total_img0[0], contour[0], img[0]
    
        
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    device = torch.device("cuda:0")
    obj_name = './car/lexus_hs.obj'
    camou_mask = './car/mask.jpg'
    camou_para = np.load('./res/res_base_norcv/10camou.npy')
    # camou_para = np.ones_like(camou_para)*0.8
    camou_para = torch.from_numpy(camou_para).to(device)

    #扩展卷积核定义
    resolution = 8
    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)

    #应用扩展卷积核
    camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    camou_para1 = torch.clamp(camou_para1, 0, 1)

    data_dir = '/data/zjh/mde_carla/'
    img_size = (320, 1024)
    
    dataset = MyDataset(data_dir, img_size, obj_name, camou_mask, device=device, phy_trans_flag=True)
    dataset.set_textures(camou_para1)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        #num_workers=2,
    )

    log_dir = './assets/'
    tqdm_loader = tqdm(loader)
    for i, (index, total_img, total_img0, mask, img, imgs_pred, imgs_pred0) in enumerate(tqdm_loader):
        index = int(index[0])
        
        total_img_np = total_img.data.cpu().numpy()[0] * 255
        # print(total_img_np.shape)
        total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))
        total_img_np.save(os.path.join(log_dir, str(i)+'test_total.jpg'))

        total_img_np0 = total_img0.data.cpu().numpy()[0] * 255
        # print(total_img_np0.shape)
        total_img_np0 = Image.fromarray(np.transpose(total_img_np0, (1,2,0)).astype('uint8'))
        total_img_np0.save(os.path.join(log_dir, str(i)+'test_total0.jpg'))
        # print(mask.shape)

        Image.fromarray((255 * imgs_pred).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, str(i)+'img_pred.png'))
        Image.fromarray((255 * imgs_pred0).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, str(i)+'img_pred0.png'))
        Image.fromarray((255 * img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, str(i)+'img.png'))
        Image.fromarray((255 * mask).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, str(i)+'mask.png'))
        if i >= 11:
            break
