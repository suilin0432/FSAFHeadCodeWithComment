import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead


import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses import sigmoid_focal_loss, FocalLoss, iou_loss
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
import pdb


@HEADS.register_module
class FSAFHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 feat_strides=[8, 16, 32, 64, 128],
                 eps_e=0.2,
                 eps_i=0.5,
                 FL_alpha=0.25,
                 FL_gamma=2.0,
                 bbox_offset_norm=4,
                 **kwargs):
        """
        :param num_classes: 物体分类类别
        :param in_channels: 输入进来的特征图的 channel 大小
        :param stacked_convs: (根据retinaNet论文) 指的是 分类 和 bbox regression 分支中堆叠了的 conv 的数量
        :param octave_base_scale: (根据retinaNet论文) 指的是每个位置或设置多少个 正方形的 anchor
        :param scales_per_octave: (根据retinaNet论文)是每个anchor的aspect ratio个数
        :param conv_cfg: 卷积层的配置信息
        :param norm_cfg: 正则化层的信息
        :param feat_strides: feature pyramid 的各层次相对于原图的步长
        # 上面的参数全部都是 retina_head 中使用了的参数, 下面的参数是针对于 FSAF 的
        :param eps_e: 论文中提到了的类别分类的那个 作为positive类别的范围的参数
        :param eps_i: 论文中提到的忽略部分的参数
        :param FL_alpha: focal loss的 alpha 参数
        :param FL_gamma: focal loss的 gamma 参数
        :param bbox_offset_norm: bbox regression 过程中要除以的那个 S 参数
        :param kwargs:
        """
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # 按照 retinaNet 论文的描述, 每个位置会设置 4 个正方形 anchor, 大小分别是 2^0, 2^(1/4), 2^(1/2), 2^(3/4) 单位大小
        # PS: 这是 FSAF 的复现啊, 每个点的位置只会产生一个 "anchor"(引号表明实际上没有anchor...) 啊
        # 所以这个值的设置是没用的... 只是复制时候复制过来的...
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        # anchor 一共的数量
        anchor_scales = octave_scales * octave_base_scale

        # 下面还是各种FSAF新增参数的简单赋值
        self.feat_strides = feat_strides
        self.eps_e = eps_e
        self.eps_i = eps_i
        self.FL_alpha = FL_alpha
        self.FL_gamma = FL_gamma
        self.bbox_offset_norm = bbox_offset_norm
        
        super(FSAFHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)
    # FSAF Head 层次的构建
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # cls 和 reg 分支都堆叠了 4 个 conv 层
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        # 两个 branch 的最后一层
        # cls_out_channel 是在 AnchorHead 类中进行赋值的, 其值是和类别相关的
        # 因为 FSAF 的 feature map 的每个 pixel 只有一个对应的 "anchor", 所以输出通道并不需要乘以 anchor 数目
        self.fsaf_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        # FSAF 的预测的 output 是 top, left, bottom, right 的距离, 是 class-agnostic 的
        self.fsaf_reg = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)

    # 进行参数的初始化
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fsaf_cls, std=0.01, bias=bias_cls)
        normal_init(self.fsaf_reg, std=0.1)
        
    # 前向过程, 就是简单的两个分支走一遍就好了, 注意这个是 forward_single, 因为 原本的 x 因为多尺度的 feature map 实际上是一个 tuple
    # 会用 forward 函数进行分发到 forward_single 中
    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        cls_score = self.fsaf_cls(cls_feat)
        bbox_pred = self.relu(self.fsaf_reg(reg_feat))
        
        return cls_score, bbox_pred
    
    # per-level loss operation
    # 这个部分是核心部分 2333, 后面着重看一下
    def loss_single(self,
             cls_scores,
             bbox_preds,
             cls_targets_list,
             reg_targets_list,
             level,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        """
        :param cls_scores: classification map
        :param bbox_preds: bbox regression map
        :param cls_targets_list: target classificaion map
        :param reg_targets_list: target bbox regression map
        :param level: current heatmap stage level
        :param img_metas: 图像的 img_metas 信息 (这里没啥用)
        :param cfg: 配置 dict (这里没啥用)
        :param gt_bboxes_ignore: 忽略的 gt bbox (这里没啥用)
        :return:
        """
        device = cls_scores[0].device
        # 获取 图片的数量
        num_imgs = cls_targets_list.shape[0]
        
        # loss-cls
        # 调整 channel -> 最后一个维度, 然后 reshape -> (imgNum * height * width, channelNum)
        # PS: 好像其实并不用后面的 reshape 就可以...
        scores = cls_scores.permute(0,2,3,1).reshape(-1,1)
        labels = cls_targets_list.permute(0,2,3,1).reshape(-1)
        # 提取出所有非 ignore 的部分
        valid_cls_idx = labels != -1
        # 提取出这个非 ignore 的部分进行 focal_loss 的计算
        loss_cls = sigmoid_focal_loss(scores[valid_cls_idx], labels[valid_cls_idx],
                           gamma=self.FL_gamma, alpha=self.FL_alpha, reduction='sum')
        # 进行所有 label 为 1(即非ignore) 的数量 (后续作为加权平均的参数吧应该)
        norm_cls = (labels == 1).long().sum()
        
        # loss-reg
        # 进行 loss-reg 的计算
        offsets = bbox_preds.permute(0,2,3,1).reshape(-1,4)
        gtboxes = reg_targets_list.permute(0,2,3,1).reshape(-1,4)
        valid_reg_idx = (gtboxes[:,0] != -1)
        if valid_reg_idx.long().sum() != 0:
            offsets = offsets[valid_reg_idx]
            gtboxes = gtboxes[valid_reg_idx]
            
            H,W = bbox_preds.shape[2:]
            y,x = torch.meshgrid([torch.arange(0,H), torch.arange(0,W)])
            y = (y.float() + 0.5) * self.feat_strides[level]
            x = (x.float() + 0.5) * self.feat_strides[level]
            xy = torch.cat([x.unsqueeze(2),y.unsqueeze(2)], dim=2).float().to(device)
            xy = xy.permute(2,0,1).unsqueeze(0).repeat(num_imgs,1,1,1)
            xy = xy.permute(0,2,3,1).reshape(-1,2)
            xy = xy[valid_reg_idx]
            
            dist_pred = offsets * self.feat_strides[level]
            bboxes = self.dist2bbox(xy, dist_pred, self.bbox_offset_norm)
            
            loss_reg = iou_loss(bboxes, gtboxes, reduction='sum')
            norm_reg = valid_reg_idx.long().sum()
        else:
            loss_reg = torch.tensor(0).float().to(device)
            norm_reg = torch.tensor(0).float().to(device)
            
        return loss_cls, loss_reg, norm_cls, norm_reg

    # 进行 loss 计算的分支, 核心步骤会在 single_loss 中完成
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        # PS: 这里的 cls_scores 什么的都是 多 stage 的信息

        # 首先会调用 fsaf_target 函数去进行 cls_reg_targets 的获取
        # cls_reg_targets 的作用是 获取到 classification 与 bbox_regression 的 训练的 target
        cls_reg_targets = self.fsaf_target(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore)
        # 将两个 target 拆出来
        (cls_targets_list, reg_targets_list) = cls_reg_targets
        # 产生一个 stage 对应表, 提供 stage_index, 方便在 single_loss 中从 self.feat_strides 中获取步长信息
        level_list = [i for i in range(len(self.feat_strides))]
        # 使用 self.loss_single 对每个 stage 的信息进行计算
        loss_cls, loss_reg, norm_cls, norm_reg = multi_apply(
             self.loss_single,
             cls_scores,
             bbox_preds,
             cls_targets_list,
             reg_targets_list,
             level_list,
             img_metas=img_metas,
             cfg=cfg,
             gt_bboxes_ignore=None)
        
        loss_cls = sum(loss_cls)/sum(norm_cls)
        loss_reg = sum(loss_reg)/sum(norm_reg)
        
        return dict(loss_cls=loss_cls, loss_bbox=loss_reg)


    def fsaf_target(self,
                     cls_scores,
                     bbox_preds,
                     gt_bboxes,
                     gt_labels,
                     img_metas,
                     cfg,
                     gt_bboxes_ignore_list=None):
        """
        :param cls_scores: 多个 stage 中 bbox forward 得到的分类的分数, 每个entry对应的 shape : classNum * height * width
        :param bbox_preds: 多个 stage 中 bbox forward 得到的bbox regression的结果, 每个entry对应的 shape : 4 * height * width
        :param gt_bboxes: GT bboxes 的信息
        :param gt_labels: GT label 的信息
        # 下面的参数都没有用到... 不做说明了
        :param img_metas:
        :param cfg:
        :param gt_bboxes_ignore_list:
        :return:
        """
        # 首先获取设备类型a, 方便后续进行计算的时候设备类型的统一
        device = cls_scores[0].device
        
        # target placeholder(记录target的两个list)
        num_levels = len(cls_scores)
        cls_targets_list = []
        reg_targets_list = []
        # 首先进行初始化 对每个阶段都进行一下初始化, 每个阶段都用其形状的全 0/1 Tensor 进行初始化
        for level in range(num_levels):
            cls_targets_list.append(torch.zeros_like(cls_scores[level]).long()) #  0 init
            reg_targets_list.append(torch.ones_like(bbox_preds[level]) * -1)    # -1 init
        
        # detached network prediction for online GT generation
        # 获取 图片长度(每个img 的 bboxes 对应 gt_bboxes 中的一个维度)
        num_imgs = len(gt_bboxes)
        # 获取 进行了 detach 的 cls_scores 与 bbox_preds
        cls_scores_list = []
        bbox_preds_list = []
        for img in range(num_imgs):
            # detached prediction for online pyramid level selection
            cls_scores_list.append([lvl[img].detach() for lvl in cls_scores])
            bbox_preds_list.append([lvl[img].detach() for lvl in bbox_preds])

        # 开始进行 GT 匹配
        # generate online GT
        num_imgs = len(gt_bboxes)
        for img in range(num_imgs):
            # sort objects according to their size
            # 取出来所有的 gt_bboxes 信息, 这个时候取出来的是 (x1, y1, x2, y2) 的形式
            gt_bboxes_img_xyxy = gt_bboxes[img]
            # 将 (x1, y1, x2, y2) 形式转化为 (x_center, y_center, width, height) 的形式 em... 直接给变成 int 类型的结果了...
            # 同时这步转换会将 gt_bboxes_img_xywh 转变为 Tensor 类型
            gt_bboxes_img_xywh = self.xyxy2xywh(gt_bboxes_img_xyxy)
            # 获取 GT Bbox 的 size
            gt_bboxes_img_size = gt_bboxes_img_xywh[:,-2] * gt_bboxes_img_xywh[:,-1]
            # 获取 gt_bboxes_img_size 的排序后顺序对应的 index, 从大到小进行排序
            _, gt_bboxes_img_idx = gt_bboxes_img_size.sort(descending=True)
            # 对每个 GT bbox 进行一定操作
            for obj_idx in gt_bboxes_img_idx:
                # 因为看了配置文件, 选择使用 sigmoid 对 cls 的预测分数进行激活, 是不要 bg 这个类别的(猜测是完全依靠阈值进行排除,
                # 因为没有 softmax 更改了其预测的比例)
                label = gt_labels[img][obj_idx]-1
                # 获取这个 gt bbox 的形状信息
                gt_bbox_obj_xyxy = gt_bboxes_img_xyxy[obj_idx]
                # get optimal online pyramid level for each object
                # 获取这个 bbox 应该被分配的 stage level
                # 这个传递进去的 cls_scores_list[img], bbox_preds_list[img] 是用来读取信息的... 这种传递引用过去的如果进行修改了那就真的修改了...
                opt_level = self.get_online_pyramid_level(
                    cls_scores_list[img], bbox_preds_list[img], gt_bbox_obj_xyxy, label)

                # 获取到分类的 effective 和 ignore 区域
                # get the effective/ignore area
                # 获取当前 stage 的 height 和 width
                H,W = cls_scores[opt_level].shape[2:]
                # 获取这个 gt anchor 在分配到的 stage level 的实际大小(相对于 feature map 的 grid 的大小)
                # 因为 gt bbox 信息都是基于原图大小的信息, 所以我们获取之后要在特定层次上放缩到对应的比例
                b_p_xyxy = gt_bbox_obj_xyxy / self.feat_strides[opt_level]
                # 使用 get_spatial_idx 对 b_p_xyxy 进行处理 获取到空间上 effective 和 ignore 的空间区域 的 mask
                e_spatial_idx, i_spatial_idx = self.get_spatial_idx(b_p_xyxy,W,H,device)
                
                # cls-GT
                # fill prob= 1 for the effective area
                # cls 的 effective 区域进行赋值为 1
                cls_targets_list[opt_level][img, label, e_spatial_idx] = 1
                
                # fill prob=-1 for the ignoring area
                # 对 cls 的 ignore 区域进行赋值为 -1
                # 这个步骤是为了防止 ignore 直接将重叠了的 gt 区域覆盖为 ignore 区域了而设置的操作
                _i_spatial_idx = cls_targets_list[opt_level][img, label] * i_spatial_idx.long()
                i_spatial_idx = i_spatial_idx - (_i_spatial_idx == 1)
                cls_targets_list[opt_level][img, label, i_spatial_idx] = -1                
                
                # fill prob=-1 for the adjacent ignoring area; lower
                # 向下进行邻近层次的 ignoring 区域的填充
                if opt_level != 0:
                    H_l,W_l = cls_scores[opt_level-1].shape[2:]
                    b_p_xyxy_l = gt_bbox_obj_xyxy / self.feat_strides[opt_level-1]
                    _, i_spatial_idx_l = self.get_spatial_idx(b_p_xyxy_l,W_l,H_l,device)
                    # preserve cls-gt that is already filled as effective area
                    _i_spatial_idx_l = cls_targets_list[opt_level-1][img, label] * i_spatial_idx_l.long()
                    i_spatial_idx_l = i_spatial_idx_l - (_i_spatial_idx_l == 1)
                    cls_targets_list[opt_level-1][img, label][i_spatial_idx_l] = -1
                    
                # fill prob=-1 for the adjacent ignoring area; upper
                # 向上进行临近层次的 ignoring 区域的填充
                if opt_level != num_levels-1:
                    H_u,W_u = cls_scores[opt_level+1].shape[2:]
                    b_p_xyxy_u = gt_bbox_obj_xyxy / self.feat_strides[opt_level+1]
                    _, i_spatial_idx_u = self.get_spatial_idx(b_p_xyxy_u,W_u,H_u,device)
                    # preserve cls-gt that is already filled as effective area
                    _i_spatial_idx_u = cls_targets_list[opt_level+1][img, label] * i_spatial_idx_u.long()
                    i_spatial_idx_u = i_spatial_idx_u - (_i_spatial_idx_u == 1)
                    cls_targets_list[opt_level+1][img, label][i_spatial_idx_u] = -1
                
                # reg-GT
                reg_targets_list[opt_level][img, :, e_spatial_idx] = gt_bbox_obj_xyxy.unsqueeze(1)
                
        return cls_targets_list, reg_targets_list
        
    def get_spatial_idx(self, b_p_xyxy, W, H, device):
        # zero-tensor w/ (H,W)
        # 首先进行全0的初始化
        e_spatial_idx = torch.zeros((H,W)).byte()
        i_spatial_idx = torch.zeros((H,W)).byte()
        
        # effective idx
        # 获取到 effective 的区域范围参数
        b_e_xyxy = self.get_prop_xyxy(b_p_xyxy, self.eps_e, W, H)
        e_xmin = b_e_xyxy[0]
        e_xmax = b_e_xyxy[2]+1
        e_ymin = b_e_xyxy[1]
        e_ymax = b_e_xyxy[3]+1
        # 进行范围的赋值
        e_spatial_idx[e_ymin:e_ymax, e_xmin:e_xmax] = 1
        
        # ignore idx
        # 同理对 ignore 区域进行处理, 但是要排除掉 effective 的部分
        b_i_xyxy = self.get_prop_xyxy(b_p_xyxy, self.eps_i, W, H)
        i_xmin = b_i_xyxy[0]
        i_xmax = b_i_xyxy[2]+1
        i_ymin = b_i_xyxy[1]
        i_ymax = b_i_xyxy[3]+1
        i_spatial_idx[i_ymin:i_ymax, i_xmin:i_xmax] = 1
        i_spatial_idx[e_ymin:e_ymax, e_xmin:e_xmax] = 0
        
        return e_spatial_idx.to(device), i_spatial_idx.to(device)
        
    def get_online_pyramid_level(self, cls_scores_img, bbox_preds_img, gt_bbox_obj_xyxy, gt_label_obj):
        """
        :param cls_scores_img: 这个图片对应的 cls_scores 信息(也是多个 stage 的信息)
        :param bbox_preds_img: 这个图片对应的 bbox_preds 信息(也是多个 stage 的信息)
        :param gt_bbox_obj_xyxy: 要进行层次选择的 gt bbox 的信息 (x1, y1, x2, y2) 格式
        :param gt_label_obj: 要进行层次选择的 gt bbox 的 label(类别) 信息
        :return:
        """
        device = cls_scores_img[0].device
        # 获取 stage 数量
        num_levels = len(cls_scores_img)
        # 为每个层次的 loss 创建一下统计求和的 placeholder
        level_losses = torch.zeros(num_levels)
        # 开始对每个层次进行统计
        for level in range(num_levels):
            # 获取 feature map 的 长宽
            H,W = cls_scores_img[level].shape[1:]
            # 获取在 feature map 上的 grid 位置
            b_p_xyxy = gt_bbox_obj_xyxy / self.feat_strides[level]
            # 获取 effective 的范围
            b_e_xyxy = self.get_prop_xyxy(b_p_xyxy, self.eps_e, W, H)
            
            # Eqn-(1)
            # 获取 effective 区域的计数个数
            N = (b_e_xyxy[3]-b_e_xyxy[1]+1) * (b_e_xyxy[2]-b_e_xyxy[0]+1)
            
            # cls loss; FL
            # 开始计算 focal loss -> classification score loss
            # 首先获取到 effective 区域的 cls_score 的值
            score = cls_scores_img[level][gt_label_obj,b_e_xyxy[1]:b_e_xyxy[3]+1,b_e_xyxy[0]:b_e_xyxy[2]+1]
            # 把score reshape -> (1, N)
            score = score.contiguous().view(-1).unsqueeze(1)
            # 设置 label 为与 score 相同形状 (1, N) 的全 1 Tensor
            label = torch.ones_like(score).long()
            label = label.contiguous().view(-1)
            # 计算 sigmoid 之后的 focal_loss
            # label 是 weight
            loss_cls = sigmoid_focal_loss(score, label, gamma=self.FL_gamma, alpha=self.FL_alpha, reduction='mean')
            # 因为已经在 loss 函数中有 "mean" 了... 所以不用除以 N 了
            #loss_cls /= N

            # 开始计算 Bbox regression loss
            # reg loss; IoU
            # 获取到区域内的 bbox 信息
            offsets = bbox_preds_img[level][:,b_e_xyxy[1]:b_e_xyxy[3]+1,b_e_xyxy[0]:b_e_xyxy[2]+1]
            # 调整维度顺序, 从 (channel * height * width) -> (height * width * channel)
            offsets = offsets.contiguous().permute(1,2,0)  # (b_e_H,b_e_W,4)
            # reshape -> (height * width *  channel)
            offsets = offsets.reshape(-1,4) # (#pix-e,4)
            # PS: 上面拿到的 offsets 中每个 offset 实际上是预测的是到 上下左右四条边 的距离
            
            # predicted bbox
            # 首先产生 feature map 指定区域的 网格位置信息 -> 要生成目标的 "anchor" 区域的
            y,x = torch.meshgrid([torch.arange(b_e_xyxy[1],b_e_xyxy[3]+1), torch.arange(b_e_xyxy[0],b_e_xyxy[2]+1)])
            # 获取到这个位置上的中心点坐标 (相对于input的原图大小)
            y = (y.float() + 0.5) * self.feat_strides[level]
            x = (x.float() + 0.5) * self.feat_strides[level]
            # 拼接中心位置
            xy = torch.cat([x.unsqueeze(2),y.unsqueeze(2)], dim=2).float().to(device)
            xy = xy.reshape(-1,2)

            # 将 offsets -> 原图的尺度上
            dist_pred = offsets * self.feat_strides[level]
            # 进行变换获取到 bboxes 信息
            bboxes = self.dist2bbox(xy, dist_pred, self.bbox_offset_norm)
            # loss_reg 通过 iou_loss 进行计算
            loss_reg = iou_loss(bboxes, gt_bbox_obj_xyxy.unsqueeze(0).repeat(N,1), reduction='mean')
            # 同样是因为 reduction = mean 所以不需要 /= N
            # PS: /= 操作好像是不行的... pytorch 要 loss_reg = loss_reg / N 这样的操作
            #loss_reg /= N
            # 计算当前的 stage 的 loss
            loss = loss_cls + loss_reg

            level_losses[level] = loss
        # 找到最小的 loss 的区域, 然后返回这个维度的 loss 信息
        min_level_idx = torch.argmin(level_losses)
        #print(level_losses, min_level_idx)
        return min_level_idx
            
    def get_prop_xyxy(self, xyxy, scale, w, h):
        # scale bbox
        xywh = self.xyxy2xywh(xyxy)
        xywh[2:] *= scale
        xyxy = self.xywh2xyxy(xywh)
        # clamp bbox
        xyxy[0] = xyxy[0].floor().clamp(0, w-2).int() # x1
        xyxy[1] = xyxy[1].floor().clamp(0, h-2).int() # y1
        xyxy[2] = xyxy[2].ceil().clamp(1, w-1).int()  # x2
        xyxy[3] = xyxy[3].ceil().clamp(1, h-1).int()  # y2
        return xyxy.int()
    
    def xyxy2xywh(self, xyxy):
        if xyxy.dim() == 1:
            return torch.cat((0.5 * (xyxy[0:2] + xyxy[2:4]), xyxy[2:4] - xyxy[0:2]), dim=0)
        else:
            return torch.cat((0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]), dim=1)
    
    def xywh2xyxy(self, xywh):
        if xywh.dim() == 1:
            return torch.cat((xywh[0:2] - 0.5 * xywh[2:4], xywh[0:2] + 0.5 * xywh[2:4]), dim=0)
        else:
            return torch.cat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4], xywh[:, 0:2] + 0.5 * xywh[:, 2:4]), dim=1)
        
    def dist2bbox(self, center, dist_pred, norm_const):
        '''
        args
            center    ; (N,2)
            bbox_pred ; (N,4)
        '''
        x1y1 = center - (norm_const * torch.cat([dist_pred[:,1].unsqueeze(1),dist_pred[:,0].unsqueeze(1)],dim=1))
        x2y2 = center + (norm_const * torch.cat([dist_pred[:,3].unsqueeze(1),dist_pred[:,2].unsqueeze(1)],dim=1))
        bbox = torch.cat([x1y1,x2y2],dim=1)
        return bbox
        
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        # note: only single-img evaluation is available now
        num_levels = len(cls_scores)
        assert len(cls_scores) == len(bbox_preds) == num_levels
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype
        scale_factor = img_metas[0]['scale_factor']
        
        # generate center-points
        xy_list = []
        for level in range(num_levels):
            H,W = bbox_preds[level].shape[2:]
            y,x = torch.meshgrid([torch.arange(0,H), torch.arange(0,W)])
            y = (y.float() + 0.5) * self.feat_strides[level]
            x = (x.float() + 0.5) * self.feat_strides[level]
            xy = torch.cat([x.unsqueeze(2),y.unsqueeze(2)], dim=2).float().to(device)
            xy = xy.permute(2,0,1).unsqueeze(0)
            xy_list.append(xy)
            
        mlvl_bboxes = []
        mlvl_scores = []
        for level, (cls_score, bbox_pred, xy) in enumerate(zip(cls_scores, bbox_preds, xy_list)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score[0].permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4)
            xy = xy[0].permute(1, 2, 0).reshape(-1, 2)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                xy = xy[topk_inds, :]
            
            # decode predicted bbox offsets to get final bbox
            dist_pred = bbox_pred * self.feat_strides[level]
            bboxes = self.dist2bbox(xy, dist_pred, self.bbox_offset_norm)
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return [(det_bboxes, det_labels)]
    
    
        
    
    
    
    
    

