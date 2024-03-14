import torch
import torch.nn as nn
from torch.nn.init import normal_, constant_
from model.basic_ops import ConsensusModule
import numpy as np

import sys
from importlib import import_module
sys.path.append('..')

class VideoNet(nn.Module):
	def __init__(self, num_class, num_segments, modality,
				backbone='ViT-B_16', net=None, consensus_type='avg',
				dropout=0.0, partial_bn=False, print_spec=True, pretrain='imagenet',
				is_shift=False, fold_div=8,
				drop_block=0, vit_img_size=224,
				vit_pretrain="", LayerNormFreeze=2, cfg=None):
		super(VideoNet, self).__init__()
		self.num_segments = num_segments
		self.modality = modality
		self.backbone = backbone
		self.net = net
		self.dropout = dropout
		self.pretrain = pretrain
		self.consensus_type = consensus_type
		self.drop_block = drop_block
		self.init_crop_size = 256
		self.vit_img_size=vit_img_size
		self.vit_pretrain=vit_pretrain

		self.is_shift = is_shift
		self.fold_div = fold_div
		self.backbone = backbone
		
		self.num_class = num_class
		self.cfg = cfg
		self._prepare_base_model(backbone)
		if "resnet" in self.backbone:
			self._prepare_fc(num_class)
		self.consensus = ConsensusModule(consensus_type)
		#self.softmax = nn.Softmax()
		self._enable_pbn = partial_bn
		self.LayerNormFreeze = LayerNormFreeze
		if partial_bn:
			self.partialBN(True)

	def _prepare_base_model(self, backbone):

		if 'vit' in backbone:
			if self.net == 'ViT':
				print('=> base model: ViT, with backbone: {}'.format(backbone))
				from timm.models import vision_transformer
				self.base_model = getattr(vision_transformer, backbone)(pretrained=True, num_classes=self.num_class)
			elif self.net == 'TokShift':
				print('=> base model: TokShift, with backbone: {}'.format(backbone))
				from timm.models import tokshift_xfmr
				self.base_model = getattr(tokshift_xfmr, backbone)(pretrained=True, num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			elif self.net == 'ViT_LAPS':
				print('=> base model: ViT_LAPS, with backbone: {}'.format(backbone))
				from timm.models import vit_laps
				self.base_model = getattr(vit_laps, backbone)(pretrained=True, num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div, s2_skip_level=self.cfg.MODEL.S2_SKIP_LEVEL)
			self.feature_dim = self.num_class
		elif 'rest' in backbone:
			if self.net == 'Rest':
				print('=> base model: Rest, with backbone: {}'.format(backbone))
				from Rest import rest
				self.base_model = getattr(rest, backbone)(num_classes=self.num_class)
			if self.net == 'Rest_base3D':
				print('=> base model: Rest_base3D, with backbone: {}'.format(backbone))
				from Rest import rest_base3D
				self.base_model = getattr(rest_base3D, backbone)(num_classes=self.num_class)
			if self.net == 'Rest_cshift_SA':
				print('=> base model: Rest_cshift_SA, with backbone: {}'.format(backbone))
				from Rest import rest_cshift_SA
				self.base_model = getattr(rest_cshift_SA, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div, s2_skip_level=self.cfg.MODEL.S2_SKIP_LEVEL)
			
		elif 'visformer' in backbone:
			if self.net == 'Visformer':
				print('=> base model: Visformer, with backbone: {}'.format(backbone))
				from visformer import models
				self.base_model = getattr(models, backbone)(num_classes=self.num_class)
			elif self.net == 'Visformer_base3D':
				print('=> base model: Visformer_base3D, with backbone: {}'.format(backbone))
				from visformer import models_base3D
				self.base_model = getattr(models_base3D, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			
			elif self.net == 'Visformer_plain_shift':
				print('=> base model: Visformer_plain_shift, with backbone: {}'.format(backbone))
				from visformer import models_plain_shift
				self.base_model = getattr(models_plain_shift, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			elif self.net == 'Visformer_PS':
				print('=> base model: Visformer_PS, with backbone: {}'.format(backbone))
				from visformer import models_periodic_shift
				self.base_model = getattr(models_periodic_shift, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div)
			elif self.net == 'Visformer_LAPS':
				print('=> base model: Visformer_LAPS, with backbone: {}'.format(backbone))
				from visformer import models_LAPS
				self.base_model = getattr(models_LAPS, backbone)(num_classes=self.num_class, n_seg=self.num_segments, fold_div=self.fold_div, 
						s2_skip_level=self.cfg.MODEL.S2_SKIP_LEVEL, 
						s3_skip_level=self.cfg.MODEL.S3_SKIP_LEVEL)
		elif 'nat_' in backbone:
			if self.net == 'nat':
				print('=> base model: natformer, with backbone: {}'.format(backbone))
				from natformer import nat
				self.base_model = getattr(nat, backbone)(num_classes=self.num_class, pretrained=True)
			elif self.net == 'nat_simple_tct_shift':
				print('=> base model: nat_simple_tct_shift, with backbone: {}'.format(backbone))
				from natformer import nat_simple_tct_shift
				self.base_model = getattr(nat_simple_tct_shift, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_simple_tct_shift_pad':
				print('=> base model: nat_simple_tct_shift_pad, with backbone: {}'.format(backbone))
				from natformer import nat_simple_tct_shift_pad
				self.base_model = getattr(nat_simple_tct_shift_pad, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_simple_tct_roll':
				print('=> base model: nat_simple_tct_roll, with backbone: {}'.format(backbone))
				from natformer import nat_simple_tct_roll
				self.base_model = getattr(nat_simple_tct_roll, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat3D':
				print('=> base model: nat3D, with backbone: {}'.format(backbone))
				from natformer import nat3D
				self.base_model = getattr(nat3D, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat3D_gt':
				print('=> base model: nat3D_gt, with backbone: {}'.format(backbone))
				from natformer import nat3D_gt
				self.base_model = getattr(nat3D_gt, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat3D_GlobalT':
				print('=> base model: nat3D_GlobalT, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT
				self.base_model = getattr(nat3D_GlobalT, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat3D_GlobalT_RmDup':
				print('=> base model: nat3D_GlobalT_RmDup, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_RmDup
				self.base_model = getattr(nat3D_GlobalT_RmDup, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat3D_GlobalT_DownSpat_RmDup':
				print('=> base model: nat3D_GlobalT_DownSpat_RmDup, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_DownSpat_RmDup
				self.base_model = getattr(nat3D_GlobalT_DownSpat_RmDup, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT_ConcentrateContx_RmDup':
				print('=> base model: nat3D_GlobalT_ConcentrateContx_RmDup, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_ConcentrateContx_RmDup
				self.base_model = getattr(nat3D_GlobalT_ConcentrateContx_RmDup, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT_CondenseContext_RmDup':
				print('=> base model: nat3D_GlobalT_CondenseContext_RmDup, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_CondenseContext_RmDup
				self.base_model = getattr(nat3D_GlobalT_CondenseContext_RmDup, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT_CondenseContext_RmDup_SE':
				print('=> base model: nat3D_GlobalT_CondenseContext_RmDup_SE, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_CondenseContext_RmDup_SE
				self.base_model = getattr(nat3D_GlobalT_CondenseContext_RmDup_SE, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT_CondenseContext_RmDup_SE2':
				print('=> base model: nat3D_GlobalT_CondenseContext_RmDup_SE2, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_CondenseContext_RmDup_SE2
				self.base_model = getattr(nat3D_GlobalT_CondenseContext_RmDup_SE2, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT_CondenseContext_RmDup_SE3':
				print('=> base model: nat3D_GlobalT_CondenseContext_RmDup_SE3, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_CondenseContext_RmDup_SE3
				self.base_model = getattr(nat3D_GlobalT_CondenseContext_RmDup_SE3, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT_CondenseContext_RmDup_shift':
				print('=> base model: nat3D_GlobalT_CondenseContext_RmDup_shift, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_CondenseContext_RmDup_shift
				self.base_model = getattr(nat3D_GlobalT_CondenseContext_RmDup_shift, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT_CondenseContext_RmDup_shiftpad':
				print('=> base model: nat3D_GlobalT_CondenseContext_RmDup_shiftpad, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT_CondenseContext_RmDup_shiftpad
				self.base_model = getattr(nat3D_GlobalT_CondenseContext_RmDup_shiftpad, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat3D_GlobalT2_CondenseContext_RmDup':
				print('=> base model: nat3D_GlobalT2_CondenseContext_RmDup, with backbone: {}'.format(backbone))
				from natformer import nat3D_GlobalT2_CondenseContext_RmDup
				self.base_model = getattr(nat3D_GlobalT2_CondenseContext_RmDup, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments, down_spat=self.cfg.MODEL.NeAT3DPlus_DOWN_SPAT)
			elif self.net == 'nat_Dilated3D_hybrid_GlobalT':
				print('=> base model: nat_Dilated3D_hybrid_GlobalT, with backbone: {}'.format(backbone))
				from natformer import nat_Dilated3D_hybrid_GlobalT
				self.base_model = getattr(nat_Dilated3D_hybrid_GlobalT, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_Dilated3D_hybrid_GlobalT_RmDup':
				print('=> base model: nat_Dilated3D_hybrid_GlobalT_RmDup, with backbone: {}'.format(backbone))
				from natformer import nat_Dilated3D_hybrid_GlobalT_RmDup
				self.base_model = getattr(nat_Dilated3D_hybrid_GlobalT_RmDup, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_Shrink3D_GlobalT_RmDup':
				print('=> base model: nat_Shrink3D_GlobalT_RmDup, with backbone: {}'.format(backbone))
				from natformer import nat_Shrink3D_GlobalT_RmDup
				self.base_model = getattr(nat_Shrink3D_GlobalT_RmDup, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_2FrontD':
				print('=> base model: nat_2FrontD, with backbone: {}'.format(backbone))
				from natformer import nat_2FrontD
				self.base_model = getattr(nat_2FrontD, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_2BackD':
				print('=> base model: nat_2BackD, with backbone: {}'.format(backbone))
				from natformer import nat_2BackD
				self.base_model = getattr(nat_2BackD, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_Dilated3D':
				print('=> base model: nat_Dilated3D, with backbone: {}'.format(backbone))
				from natformer import nat_Dilated3D
				self.base_model = getattr(nat_Dilated3D, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_Dilated3D_hybrid':
				print('=> base model: nat_Dilated3D_hybrid, with backbone: {}'.format(backbone))
				from natformer import nat_Dilated3D_hybrid
				self.base_model = getattr(nat_Dilated3D_hybrid, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_simple_tct_roll_split':
				print('=> base model: nat_simple_tct_roll_split, with backbone: {}'.format(backbone))
				from natformer import nat_simple_tct_roll_split
				self.base_model = getattr(nat_simple_tct_roll_split, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_seq_tct_roll':
				print('=> base model: nat_seq_tct_roll, with backbone: {}'.format(backbone))
				from natformer import nat_seq_tct_roll
				self.base_model = getattr(nat_seq_tct_roll, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_fuse_tct_roll':
				print('=> base model: nat_fuse_tct_roll, with backbone: {}'.format(backbone))
				from natformer import nat_fuse_tct_roll
				self.base_model = getattr(nat_fuse_tct_roll, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_bridge_fuse_tct_roll':
				print('=> base model: nat_bridge_fuse_tct_roll, with backbone: {}'.format(backbone))
				from natformer import nat_bridge_fuse_tct_roll
				self.base_model = getattr(nat_bridge_fuse_tct_roll, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			elif self.net == 'nat_fuse_motion_tct_roll':
				print('=> base model: nat_fuse_motion_tct_roll, with backbone: {}'.format(backbone))
				from natformer import nat_fuse_motion_tct_roll
				self.base_model = getattr(nat_fuse_motion_tct_roll, backbone)(num_classes=self.num_class, pretrained=True, n_seg=self.num_segments)
			#######
			self.feature_dim = self.num_class

		else:
			raise ValueError('Unknown backbone: {}'.format(backbone))


	def _prepare_fc(self, num_class):
		if self.dropout == 0:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(self.feature_dim, num_class))
			self.new_fc = None
		else:
			setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
			self.new_fc = nn.Linear(self.feature_dim, num_class)

		std = 0.001
		if self.new_fc is None:
			normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
			constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
		else:
			if hasattr(self.new_fc, 'weight'):
				normal_(self.new_fc.weight, 0, std)
				constant_(self.new_fc.bias, 0)

	#
	def train(self, mode=True):
		# Override the default train() to freeze the BN parameters
		super(VideoNet, self).train(mode)
		count = 0
		if self._enable_pbn and mode:
			print("Freezing LayerNorm.")
			for m in self.base_model.modules():
				if isinstance(m, nn.LayerNorm):
					count += 1
					if count >= (self.LayerNormFreeze if self._enable_pbn else 1):
						m.eval()
						print("Freeze {}".format(m))
						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False


	#
	def partialBN(self, enable):
		self._enable_pbn = enable


	def forward(self, input, peframe=False):
		# input size [batch_size, num_segments, 3, h, w]
		b, t, c, h, w = input.shape
		input = input.view((-1, 3) + input.size()[-2:])
		base_out = self.base_model(input)
		base_out = base_out.view((b, -1)+base_out.size()[1:])
		#print("Baseout {}".format(base_out.shape))
		#print(base_out[0,:,1:10])
		#
		output = self.consensus(base_out)

		if peframe:
			return base_out
		else:
			return output.squeeze(1)

