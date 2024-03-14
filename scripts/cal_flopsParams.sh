#!/usr/bin/env python
import os
os.system('export CUDA_VISIBLE_DEVICES=0')

cmd = "python -u cal_flops.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23678 \
		--eval \
		--cfg_file  config/custom/k400/nat/paper_run/nat3D_small_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_large_GlobalT_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_base_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_base_GlobalT_CondenseContext_RmDup_8x8_320_DownSpat[11,5.5,2.6,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_32x2_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_base_GlobalT_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat_2FrontD_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_SE3_8x8_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_SE_8x8_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_16x8_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_shiftpad_8x8_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat2D_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT2_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x8_DownSpat[4,4,2,0].yaml"
		#--cfg_file  config/custom/k400/nat/paper_run/nat3D_GlobalT_ConcentrateContx_RmDup_8x8_DownSpat[5,5,3,1].yaml"
		#--cfg_file  config/custom/k400/nat/nat3D_GlobalT_DownSpat_RmDup_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_Shrink3D_GlobalT_RmDup_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_Dilated3D_hybrid_GlobalT_RmDup_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_Dilated3D_hybrid_GlobalT_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat3D_GlobalT_RmDup_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat3D_gt_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_Dilated3D_hybrid_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_Dilated3D_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_2FrontD_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat3D_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_fuse_motion_tct_8x8_roll.yaml"
		#--cfg_file  config/custom/k400/nat/nat_bridge_fuse_tct_8x8_roll.yaml"
		#--cfg_file  config/custom/k400/nat/nat_fuse_tct_8x8_roll.yaml"
		#--cfg_file  config/custom/k400/nat/nat_seq_tct_8x8_roll.yaml"
		#--cfg_file  config/custom/k400/nat/nat2D_8x8.yaml"
		#--cfg_file  config/custom/k400/nat/nat_simple_tct_8x8_roll_split.yaml"
		#--cfg_file  config/custom/k400/nat/nat_simple_tct_8x8_roll.yaml"
        #--cfg_file config/custom/k400/visformer/visformer_LAPS_8x8.yaml"
        #--cfg_file config/custom/k400/visformer/visformer_8x8_LA.yaml"
        #--cfg_file config/custom/k400/visformer/visformer_8x8_PS.yaml"
        #--cfg_file config/custom/k400/visformer/visformer_plain_shift.yaml"
		#--cfg_file config/custom/k400/visformer/visformer_8x8_base3D.yaml"
		#--cfg_file config/custom/k400/visformer/visformer_8x8.yaml"
		#--cfg_file config/custom/k400/tokshift/tokshift_16x32_b16.yaml"
		#--cfg_file config/custom/k400/tokshift/tokshift_8x32_b16_384.yaml"
		#--cfg_file config/custom/k400/vit_8x32_b16.yaml"
os.system(cmd)

