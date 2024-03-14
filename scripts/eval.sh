#!/usr/bin/env python
import os
cmd = "python -u main_tokshift.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--evaluate \
		--resume video_checkpoints/nat3D_nat_base_kinetics_C400_8x8_E28_LR0.04_B8_S224_D0//ckpt_e27.pth \
		--cfg_file config/custom/k400/nat/paper_run/nat3D_base_8x8.yaml"
		#--resume video_checkpoints/nat3D_GlobalT_CondenseContext_RmDup_nat_mini_kinetics_C400_8x8_E28_LR0.09_B18_S224_D0_DownSpat_8_4_2_0/best_ckpt_e27.pth \
		#--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
		#--resume video_checkpoints/nat_bridge_fuse_tct_roll_nat_mini_kinetics_C400_8x8_E28_LR0.1_B20_S224_D0/best_ckpt_e25.pth \
        #--cfg_file config/custom/k400/nat/nat_bridge_fuse_tct_8x8_roll.yaml"
        #--resume video_checkpoints/Visformer_LAPS_visformer_small_kinetics_C400_8x8_E18_LR0.09023_B40_S224_D4_SLevel_s2_1_2_3_1_s3_2_3_1_2/best_ckpt_e15.pth \
        #--cfg_file config/custom/k400/visformer/visformer_LAPS_8x8.yaml"
        #--resume video_checkpoints/TokShift_vit_base_patch16_224_in21k_kinetics_C400_8x32_E18_LR0.23_B24_S224_D4/tokshift_8x32_224_e17.pth \
		#--cfg_file config/custom/k400/tokshift/tokshift_8x32_b16.yaml"
os.system(cmd)
