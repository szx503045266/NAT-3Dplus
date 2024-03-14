#!/usr/bin/env python
import os
### nat2D_8x8
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat2D_8x8.yaml"
# os.system(cmd)

### nat_simple_tct_roll
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_simple_tct_8x8_roll.yaml"
# os.system(cmd)

### nat_simple_tct_shift
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/nat_simple_tct_8x8_shift.yaml"
#os.system(cmd)

### nat_simple_tct_shift_pad
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_simple_tct_8x8_shift_pad.yaml"
# os.system(cmd)

### nat_simple_tct_roll_split
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_simple_tct_8x8_roll_split.yaml"
# os.system(cmd)

### nat_seq_tct_roll
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_seq_tct_8x8_roll.yaml"
# os.system(cmd)

### nat_fuse_tct_roll
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_fuse_tct_8x8_roll.yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_fuse_tct_8x8_roll.yaml \
# 		--resume video_checkpoints/nat_fuse_tct_roll_nat_mini_kinetics_C400_8x8_E27_LR0.1_B20_S224_D0/best_ckpt_e19.pth"
# os.system(cmd)

### nat_bridge_fuse_tct_roll
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_bridge_fuse_tct_8x8_roll.yaml"
# os.system(cmd)

### nat_fuse_motion_tct_roll
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/nat_fuse_motion_tct_8x8_roll.yaml"
#os.system(cmd)

### nat3D
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/nat3D_8x8.yaml"
#os.system(cmd)

### nat3D_gt
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/nat3D_gt_8x8.yaml"
#os.system(cmd)

### nat3D_GlobalT
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/nat3D_GlobalT_8x8.yaml"
#os.system(cmd)

### nat3D_GlobalT_RmDup
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat3D_GlobalT_RmDup_8x8.yaml"
# os.system(cmd)

### nat3D_GlobalT_DownSpat_RmDup
cmd = "python -u main_tokshift.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--cfg_file config/custom/k400/nat/nat3D_GlobalT_DownSpat_RmDup_8x8.yaml"
os.system(cmd)

### nat_Dilated3D_hybrid_GlobalT
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_Dilated3D_hybrid_GlobalT_8x8.yaml"
# os.system(cmd)

### nat_Dilated3D_hybrid_GlobalT_RmDup
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_Dilated3D_hybrid_GlobalT_RmDup_8x8.yaml"
# os.system(cmd)

### nat_Shrink3D_GlobalT_RmDup
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_Shrink3D_GlobalT_RmDup_8x8.yaml"
# os.system(cmd)

## nat_Dilated3D
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_Dilated3D_8x8.yaml"
# os.system(cmd)

## nat_Dilated3D_hybrid
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/nat_Dilated3D_hybrid_8x8.yaml"
# os.system(cmd)

### nat_2FrontD
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/nat_2FrontD_8x8.yaml"
#os.system(cmd)

### nat_2BackD
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/nat_2BackD_8x8.yaml"
#os.system(cmd)
