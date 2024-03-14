#!/usr/bin/env python
import os

### nat2D
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat2D_8x8.yaml"
# os.system(cmd)

### nat2.5D
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat_2FrontD_8x8.yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat_2BackD_8x8.yaml"
# os.system(cmd)

### nat3D
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_8x8.yaml"
# os.system(cmd)

### nat3D_GlobalT_ConcentrateContx_RmDup, [5,5,3,1]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_ConcentrateContx_RmDup_8x8_DownSpat[5,5,3,1].yaml"
# os.system(cmd)

### nat3D_GlobalT2_CondenseContext_RmDup, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT2_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup_SE, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_SE_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup_SE2, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_SE2_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup_SE3, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_SE3_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup, [F,F,F,F]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x8_DownSpat[56,28,14,7].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup_shift, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_shift_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup_shiftpad, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_shiftpad_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup, [16,8,4,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x8_DownSpat[16,8,4,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x2_DownSpat[8,4,2,0].yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x4_DownSpat[8,4,2,0].yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x16_DownSpat[8,4,2,0].yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_16x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_16x4_DownSpat[8,4,2,0].yaml"
# os.system(cmd)
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_32x2_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

### nat3D_GlobalT_CondenseContext_RmDup, [4,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_GlobalT_CondenseContext_RmDup_8x8_DownSpat[4,4,2,0].yaml"
# os.system(cmd)

### Big model

### nat3D_small
cmd = "python -u main_tokshift.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--cfg_file config/custom/k400/nat/paper_run/nat3D_small_8x8.yaml"
os.system(cmd)

### nat3D_base
#cmd = "python -u main_tokshift.py \
#		--multiprocessing-distributed --world-size 1 --rank 0 \
#		--dist-ur tcp://127.0.0.1:23677 \
#		--cfg_file config/custom/k400/nat/paper_run/nat3D_base_8x8.yaml"
#os.system(cmd)

## nat3D_small_GlobalT_CondenseContext_RmDup, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_small_GlobalT_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)

## nat3D_base_GlobalT_CondenseContext_RmDup, [8,4,2,0]
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_base_GlobalT_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml \
# 		--resume video_checkpoints/nat3D_GlobalT_CondenseContext_RmDup_nat_base_kinetics_C400_8x8_E28_LR0.04_B8_S224_D0_DownSpat_8_4_2_0/best_ckpt_e24.pth"
# os.system(cmd)

## nat3D_base_GlobalT_CondenseContext_RmDup, [11,5.5,2.6,0], 320
# cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_base_GlobalT_CondenseContext_RmDup_8x8_320_DownSpat[11,5.5,2.6,0].yaml \
# 		--resume video_checkpoints/nat3D_GlobalT_CondenseContext_RmDup_nat_base_kinetics_C400_8x8_E28_LR0.015_B3_S320_D0_DownSpat_11_5.5_2.6_0/ckpt_e2.pth"
# os.system(cmd)

## nat3D_base_GlobalT_CondenseContext_RmDup, [8,4,2,0], 320-v2
#cmd = "python -u main_tokshift.py \
# 		--multiprocessing-distributed --world-size 1 --rank 0 \
# 		--dist-ur tcp://127.0.0.1:23677 \
# 		--cfg_file config/custom/k400/nat/paper_run/nat3D_base_GlobalT_CondenseContext_RmDup_8x8_320_v2_DownSpat[8,4,2,0].yaml"
#os.system(cmd)

# nat3D_large_GlobalT_CondenseContext_RmDup, [8,4,2,0], 320-v2
# cmd = "python -u main_tokshift.py \
#  		--multiprocessing-distributed --world-size 1 --rank 0 \
#  		--dist-ur tcp://127.0.0.1:23677 \
#  		--cfg_file config/custom/k400/nat/paper_run/nat3D_large_GlobalT_CondenseContext_RmDup_8x8_DownSpat[8,4,2,0].yaml"
# os.system(cmd)
