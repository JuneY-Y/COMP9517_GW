#!/usr/bin/env bash

# --------------------------
# swinv2_base_patch4_window8_256_with_pretrain-ft.yaml
# --------------------------
python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-ft-LMA.yaml \
              --data-path datasets \
              --batch-size 128 \
              --optim adamw \
              --pretrained swinv2_base_patch4_window8_256.pth \
              --output outputs/swinv2_base_patch4_window8_256_with_pretrain-ft-LMA
echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain-ft.yaml"
sleep 2
clear

# --------------------------
# swinv2_base_patch4_window8_256_with_pretrain-nocolor.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-color.yaml \
#               --data-path datasets \
#               --batch-size 256 \
#               --optim adamw \
#               --pretrained swinv2_base_patch4_window8_256.pth \
#               --output outputs/swinv2_base_patch4_window8_256_with_pretrain-color
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain-nocolor.yaml"
# sleep 2
# clear

# --------------------------
# swinv2_base_patch4_window8_256_with_pretrain-noauto.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-auto.yaml \
#               --data-path datasets \
#               --batch-size 256 \
#               --optim adamw \
#               --pretrained swinv2_base_patch4_window8_256.pth \
#               --output outputs/swinv2_base_patch4_window8_256_with_pretrain-auto
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain-noauto.yaml"
# sleep 2
# clear

# --------------------------
# swinv2_base_patch4_window8_256_with_pretrain-ft_longtail.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-ft_longtail.yaml \
#               --data-path datasets_longtail \
#               --batch-size 256 \
#               --optim adamw \
#               --pretrained swinv2_base_patch4_window8_256.pth \
#               --output outputs/swinv2_base_patch4_window8_256_with_pretrain-ft_longtail
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain-ft_longtail.yaml"
# sleep 2
# clear
# # --------------------------
# # swinv2_base_patch4_window8_256_with_pretrain-trivial.yaml
# # --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-trivial.yaml \
#                --data-path datasets \
#                --batch-size 256 \
#                --optim adamw \
#                --pretrained swinv2_base_patch4_window8_256.pth \
#                --output outputs/swinv2_base_patch4_window8_256_with_pretrain-trivial
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain-AUG.yaml"
# sleep 2
# clear

# --------------------------
# swinv2_base_patch4_window8_256_with_pretrain-noocc.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-occ.yaml \
#               --data-path datasets \
#               --batch-size 256 \
#               --optim adamw \
#               --pretrained swinv2_base_patch4_window8_256.pth \
#               --output outputs/swinv2_base_patch4_window8_256_with_pretrain-occ
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain-noocc.yaml"
# sleep 2
# clear

# # --------------------------
# # swinv2_tiny_patch4_window16_256_with_pretrain-ft.yaml
# # --------------------------
# python main.py --cfg configs/swinv2/swinv2_tiny_patch4_window16_256_with_pretrain-ft.yaml \
#                --data-path datasets \
#                --batch-size 256 \
#                --optim adamw \
#                --pretrained swinv2_tiny_patch4_window16_256.pth \
#                --output outputs/swinv2_tiny_patch4_window16_256_with_pretrain-ft
# echo "✅ Finished: swinv2_tiny_patch4_window16_256_with_pretrain-ft.yaml"
# sleep 2
# clear






# --------------------------
# swinv2_base_patch4_window8_256-nopretrain-changewindow.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256-nopretrain-changewindow.yaml \
#               --data-path datasets \
#               --batch-size 256 \
#               --optim adamw \
#               --output outputs/swinv2_base_patch4_window8_256-nopretrain-changewindow-nosw
# echo "✅ Finished: swinv2_base_patch4_window8_256-nopretrain-changewindow.yaml"
# sleep 2
# clear

# --------------------------
# swinv2_base_patch4_window8_256-nopretrain-changedeep.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256-nopretrain-changedeep.yaml \
#               --data-path datasets \
#               --batch-size 256 \
#               --optim adamw \
#               --output outputs/swinv2_base_patch4_window8_256-nopretrain-changedeep
# echo "✅ Finished: swinv2_base_patch4_window8_256-nopretrain-changedeep.yaml"
# sleep 2
# clear

# --------------------------
# swinv2_base_patch4_window8_256-nopretrain-origin.yaml
# --------------------------
#python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_nopretrain_origin.yaml \
#               --data-path datasets \
#               --batch-size 256 \
#               --optim adamw \
#               --output outputs/swinv2_base_patch4_window8_256_nopretrain_origin
#echo "✅ Finished: swinv2_base_patch4_window8_256-nopretrain-origin.yaml"
#sleep 2
#clear



# --------------------------
# swinv2_base_patch4_window8_256_with_pretrain-lr.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain-lr.yaml \
#               --data-path datasets \
#               --batch-size 256 \
#               --optim adamw \
#               --pretrained swinv2_base_patch4_window8_256.pth \
#               --output outputs/swinv2_base_patch4_window8_256_with_pretrain-lr
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain-lr.yaml"
# sleep 2
# clear

# --------------------------
# swinv2_base_patch4_window8_256_with_pretrain_origin_longtail.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain_origin_longtail.yaml \
#                --data-path datasets_longtail \
#                --batch-size 256 \
#                --optim adamw \
#                --pretrained swinv2_base_patch4_window8_256.pth \
#                --output outputs/swinv2_base_patch4_window8_256_with_pretrain_origin_longtail
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain_origin_longtail.yaml"
# sleep 2
# clear




# # --------------------------
# # swinv2_base_patch4_window8_256_with_pretrain_origin.yaml
# # --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_with_pretrain_origin.yaml \
#                --data-path datasets \
#                --batch-size 256 \
#                --optim adamw \
#                --pretrained swinv2_base_patch4_window8_256.pth \
#                --output outputs/swinv2_base_patch4_window8_256_with_pretrain_origin
# echo "✅ Finished: swinv2_base_patch4_window8_256_with_pretrain_origin.yaml"
# sleep 2
# clear

# --------------------------
# swinv2_base_patch4_window8_256_nopretrain_ft.yaml
# --------------------------
# python main.py --cfg configs/swinv2/swinv2_base_patch4_window8_256_nopretrain_ft.yaml \
#                --data-path datasets \
#                --batch-size 256 \
#                --optim adamw \
#                --output outputs/swinv2_base_patch4_window8_256_nopretrain_ft-nosw
# echo "✅ Finished: swinv2_base_patch4_window8_256_nopretrain_ft.yaml"
# sleep 2
# clear
