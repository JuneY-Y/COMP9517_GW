# ==============================================================================
# name: run_all_models.py
# aim: 自动运行当前目录下所有 model_*.py 的 YOLOv8 分类训练脚本
#
# Written by: JiamingYang (see my website niyoumengma.cn) z5452842 
# Date: 2025-04-10
# For automated experiment batch training
# ==============================================================================

import os
import subprocess
from pathlib import Path

# 设定模型文件匹配前缀
model_prefix = "model_"
model_suffix = ".py"

# 获取当前目录下所有以 model_ 开头的训练脚本
scripts = [f for f in os.listdir(".") if f.startswith(model_prefix) and f.endswith(model_suffix)]

# 按名称排序（可选）
scripts.sort()

print(f"🧠 检测到 {len(scripts)} 个模型脚本：")
for script in scripts:
    print(f"  → {script}")

# 自动运行每个脚本
for script in scripts:
    print(f"\n🚀 正在运行: {script}")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ 完成: {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 出错于: {script}，错误信息如下：")
        print(e)

print("\n🎉 所有模型脚本执行完毕。")