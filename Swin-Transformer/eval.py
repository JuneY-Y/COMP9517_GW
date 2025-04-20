import os
import subprocess
import json
import pandas as pd
from glob import glob

CONFIG_BASE = "configs/swinv2"
OUTPUT_ROOT = "outputs"
DATA_PATH = "datasets"
DEBUG_LOG_DIR = "debug_logs"

os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

results = []

for root, dirs, files in os.walk(OUTPUT_ROOT):
    for file in files:
        if file.endswith(".pth"):
            pth_path = os.path.join(root, file)

            folder = root.split(os.sep)[-3] if "default" in root else root.split(os.sep)[-1]
            cfg_path = os.path.join(CONFIG_BASE, f"{folder}.yaml")

            if not os.path.exists(cfg_path):
                print(f"⚠️ Config not found for: {folder}, skipping.")
                continue

            print(f"🚀 Evaluating: {folder}")
            cmd = [
                "python", "main.py",
                "--cfg", cfg_path,
                "--data-path", DATA_PATH,
                "--output", os.path.join(OUTPUT_ROOT, folder),
                "--resume", pth_path,
                "--eval"
            ]

            # Run subprocess
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = process.stdout + process.stderr

            # Save full log for inspection
            with open(os.path.join(DEBUG_LOG_DIR, f"{folder}.txt"), "w") as f:
                f.write(output)

            acc1, acc5, f1, recall, loss = "N/A", "N/A", "N/A", "N/A", "N/A"
            for line in output.splitlines():
                if "Accuracy@1" in line and "Accuracy@5" in line and "Loss" in line:
                    try:
                        acc1 = float(line.split("Accuracy@1:")[1].split("%")[0].strip())
                        acc5 = float(line.split("Accuracy@5:")[1].split("%")[0].strip())
                        loss = float(line.split("Loss:")[1].split()[0])
                    except:
                        pass
                if "Classification Report" in line and "F1-score" in line:
                    try:
                        f1 = float(line.split("F1-score:")[1].split(",")[0].strip())
                        recall = float(line.split("Recall:")[1].split(",")[0].strip())
                    except:
                        pass

            results.append({
                "model": folder,
                "acc1": acc1,
                "acc5": acc5,
                "loss": loss,
                "macro_f1": f1,
                "macro_recall": recall,
                "ckpt": file
            })

# Save result CSV
df = pd.DataFrame(results)
csv_path = "eval.csv"
df.to_csv(csv_path, index=False)
print(f"\n✅ Summary saved to: {csv_path}")
print(df)