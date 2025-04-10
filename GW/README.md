# YOLOv8 Aerial Image Classifier 🌍
build by JiamingYang

This project uses **YOLOv8's classification mode** to classify aerial landscape images (e.g., Beach, Forest, City).

## 🔧 Setup

```bash
pip install -r requirements.txt

python split_dataset.py

python model.py

python predict.py path/to/image.jpg

Aerial_Landscapes/
├── Agriculture/
├── Beach/
├── City/
└── ...

runs/classify/train/weights/best.pt