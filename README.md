# Multimodal Fusion CRNN

<img src="resources/feature_fusion.png" alt="Feature Fusion architecture" width="350"/>

## Setup

```
pip install -r requirements.txt
```

## Training

```
python train_DHG.py --conf <path/to/config.yaml>
```
```
python train_SHREC.py --conf <path/to/config.yaml>
```
See the [sample configs](sample_configs/) for config examples.

## Grayscale Variation
Original 16-bit depth image:<br>
<img src="resources/depth_hand.png" alt="Normal" width="250"/> <br>

Depth quantized 8-bit gvar image:<br>
<img src="resources/depth_quantized_hand.png" alt="gVar" width="250"/>

## Citation

```bibtex
@misc{mahmud2021deep,
      title   = {A Deep Learning-based Multimodal Depth-Aware Dynamic Hand Gesture Recognition System}, 
      author  = {Hasan Mahmud and Mashrur M. Morshed and Md. Kamrul Hasan},
      year    = {2021},
      eprint  = {2107.02543},
      archivePrefix = {arXiv},
      primaryClass = {cs.CV}
}
```