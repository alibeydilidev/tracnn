# TraCNN

TraCNN is a hybrid AI architecture that unifies Convolutional Neural Networks (CNN) and Transformer layers into a single spatio-temporal reasoning framework. Designed for experimental research using synthetic data, TraCNN explores model fusion, temporal encoding, and interpretability in AI systems.

## Project Structure

- `training/` — training and visualization scripts
- `data/` — synthetic dataset generation modules
- `models/` — CNN, Transformer, and Fusion modules
- `outputs/` — saved models and visualizations
- `conclusion.md` — experiment report with attention, embedding, and saliency analyses

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/alibeydilidev/tracnn.git
cd tracnn
pip install -r requirements.txt
```

## Configuration

Edit `training/config.yaml` to set model and training parameters.

## Running Training

```bash
python training/train.py
```

## Running Embedding Visualization

```bash
python training/embedding_viz.py
```

This will produce PCA plots, saliency maps, attention visualizations, and fusion insights under `outputs/`.

## Experiment Summary

For a full analysis of model behavior and performance, see the [`conclusion.md`](./conclusion.md) file.

## Author

Ali Beydili  
AI Systems Architect  
Focus: Model fusion, decision synthesis, synthetic experimentation
