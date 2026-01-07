# ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation

The paper has been accepted to the main conference of EACL 2026, and can be viewed [here](https://arxiv.org/abs/2601.03121).

## Overall Framework of ToxiGAN
ToxiGAN with k toxic generators, one neutral texts provider, and one multi-class discriminator:
- Toxic Generator Module ($G$): Consists of multiple LSTM-based toxic generators and learns to generate samples for each toxic class from a noise distribution. Each class has a dedicated decoding branch.
- Multi-class Discriminator ($D$): Classifies input text into $K+2$ classes: $K$ toxic classes, one neutral class, and one fake class to capture unrealistic generations.
- LLM-based Neutral Text Provider: A pre-trained LLM (e.g., Llama 3.2) is used to generate neutral in-domain examples for training $D$ and guiding $G$ via few-shot learning from the real neutral texts.

<img src="figures/Framework.png" width="60%">

## Two-Step Alternating Directional Learning in Embedding Space
The black arrow shows the initial generation after pretraining. Gray arrows represent updates during alternating optimization: shifting toward toxicity and authenticity directions by penalizing unexpected directional evaluations.

<img src="figures/Two-Step.png" width="40%">

t-SNE visualization of real and synthetic texts. Arrows indicate semantic shifts: neutral to toxic (green), out-of-domain to in-domain (purple), and their composite (blue).

<img src="figures/t-sne.png" width="50%">

## To Start
Requirements
- Python 3.10
- PyTorch 2.6.0 + CUDA/cuDNN backend
- transformers==4.53.3
- sentence-transformers==4.1.0
- scikit-learn==1.6.1

Direct to [ToxiGAN/codes/](codes/): Run train.py for training of ToxiGAN. 
```
$ python train.py 
```
After training of ToxiGAN, generate samples by command.
```
$ python generate_samples.py 
```

