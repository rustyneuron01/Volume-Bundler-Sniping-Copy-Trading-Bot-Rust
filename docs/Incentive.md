# Incentive Mechanism

## Benchmark Runs
Submitted discriminator miners are evaluated against a subset of the data sources listed below. Models are evaluated on cloud infrastructure -- miners do not need to host hardware for inference. A portion of the evaluation data comes from generative miners, who are rewarded based on their ability to submit data that both pass validator sanity checks (prompt alignment, etc.) and fool discriminators in benchmark runs.

Each modality (image, video, audio) is scored independently using the `sn34_score` metric, which combines discrimination accuracy (MCC) with calibration quality (Brier score).

<details>
<summary><strong>Evaluation Datasets</strong></summary>

Benchmark datasets are regularly expanded. Each modality includes a mix of real, synthetic, and semi-synthetic content from diverse sources (including continuously-updated [GAS-Station](https://huggingface.co/gasstation) data from generative miners).

**Public datasets** (available for training via gasbench):
- **Image**: [`image_datasets.yaml`](https://github.com/BitMind-AI/gasbench/blob/main/src/gasbench/dataset/configs/image_datasets.yaml)
- **Video**: [`video_datasets.yaml`](https://github.com/BitMind-AI/gasbench/blob/main/src/gasbench/dataset/configs/video_datasets.yaml)
- **Audio**: [`audio_datasets.yaml`](https://github.com/BitMind-AI/gasbench/blob/main/src/gasbench/dataset/configs/audio_datasets.yaml)

**Holdout datasets**: In addition to the public datasets above, each benchmark round includes holdout datasets that are not publicly available during the round. Holdout data is critical to ensure models generalize well and to mitigate overfitting. At the end of each round, many of the holdout datasets are released and added to the public gasbench datasets for future training. Some holdout datasets cannot be released publicly due to licensing or other restrictions.

</details>

<details>
<summary><strong>Generative Models</strong></summary>

The following models are run by validators to produce a continual, fresh stream of synthetic and semisynthetic data. The outputs of these models are uploaded at regular intervals to public datasets in the [GAS-Station](https://huggingface.co/gasstation) Hugging Face org for miner training and evaluation.

### Text-to-Image Models

- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SG161222/RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)
- [Corcelio/mobius](https://huggingface.co/Corcelio/mobius)
- [prompthero/openjourney-v4](https://huggingface.co/prompthero/openjourney-v4)
- [cagliostrolab/animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) + [Kvikontent/midjourney-v6](https://huggingface.co/Kvikontent/midjourney-v6) LoRA
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [DeepFloyd/IF](https://huggingface.co/DeepFloyd/IF)
- [deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
- [THUDM/CogView4-6B](https://huggingface.co/THUDM/CogView4-6B)

### Image-to-Image Models

- [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
- [Lykon/dreamshaper-8-inpainting](https://huggingface.co/Lykon/dreamshaper-8-inpainting)

### Text-to-Video Models

- [tencent/HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)
- [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview)
- [THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
- [ByteDance/AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning)
- [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)

### Image-to-Video Models

- [THUDM/CogVideoX1.5-5B-I2V](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V) 

</details> 


## Generator Rewards

The generator incentive mechanism combines two components: a base reward for passing data validation checks, and a multiplier based on adversarial performance against discriminators.

### Base Reward (Data Validation)

Generators receive a base reward based on their data verification pass rate:

$$R_{\text{base}} = p \cdot \min(n, 10)$$

Where:
- $p$ = pass rate (proportion of generated content that passes validation)
- $n$ = number of verified samples (`min(n, 10)` creates a rampup of incentive for the first 10 samples)

### Fool Rate Multiplier (Adversarial Performance)

Generators earn additional rewards by successfully fooling discriminators. The multiplier is calculated as:

$$M = \max(0, \min(2.0, f \cdot s))$$

Where:
- $f$ = fool rate = $\frac{N_{\text{fooled}}}{N_{\text{fooled}} + N_{\text{not fooled}}}$
- $s$ = sample size multiplier

The sample size multiplier encourages generators to be evaluated on more samples, similar to the sample size ramp used in the base reward.

$$s = \begin{cases}
\max(0.5, \frac{c}{20}) & \text{if } c < 20 \\
\min(2.0, 1.0 + \ln(\frac{c}{20})) & \text{if } c \geq 20
\end{cases}$$

Where:
- $c$ = total evaluation count (fooled + not fooled)
- Reference count of 20 gives multiplier of 1.0
- Sample sizes below 20 are penalized
- Sample sizes above 20 receive logarithmic bonus up to 2.0x

### Final Generator Reward

The total generator reward combines both components:

$$R_{\text{total}} = R_{\text{base}} \cdot M$$

This design incentivizes generators to:
1. Produce high-quality, valid content (base reward)
2. Create adversarially robust content that can fool discriminators (multiplier)
3. Participate in more evaluations for sample size bonuses



## Discriminator Rewards

### Scoring: `sn34_score`

Each discriminator model is scored per modality using the `sn34_score`, which combines two metrics:

1. **Binary MCC (Matthews Correlation Coefficient)** -- measures how well the model discriminates between real and synthetic content. Ranges from -1 (worst) to +1 (perfect).

2. **Brier Score** -- measures calibration quality (how well predicted probabilities match actual outcomes). Ranges from 0 (perfect) to 0.25 (random baseline).

These are combined as follows:

$$MCC_{norm} = \left(\frac{MCC + 1}{2}\right)^{\alpha}$$

$$Brier_{norm} = \left(\frac{0.25 - Brier}{0.25}\right)^{\beta}$$

$$sn34_{score} = \sqrt{MCC_{norm} \cdot Brier_{norm}}$$

With default parameters $\alpha = 1.2$ and $\beta = 1.8$. The geometric mean penalizes models that are strong on one axis but weak on the other -- a model must be both accurate *and* well-calibrated to score highly.

### Competition Rounds

The discriminator competition is organized into **rounds**. Each round introduces new benchmark datasets and evaluates all submitted models. Winners are determined **per modality** (image, video, audio) independently.

#### How Rounds Work

1. **New round begins**: Benchmark datasets are updated (new GAS-Station data, potentially new static datasets). All modalities share the same benchmark version number.
2. **Models are benchmarked**: All submitted discriminator models are evaluated against the current round's datasets and scored using `sn34_score`.
3. **Winner determined per modality**: The highest-scoring model for each modality wins that round.
4. **Alpha reward**: The round winner for each modality receives an alpha reward.

#### Winner-Take-All Per Round

Each round is winner-take-all -- only the top-scoring discriminator for each modality receives the alpha reward for that round. This incentivizes miners to continuously improve their models and push the state of the art in AI-generated content detection.

Rounds progress as benchmark versions are incremented, ensuring that models are always evaluated against fresh, evolving data.
