# Project Proposal — Grounded VQA with a Lightweight Bridge Between Frozen Vision and Language Models

CS-747 Deep Learning for Computer Vision · Topic 30 (Cluster 6) · Roll No. 23DCS004 · Revised 8 October 2026

## At a glance

This project builds a small-scale BLIP-2-style VQA system and tests whether its answers come from the image or from language priors. Only a bridge of about 10M parameters is trained, between a frozen SigLIP 2 Large vision encoder and a frozen Llama 3.2 1B language model. Every experiment is planned to fit in one RTX 5090 session of about 3 h 15 min.

| Item | Plan |
| --- | --- |
| Course and topic | CS-747, Cluster 6, Topic 30: Multimodal Foundation Models for Visual Question Answering |
| Roll No. | 23DCS004 |
| Assigned reference | BLIP-2, Li et al., ICML 2023 [1] |
| Task | Image + natural-language question → short free-form answer |
| Dataset | VQA v2 subset: 20,000 training images by default (about 107K question-answer pairs) from a 40,000-image pool, and 5,000 held-out validation images (44,119 questions) |
| Vision encoder (frozen) | SigLIP 2 Large, 303M parameters, 256 px [2]; patch features cached once |
| Language model (frozen) | Llama 3.2 1B Instruct, text-only, 1.24B parameters [3] |
| Trainable part | Query-based cross-attention bridge, 32 visual tokens, about 10.1M parameters |
| Metrics | VQA accuracy (overall, Yes/No, Number, Other) plus four groundedness measures |
| Compute | One RTX 5090 (32 GB); 3 h 15 min planned, 4 h hard cap |
| Final deliverable | IEEE two-column conference paper and code |

## 1. Problem Formulation

The task is open-ended visual question answering: given an RGB image and an English question, generate a short free-form answer.

**Input and output.** The input is one RGB image, resized to 256 × 256, and a question as text. The output is an answer string, usually one to three words, generated token by token and not picked from a fixed answer list.

**Why it is hard**

- **Modality gap.** The language model is text-only and has never seen visual features. A small bridge must turn 256 patch embeddings into tokens it can read, while both large models stay frozen.
- **Language priors.** A model can score well without looking at the image. VQA v2 pairs each question with two similar images that have different answers to weaken this shortcut [4]. VQA-CP shows accuracy still drops sharply when answer priors change between train and test [5]. Accuracy alone cannot certify grounding.
- **Compute and data.** BLIP-2 aligns its bridge on 129M images for 250k + 80k steps on 16 A100 GPUs [1]. This project has under 4 GPU-hours, so large-scale alignment pre-training is not available.
- **Information bottleneck.** A few dozen visual tokens must carry counts, colours and positions for any question that might be asked.

**Scope**

| In scope | Out of scope |
| --- | --- |
| English questions with short answers | Long-form answers, dialogue, multi-image input |
| VQA v2 subset: up to 40,000 training images, 5,000 held-out validation images | Full VQA v2 training; test-dev and test-std server submission |
| Frozen SigLIP 2 Large at 256 px | Fine-tuning the vision encoder; higher resolutions |
| Frozen Llama 3.2 1B, plus one lightly fine-tuned (LoRA) variant | Larger language models; vision-language models that already accept images |
| Bridge trained from scratch on VQA pairs | Stage-1 image-text pre-training (contrastive, matching, captioning losses) |
| Groundedness tests on object existence, counting and left/right relations | Knowledge-based VQA (OK-VQA), GQA, retraining on the VQA-CP split |

**Research question.** With under 4 GPU-hours, can a bridge of about 10M parameters between two frozen models give answers that depend on the image? Which bridge design choices improve grounding, and which only improve the use of language priors?

**Hypotheses.** Each one is tested by a named run in section 7.

- **H1, grounding.** The image-conditioned model beats its image-blind twin on every groundedness measure, with the largest gap on Number and Other questions.
- **H2, bridge type.** Without stage-1 pre-training, the query bridge does not beat a pooled MLP at an equal 16 visual tokens. BLIP-2 reports that its Q-Former bridges the modality gap poorly without that stage [1].
- **H3, LLM adaptation.** Lightly fine-tuning the language model raises standard VQA accuracy more than it raises groundedness scores, so part of its gain is prior fitting.

## 2. Related Work

BLIP-2 shows that a small trainable bridge can connect a frozen image encoder to a frozen language model. This project asks what survives of that idea at a tiny fraction of the compute, and whether the resulting answers are grounded.

### The assigned reference: BLIP-2 [1]

- **Core contribution.** A lightweight Querying Transformer (Q-Former) sits between a frozen image encoder and a frozen LLM, so vision-language pre-training updates only the bridge.
- **Method.** The Q-Former has 188M parameters, starts from BERT-base weights, and uses 32 learned queries of width 768. The queries read frozen image features through cross-attention. Stage 1 trains it with three losses: image-text contrastive, image-text matching and image-grounded text generation. Stage 2 adds one linear layer that projects the query outputs into the LLM's embedding space, where they act as soft visual prompts for a frozen OPT or FlanT5.
- **Training cost.** 129M images, 250k steps for stage 1 and 80k steps for stage 2. The largest model needs under 9 days on 16 A100 GPUs.
- **Results.** Zero-shot VQAv2 accuracy is 65.0 on test-dev (ViT-g with FlanT5-XXL), 8.7 points above Flamingo80B with 54× fewer trainable parameters. Fine-tuned VQAv2 accuracy reaches 82.19 on test-dev (ViT-g with OPT-6.7B). Without stage 1, zero-shot VQA accuracy is substantially lower.
- **Limitation for this project.** The method depends on pre-training that a course budget cannot reproduce. Its VQA fine-tuning also unfreezes the image encoder, giving 1.2B trainable parameters. It reports accuracy only and does not test whether answers are grounded in the image.

### Theme A: connecting frozen vision and language models

| Work | What it does | Limitation for this project |
| --- | --- | --- |
| Frozen [6] | Trains a vision encoder whose outputs form a prefix for a frozen language model | Low VQA accuracy (29.6 zero-shot on VQAv2 val, as listed in [1]); the vision encoder is trained, not reused |
| Flamingo [7] | Perceiver Resampler plus gated cross-attention layers inserted into a frozen LLM | 10.2B trainable parameters in the 80B model [1]; trained on non-public interleaved data |
| LLaVA [8], LLaVA-1.5 [9] | A linear layer, later a 2-layer MLP, maps every patch token into the LLM, which is then fine-tuned | Hundreds of visual tokens per image and a fine-tuned 7B to 13B LLM |
| InstructBLIP [10] | Makes the Q-Former instruction-aware, so the extracted features depend on the question | Starts from BLIP-2's pre-trained Q-Former, so it inherits the same pre-training cost |
| Honeybee [11] | Compares projector designs: resamplers are flexible in token count but lose spatial locality; linear projectors keep locality but fix the token count | Conclusions come from 7B and 13B LLMs; untested at small scale |

### Theme B: language priors and grounding evaluation

| Work | What it does | Limitation for this project |
| --- | --- | --- |
| VQA v2 [4] | Balances VQA by collecting complementary images, so one question has different answers on two similar images | Priors remain exploitable, and the headline metric is still plain accuracy |
| VQA-CP [5] | Re-splits VQA so answer priors differ between train and test; prior-reliant models drop sharply | Needs retraining on a separate split, outside the GPU budget |
| POPE [12] | Polls a model with yes/no questions on object presence, using random, popular and adversarial absent objects | Covers object existence only; no counting or spatial relations |

**Building blocks reused as they are:** SigLIP 2 [2] for vision features, Llama 3.2 1B [3] as the language model, and LoRA [13] for the lightly fine-tuned variant.

**Gap this project targets.** The bridge papers above report accuracy on large models, and the grounding papers probe finished models. This project measures accuracy and grounding together on one small bridge, and attributes each to a specific design choice.

## 3. Proposed Methodology

One small bridge is trained between a frozen SigLIP 2 Large and a frozen Llama 3.2 1B. Nothing else receives gradient updates.

```
Image, 256 x 256 RGB                                            Question, English text
        |                                                                |
        v                                                                v
SigLIP 2 Large (frozen)   ->   Query bridge (TRAINED)   ->   Llama 3.2 1B (frozen)
second-last layer output       32 learned queries            reads 32 visual tokens,
256 patch tokens x 1024-d      2 cross-attention blocks      then the question
run once, cached to disk       linear layer to LLM width     generates the answer
                                      ^                                  |
                                      |                                  v
                           answer cross-entropy loss   <-   Answer, short text

Gradients pass back through the frozen LLM; only the bridge is updated.
```

The vision encoder runs once per image and its output is cached, so each training run pays only for the bridge and the language model.

### Data

| Split | Source | Images | Questions | Used for |
| --- | --- | --- | --- | --- |
| Train pool | VQA v2 train (COCO train2014) | 40,000 | 214,642 | A run trains on the first N images; default N = 20,000 (about 107,000 questions) |
| Dev | VQA v2 train, disjoint images | 2,000 | 10,722 | Learning-rate choice and convergence curves |
| Test | VQA v2 val (COCO val2014) | 5,000 | 44,119 | Every reported number |
| GroundProbe | Built on the 5,000 test images | 5,000 | 1,954 | Groundedness measure G4 |

VQA v2 images come from COCO [14]. The dataset has 443,757 training questions on 82,783 images and 214,354 validation questions on 40,504 images [15]. Test images are picked pair-first: complementary pairs are drawn at random from the official validation list of 95,144 pairs [15], and both images of a pair are added until 5,000 images are reached. This yields 6,352 intact pairs for groundedness measure G3. Pair-first sampling favours images with many questions, so the test split has 8.8 questions per image against 5.3 for the full validation set. Every split uses a fixed seed. The training-set size N is set at the start of the GPU session from the measured training speed (section 9).

### Modules and data flow

1. **Vision encoder (frozen, reused).** SigLIP 2 Large at 256 px with 16 px patches. The second-last layer's 256 patch tokens of 1024 dimensions are kept, as BLIP-2 does with its encoder [1]. They are computed once per image and stored as float16: 0.5 MB per image, about 14 GB for 27,000 images.
2. **Query bridge (trained, own).** A linear layer maps patch tokens from 1024 to 512 dimensions and a learned position embedding is added. Then 32 learned queries pass through 2 blocks of self-attention, cross-attention to the patch tokens, and a feed-forward layer. A final linear layer maps the 32 outputs to the LLM width of 2048 and is initialised so the visual tokens start at the scale of the LLM's own word embeddings. Weights start random. The bridge has 10.1M parameters, 5% of BLIP-2's Q-Former.
3. **Language model (frozen, reused).** Llama 3.2 1B Instruct, a text-only decoder. The 32 visual tokens are placed at the start of the user turn, followed by the prompt `Question: {q} Short answer:`, the wording BLIP-2 uses [1]. The model then generates the answer in the assistant turn.
4. **Training signal.** Cross-entropy on the answer tokens and the end-of-turn token. Each step samples its target from the 10 human answers, so common answers are seen in proportion to annotator agreement. Gradients pass back through the frozen LLM and update only the bridge.
5. **Inference.** Greedy decoding with at most 8 new tokens, followed by the official VQA answer normalisation.

### Comparison bridges

- **Pooled MLP.** The 16 × 16 patch grid is average-pooled 4 × 4 to 16 tokens, then a 2-layer MLP maps them to the LLM width, in the style of LLaVA-1.5 [9]. It has 6.3M parameters.
- **Blind twin.** The query bridge with image features replaced by zeros at train and test time. Its 32 tokens become a learned soft prompt, which gives the language-prior floor at equal capacity.

### Own work versus reused components

| Component | Status |
| --- | --- |
| SigLIP 2 Large weights | Reused: pretrained, loaded through Hugging Face Transformers, frozen |
| Llama 3.2 1B Instruct weights | Reused: pretrained, loaded through Hugging Face Transformers, frozen. Released under Meta's Llama 3.2 Community License, which is not an open-source licence |
| LoRA layers for the lightly fine-tuned variant | Reused library (PEFT); configuration is own |
| VQA accuracy and answer normalisation | Adapted from the official VQA evaluation script |
| Query bridge, pooled-MLP bridge, blind twin | Own implementation in PyTorch |
| Feature caching, split construction, training loop, session runner | Own implementation |
| GroundProbe set and the four groundedness measures | Own implementation |
| Attention-map visualisation and speed benchmark | Own implementation |

### Deviations from BLIP-2 and their motivation

| BLIP-2 [1] | This project | Why |
| --- | --- | --- |
| Q-Former with BERT-base initialisation, 188M parameters | 2-block query bridge, random initialisation, 10.1M parameters | Fits the GPU budget |
| Two-stage pre-training on 129M images before any VQA data | No pre-training; the bridge learns directly from VQA pairs | 250k + 80k steps on 16 A100 GPUs is out of reach; H2 tests what this costs |
| EVA-CLIP ViT-g/14 or CLIP ViT-L/14 | SigLIP 2 Large | A newer encoder of the same size class as CLIP ViT-L/14; BLIP-2 observes that a stronger image encoder helps |
| FlanT5-XL and XXL, OPT-2.7B and 6.7B | Llama 3.2 1B | A current, widely used small LLM; the LLM's forward and backward pass dominates training time |
| VQA fine-tuning updates the Q-Former and the image encoder at 490 px | Image encoder frozen at 256 px, features cached | Caching removes the encoder from every training step |
| Q-Former also receives the question | Bridge sees the image only | Keeps the core design minimal |
| VQA data: VQAv2 train and val plus Visual Genome | Up to 40,000 training images of VQA v2 | Budget; validation images stay held out for testing |

## 4. Hyperparameter Choices

Every run uses one fixed recipe, so differences between runs come from the design choice under test and not from tuning. These are planned values; the final report will state the values actually used.

| Setting | Planned value |
| --- | --- |
| Optimiser | AdamW, betas (0.9, 0.999), weight decay 0.05 |
| Peak learning rate | 2e-4 for the bridge, confirmed by a check over 1e-4, 2e-4 and 5e-4; 1e-4 for LoRA weights |
| Schedule | Linear warm-up over the first 5% of steps, then cosine decay |
| Batch size | 128 |
| Epochs | 2 for every run; about 1,680 steps with 20,000 training images |
| Loss | Token-level cross-entropy on the answer, teacher forcing, no label smoothing |
| Precision | bfloat16 autocast; language model weights held in bfloat16 |
| Gradient clipping | Max norm 1.0 |
| Regularisation | Dropout 0.1 inside the bridge; weight decay as above |
| Augmentation | None on images, because features are cached; the target answer is re-sampled from the 10 human answers each step |
| Trainable parameters | Bridge only, 10.1M; the LoRA run adds rank 8, alpha 16 on the query and value projections, about 0.85M |
| Decoding | Greedy, at most 8 new tokens |
| Seeds | Seed 0 for every run; the reference run is repeated with seeds 1 and 2 if time is left |

**Justification of the most impactful choices**

1. **Cached frozen features with bfloat16.** This replaces a vision-encoder forward pass per sample with a memory read, which is what makes several training runs fit in the budget. The price is no image augmentation and a fixed 256 px input.
2. **Learning rate 2e-4.** BLIP-2 fine-tunes for VQA at 1e-5, but from a pre-trained Q-Former [1]. This bridge starts random and gets under 2,000 steps, so it needs a higher rate. The value is confirmed on the dev split with three short runs; a wider search does not fit the budget.
3. **Training-set size and 2 epochs.** Both are set by compute, not by convergence. The GPU session first measures real training speed and then picks the largest training set, from 5,000 to 40,000 images, whose planned runs fit the budget. All runs share the same subset and schedule so comparisons stay fair.
4. **Batch size 128.** GPU memory is not the limit here. A smaller batch gives more optimisation steps per epoch, which matters for a bridge trained from scratch.

**Hardware and compute budget.** One NVIDIA RTX 5090 with 32 GB of memory, 3 h 15 min planned and a hard cap of 4 h. Apart from the three-value learning-rate check, no hyperparameter search is feasible.

## 5. Evaluation Metrics

Success is measured on two axes: how often the answer is right, and whether it is right because of the image. Every reported number comes from the held-out test split. The dev split serves every choice made during training.

### Answer accuracy

The standard VQA accuracy used with VQA v2 [4] counts an answer as fully correct when at least 3 of the 10 annotators gave it:

```
Acc(a) = min( (number of annotators who answered a) / 3 , 1 )
```

The official script averages this over the ten leave-one-annotator-out subsets, after normalising punctuation, articles and number words. Accuracy is reported overall and for the dataset's three answer types: Yes/No, Number and Other.

### Groundedness

| Measure | How it is computed | What a grounded model shows |
| --- | --- | --- |
| G1 Blind gap | Accuracy of the full model minus accuracy of its image-blind twin, per answer type | A large positive gap |
| G2 Image-swap sensitivity | The test set is re-run with every image replaced by another test image; report the accuracy drop and the share of answers that change | Accuracy falls to about the blind level and many answers change |
| G3 Complementary-pair accuracy | On the 6,352 VQA v2 complementary pairs inside the test split (same question, two similar images, different answers [4]): share of pairs with both answers correct | A high share; a model that ignores the image gives one answer to both and cannot get both right |
| G4 GroundProbe accuracy | Accuracy on a new balanced probing set whose answers are verifiable from COCO annotations | Well above the chance level of each category |

### GroundProbe: the targeted probing set

GroundProbe is generated from the COCO instance annotations [14] of the 5,000 test images, so every answer can be checked against a bounding box. Each category is balanced, so a model that only uses language priors scores at chance.

| Category | Question template | Questions | Balance | Chance |
| --- | --- | --- | --- | --- |
| Existence | "Is there a {object} in the image?" with absent objects sampled as random, popular and adversarial, following POPE [12] | 1,000 | 50% yes, 50% no | 50% |
| Counting | "How many {objects} are in the image?" for 1 to 4 non-crowd instances, each covering at least 1% of the image | 496 | Equal shares of 1, 2, 3 and 4 | 25% |
| Left/right | "Is the {A} to the left of the {B}?" for images with exactly one A and one B whose boxes are separated horizontally | 458 | 50% yes, 50% no, by swapping A and B | 50% |

The counting and left/right sets are smaller than the 1,000 first planned, because the 5,000 test images do not contain more clean cases. A random sample of 200 questions is checked by hand, and the measured label-noise rate is reported next to the results.

### Efficiency

Each run also reports trainable parameters, training time in GPU-minutes, training speed in samples per second, and inference time per question.

## 6. Visualization of Results and Comparison with State of the Art

The report will show where the model looks, how accuracy trades against cost, and where the result stands next to published numbers.

### Planned figures

| Figure | Content | Purpose |
| --- | --- | --- |
| Qualitative grid | 8 test images with question, human answer, full-model answer and blind-twin answer; sampled by a fixed seed, half correct and half wrong | Sample predictions, failures included |
| Attention maps | Cross-attention of the queries over the 16 × 16 patch grid, overlaid on each image of the grid | Shows where the bridge looks |
| Trade-off plot | VQA accuracy and GroundProbe accuracy against inference time as the number of visual tokens changes | The accuracy-versus-cost curve |
| Convergence curves | Dev accuracy against training steps for each run | Evidence for H2 |
| Grounding bars | Accuracy per answer type for the full model, the blind twin and the image-swapped test | Evidence for H1 |

### Comparison table

The published rows are fixed. This project's row is filled in after the GPU session.

| Model | Setting | Trainable parameters | VQAv2 accuracy (%) | Split | Source |
| --- | --- | --- | --- | --- | --- |
| BLIP-2, ViT-g with OPT-6.7B | Fine-tuned | 1.2B | 82.19 | test-dev | [1] |
| Flamingo80B | Fine-tuned | 10.6B | 82.00 | test-dev | [1] |
| BLIP-2, ViT-g with FlanT5-XL | Fine-tuned | 1.2B | 81.55 | test-dev | [1] |
| LLaVA-1.5, Vicuna-13B | VQAv2 in its training mix | LLM fully fine-tuned | 80.0 | test-dev | [9], [16] |
| LLaVA-1.5, Vicuna-7B | VQAv2 in its training mix | LLM fully fine-tuned | 78.5 | test-dev | [9], [16] |
| BLIP-2, ViT-g with FlanT5-XXL | Zero-shot | 108M | 65.0 | test-dev | [1] |
| BLIP-2, ViT-L with FlanT5-XL | Zero-shot | 103M | 62.3 | test-dev | [1] |
| Flamingo80B | Zero-shot | 10.2B | 56.3 | test-dev | [1] |
| Frozen | Zero-shot | 40M | 29.6 | val | [1] |
| This project, query bridge with Llama 3.2 1B | Trained on a VQA v2 subset | 10.1M | To be measured | 5,000-image val subset | This project |

### Where the comparison is not like-for-like

- **Training data.** This project trains on at most 48% of the VQA v2 training images, and 24% by default. BLIP-2 fine-tunes on VQAv2 train and val plus Visual Genome [1].
- **Pre-training.** The bridge here never sees the 129M image-text pairs that BLIP-2's Q-Former is aligned on.
- **Evaluation split.** Results come from a 5,000-image subset of val, not from the test-dev server.
- **Model scale.** About 1.5B total parameters here, against 3.4B to 80B for the listed models.
- **Training length and resolution.** 2 epochs at 256 px with a frozen encoder, against 5 epochs at 490 px with the encoder unfrozen for BLIP-2 [1].
- **Zero-shot rows.** Those models saw no VQA training data, so they are context and not a like-for-like lower bar.

The project's claim therefore rests on the groundedness gaps and the ablation differences, which are all measured under identical conditions, and not on the absolute accuracy.

## 7. Ablation Studies

Five runs change one design choice at a time. All share the same data, seed, language model and 2-epoch schedule.

| Run | Bridge | Visual tokens | Language model | Compared with | Design choice isolated |
| --- | --- | --- | --- | --- | --- |
| `reference` (main model) | Query | 32 | Llama 3.2 1B, frozen | All others | Baseline for every comparison |
| `blind` | Query, image zeroed | 32 | Frozen | `reference` | Use of the image (H1) |
| `lora` | Query | 32 | With LoRA | `reference` | Frozen versus lightly fine-tuned LLM (H3) |
| `tokens16` | Query | 16 | Frozen | `reference` | Visual token budget |
| `mlp16` | Pooled MLP | 16 | Frozen | `tokens16` | Bridge architecture at equal tokens (H2) |

Extra runs, only if the session is ahead of plan: `seed1` and `seed2` (the reference run with other seeds, for a noise band), `tokens64` and `mlp64` (the bridge comparison repeated at 64 tokens).

**How results will be presented.** One table with a row per run and these columns: VQA accuracy (overall, Yes/No, Number, Other), image-swap accuracy and answer-change rate, complementary-pair accuracy, GroundProbe accuracy per category, trainable parameters, GPU-minutes and inference time.

**How each comparison will be read**

- **`reference` against `blind`.** The gap is the image's contribution. A small gap means the model answers from priors.
- **`reference` against `tokens16`.** Shows whether halving the visual tokens costs accuracy, and whether counting and left/right questions suffer more than Yes/No questions.
- **`tokens16` against `mlp16`.** Shows whether learned queries beat simple pooling when there is no alignment pre-training. The two bridges differ in size (10.1M against 6.3M parameters), which is reported as a confound.
- **`reference` against `lora`.** Compares the gain in VQA accuracy with the gain in pair and GroundProbe accuracy. Equal gains mean better visual reading; an accuracy-only gain means prior fitting.
- **Noise.** If the seed runs are done, any difference smaller than the spread across seeds is reported as no difference. If not, small differences are reported as inconclusive.

## 8. Summarization of Findings: Expected Outcomes, Risks and Limitations

The report will close by answering the research question with a verdict on each hypothesis, each backed by one comparison from section 7. No results exist yet; this section states what is expected and what could go wrong.

### Expected outcomes

- **Absolute accuracy.** Well below the fine-tuned systems in section 6, for the reasons listed there.
- **H1.** The full model is expected to beat its blind twin clearly. This is the minimum result for the project to count as a working VQA system.
- **H2.** Either verdict is informative. If the pooled MLP wins, it supports BLIP-2's claim that a query bridge needs alignment pre-training [1]. If the query bridge wins, direct VQA supervision is enough at this scale.
- **H3.** If LoRA lifts VQA accuracy but not pair or GroundProbe accuracy, the report will recommend keeping the LLM frozen when grounding matters.

### Risks and responses

| Risk | Early sign | Response |
| --- | --- | --- |
| Query bridge under-trains in under 2,000 steps without pre-training | `reference` is close to `blind` on the dev split | Report it as the H2 finding and present the pooled MLP as the stronger bridge |
| Training is slower than assumed | Speed test in step 1 of the GPU session | The session runner shrinks the training set; each run is also capped in minutes |
| GroundProbe labels are noisy because COCO boxes are imperfect | Hand-check of 200 questions | Report the noise rate; drop a category above 10% noise |
| Templated probe questions are unfamiliar to a model trained on human-written questions | All runs near chance on GroundProbe, including existence | Treat complementary-pair accuracy as the primary grounding measure, since its questions are human-written |
| Small differences between runs cannot be told from noise | Differences of about one point | Report such comparisons as inconclusive unless the seed runs were done |

### Known limitations

- Training on a subset and testing on a val subset means the accuracy is not a leaderboard number.
- Most runs have a single seed.
- Frozen 256 px features make small objects and text in images hard, and rule out image augmentation.
- GroundProbe covers three skills with templated questions; it does not test colour, actions or reading.
- A 1B language model limits questions that need world knowledge.
- With one language model size, the effect of LLM size is not measured.

### With more time and compute

The next steps would be a caption-alignment stage before VQA training, a question-conditioned bridge, training on all of VQA v2, evaluation on VQA-CP [5], and a second language model size.

## 9. Compute Budget and Execution Plan

All GPU work is run by one script inside a 195-minute budget. Everything that does not need the GPU is finished before the session starts.

### Before the GPU session (no GPU needed)

- [x] Fix the seeds and write the split files for train, dev and test.
- [x] Generate GroundProbe.
- [ ] Hand-check 200 GroundProbe questions (`data/splits/groundprobe_handcheck.csv`).
- [x] Write all code and run it end to end on CPU with a tiny language model and 96 images.
- [x] Download only the 47,000 images the splits need.
- [x] Write one run-all script that executes the session steps in order and saves results after every run.
- [ ] Obtain access to Llama 3.2 1B on Hugging Face and download its weights.

### GPU session (`python scripts/run_session.py`)

| Step | Work |
| --- | --- |
| 1 | Speed test: measure training samples per second for the reference configuration |
| 2 | Choose the largest training set (40,000, 30,000, 20,000, 15,000, 10,000 or 5,000 images) whose five runs fit the remaining budget with 10% held in reserve |
| 3 | Cache SigLIP 2 features for the chosen images |
| 4 | Learning-rate check: three short runs scored on the dev split |
| 5 | The five planned runs, each followed by its full evaluation; every run is capped in minutes so the remaining ones still fit |
| 6 | Extra runs if time is left |
| 7 | Tables and figures |

**Assumptions.** Training speed on Llama 3.2 1B has not been measured yet. The working estimate is about 15 minutes per run on 20,000 images, with a plausible range of 7 to 35 minutes. Step 1 replaces this estimate with a measurement before any long run starts, and step 2 sizes the training set from it. A finished step is skipped on a rerun, so a crash costs at most one run. The session is not continued past 4 hours.

## 10. Deliverables

- **Final report.** An IEEE two-column conference paper built from the official IEEE template [17]: Title, Author, Abstract, Keywords, then Introduction, Related Work, Methodology, Experiments and Results, Ablation, Conclusion and IEEE-style References. It integrates all eight pipeline stages in the order of this proposal.
- **Source code.** Feature caching, the bridges, training, evaluation and the session script, with configuration files and seeds.
- **GroundProbe.** Question and answer files, the generation script and the hand-check sheet.
- **Results package.** Per-run result files, the comparison and ablation tables, and the figures listed in section 6.

## References

[1] J. Li, D. Li, S. Savarese, and S. Hoi, "BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2023. arXiv:2301.12597. https://arxiv.org/pdf/2301.12597

[2] M. Tschannen *et al.*, "SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features," arXiv:2502.14786, 2025. https://arxiv.org/pdf/2502.14786

[3] Meta, "Llama 3.2 1B Instruct," model card, 2024. [Online]. Available: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

[4] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, "Making the V in VQA matter: Elevating the role of image understanding in visual question answering," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017.

[5] A. Agrawal, D. Batra, D. Parikh, and A. Kembhavi, "Don't just assume; look and answer: Overcoming priors for visual question answering," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018.

[6] M. Tsimpoukelli, J. Menick, S. Cabi, S. M. A. Eslami, O. Vinyals, and F. Hill, "Multimodal few-shot learning with frozen language models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021.

[7] J.-B. Alayrac *et al.*, "Flamingo: A visual language model for few-shot learning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022.

[8] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual instruction tuning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[9] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved baselines with visual instruction tuning," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024.

[10] W. Dai *et al.*, "InstructBLIP: Towards general-purpose vision-language models with instruction tuning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[11] J. Cha, W. Kang, J. Mun, and B. Roh, "Honeybee: Locality-enhanced projector for multimodal LLM," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024.

[12] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, "Evaluating object hallucination in large vision-language models," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2023.

[13] E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.

[14] T.-Y. Lin *et al.*, "Microsoft COCO: Common objects in context," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2014.

[15] VQA Team, "VQA: Visual Question Answering, VQA v2 download page." [Online]. Available: https://visualqa.org/download.html. Accessed: Oct. 8, 2026.

[16] H. Liu *et al.*, "LLaVA model zoo." [Online]. Available: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md. Accessed: Oct. 8, 2026.

[17] IEEE, "IEEE manuscript templates for conference proceedings." [Online]. Available: https://www.ieee.org/conferences/publishing/templates.html

The published VQAv2 accuracies, the parameter counts of the compared models and the VQA v2 dataset sizes were read from [1], [15] and [16] on Oct. 8, 2026. The split and GroundProbe sizes are the counts produced by this project's own scripts.
