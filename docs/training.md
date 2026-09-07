# Future training path

No fine-tuning was performed. Establish a pretrained Qwen baseline first. Available scaffolding: a JSONL data validator, lazy PEFT adapter factory and learned change-detector contract. These are not a working trainer.

## LoRA data contract

```json
{"image":"tiles/scene01.png","question":"Locate the visible river","answer":"A structured grounded answer","scene_id":"scene01","split":"train"}
```

Every record needs image/question/answer/scene_id strings and train/validation/test split. Preflight verifies file existence and disallows the same scene ID across splits.

```bash
python -m training.lora path/to/training.jsonl
```

On an isolated cloud environment, `.[ml,training]` supplies the optional stack. `training.lora.attach_lora(model)` creates a PEFT adapter on an **already-loaded** model: rank 8, alpha 16, dropout 0.05, q_proj/v_proj targets. These are starting settings, not optimized results. See the [PEFT quicktour](https://huggingface.co/docs/peft/v0.17.0/en/quicktour) for adapter mechanics.

Before a real training run:

1. Verify grounding annotations, rights, scene splits and hardware.
2. Implement/test Qwen multimodal collation, image token placement and visual fields. Mask padding, image and prompt tokens from supervised answer loss.
3. Inspect actual module names and choose language-only or visual-layer adaptation. The suffix-based factory is an integration hook, not a validated remote-sensing strategy.
4. Add an explicit-run trainer with bounded images, accumulation, checkpointing and a small dry-run batch. Never launch through app import.
5. Save adapter plus base-model commit, preprocessing, output schema, metrics, splits and seeds. Add explicit adapter-checkpoint loading to inference; it is not currently implemented.
6. Compare held-out answer/grounding/reliability against the frozen pretrained baseline, without selecting checkpoints on test data.

## Learned change and SAR

`python -m training.change` prints the future contract: aligned bands, common valid masks, binary labels, scene/event splits, preprocessing and checkpoint version. `LearnedChangeDetector` raises an explicit unavailable error. No Siamese U-Net or trained checkpoint exists in this repository.

SAR interpretation needs product/orbit/calibration ingestion, amplitude/intensity/dB handling, calibration, speckle treatment, validated encoders and measured optical/SAR consistency. Percentile display alone is not sufficient preprocessing for inference.
