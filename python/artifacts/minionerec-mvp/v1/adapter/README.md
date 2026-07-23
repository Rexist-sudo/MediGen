---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
tags:
- lora
- direct-sid-ranking
---

# MediGen MiniOneRec Direct-SID LoRA

This adapter ranks the fixed MediGen education-topic SID catalog. 
It is loaded with the pinned local base-model revision and the 
tokenizer packaged beside the adapter.

- Model version: `minionerec-mvp-direct-sid-v1`
- Base model: `Qwen/Qwen2.5-0.5B`
- Base revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Output: constrained candidate Topic SID tokens
- Intended use: local education-topic ranking inside MediGen

The model does not generate clinical diagnoses, prescriptions, 
dosages, or education-card prose.
