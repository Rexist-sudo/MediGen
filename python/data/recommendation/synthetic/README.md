# Synthetic Mini-OneRec training data

The JSONL files in this directory are generated from the reviewed 15-topic
catalog and the templates under `recommendation_model/templates`. They contain
structured fictional contexts, controlled preferences, and synthetic ordered
interactions. `scenario_group_id` keeps every counterfactual pair, candidate
permutation, and Top-K position inside one split.

Rebuild and validate from `python/`:

```powershell
python recommendation_model\build_synthetic_dataset.py --config recommendation_model\config.yaml --seed 42
python recommendation_model\validate_dataset.py --config recommendation_model\config.yaml
```

`data_manifest.json` records exact row counts and SHA256 hashes. The generator
does not call an external language model and the manifest declares
`contains_real_phi=false`.

