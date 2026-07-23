# MiniOneRec 模型工具

本目录提供 MediGen Direct-SID 推荐排序器的数据生成、基础模型下载、LoRA SFT、独立评估、产物封装和离线推理工具。

完整原理、数据设计、配置说明、训练结果、在线集成、真实验收、Git 大文件策略和故障排查见 [`docs/08-minionerec推荐.md`](../../docs/08-minionerec推荐.md)。

## 环境

以下命令均在 `python/` 目录执行：

```powershell
.\scripts\local.ps1 setup -RunTests

$python = if (Test-Path .\.venv-minionerec\python.exe) {
  '.\.venv-minionerec\python.exe'
} else {
  '.\.venv-minionerec\Scripts\python.exe'
}

& $python -m pip check
```

## 从数据到部署产物

```powershell
# 1. 校验稳定 Topic SID
& $python recommendation_model\migrate_topic_tokens.py --check

# 2. 生成并校验合成数据
& $python recommendation_model\build_synthetic_dataset.py `
  --config recommendation_model\config.yaml `
  --seed 42
& $python recommendation_model\validate_dataset.py `
  --config recommendation_model\config.yaml

# 3. 下载固定 revision 的基础模型
& $python recommendation_model\download_base_model.py `
  --model-id Qwen/Qwen2.5-0.5B `
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 `
  --output artifacts\base-models\qwen2.5-0.5b `
  --load-smoke

# 4. 使用本地模型训练、评估并封装
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

& $python recommendation_model\train_sft.py `
  --config recommendation_model\config.yaml
& $python recommendation_model\evaluate.py `
  --config recommendation_model\config.yaml `
  --checkpoint recommendation_model\runs\minionerec-mvp-v1 `
  --device cuda
& $python recommendation_model\package_artifact.py `
  --config recommendation_model\config.yaml

# 5. 对最终封装产物执行 GPU 与 CPU 离线推理
& $python recommendation_model\smoke_inference.py `
  --config recommendation_model\config.yaml `
  --artifact artifacts\minionerec-mvp\v1 `
  --device cuda
& $python recommendation_model\smoke_inference.py `
  --config recommendation_model\config.yaml `
  --artifact artifacts\minionerec-mvp\v1 `
  --device cpu
```

## 产物位置

| 路径 | 内容 |
|---|---|
| `data/recommendation/synthetic/` | train、validation、test 与数据 manifest |
| `recommendation_model/runs/` | checkpoint、optimizer、日志和训练中间结果 |
| `artifacts/base-models/qwen2.5-0.5b/` | 固定 revision 的基础模型 |
| `artifacts/minionerec-mvp/v1/` | 在线部署使用的 adapter、tokenizer、manifest 和报告 |

基础模型、adapter、tokenizer、checkpoint 和训练状态保存在本机或模型制品库，由 `.gitignore` 排除；Git 保存源码、配置、合成数据、manifest、指标和报告。

## 在线验收

```powershell
.\scripts\local.ps1 deploy -SkipInstall
Invoke-RestMethod http://127.0.0.1:8001/ready |
  ConvertTo-Json -Depth 6
.\scripts\demo-real.ps1 -Case none -ModelScenarios -RequireFallbackDisabled
.\scripts\demo-real.ps1 -Case all -ExpectedContent auto -RequireFallbackDisabled
```

严格主路径应返回 `ranking_strategy_used=mini_onerec_mvp`、`model_ready=true` 和模型版本 `minionerec-mvp-direct-sid-v1`。
