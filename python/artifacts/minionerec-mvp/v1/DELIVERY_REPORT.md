# MiniOneRec MVP v1 交付报告

记录时间：2026-07-23（Asia/Shanghai）

## 基线与环境

| 项目 | 实际值 |
|---|---|
| 仓库基线 SHA | `c372249d2ad582c55ce4622b325521fb9ad402ce` |
| 操作系统 | Windows 11 家庭中文版 10.0.26200（Build 26200） |
| CPU / 内存 | Intel Core i7-12800HX / 31.7 GiB |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU，8,188 MiB |
| NVIDIA Driver | 572.83 |
| Python | 3.11.15 |
| PyTorch / CUDA | 2.7.1+cu128 / 12.8 |
| Transformers / PEFT | 4.53.2 / 0.16.0 |
| Tokenizers / Safetensors | 0.21.2 / 0.5.3 |

## 基础模型与不可变标识

| 项目 | 实际值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B` |
| resolved revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| snapshot SHA-256 | `467b704be1335bcb62b70e0fb16dd494d43b7fa5a74b89eea02a95f7dc6642cf` |
| base `model.safetensors` SHA-256 | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| Topic catalog SHA-256 | `d21429f89d91fbf1ddb47e5dac470ca0dc506b21bdba31052512329697ce8c0e` |
| Topic token map SHA-256 | `3ebac0df4722d889113f467afec1821c81ef17d7002b91847385af28e86d2485` |
| 数据集 SHA-256 | `c27c4ac4b098ca92468a3209088772ee25f622aa6f62670880ac661e05bc07df` |
| adapter SHA-256 | `3ad3ad732857f9bb0de4158ed51c7b2b3af7d41cfda68d9fc62a56cdfe4e5a70` |
| tokenizer SHA-256 | `c1e596897a35192552e98beebf92340c98acc1b0314c20f13e6b6af9cbb69224` |
| training config SHA-256 | `b6a9f0f6ca6af4f23d5a6bac99f44561e8ab52846f19485bd691b4f688fd419e` |

基础模型下载 manifest 位于 `../../base-models/qwen2.5-0.5b/base_model_manifest.json`；部署 manifest 位于 `model_manifest.json`。

## 数据

- seed：42；真实 PHI 标志：false。
- 600 个 scenario 按 scenario group 固定拆分为 480 train、60 validation、60 test，无 group 泄漏。
- 1,563 train 行、190 validation 行、198 test 行。
- 1,560 条 next-topic SFT，391 条 Topic metadata 到 SID 对齐样本。
- 180 个无历史场景、150 个负反馈场景、280 对反事实样例、24 个候选排列场景。
- test 中固定保留 20 对历史敏感样例。

数据验证命令：

```powershell
& $python recommendation_model\build_synthetic_dataset.py --config recommendation_model\config.yaml --seed 42
& $python recommendation_model\validate_dataset.py --config recommendation_model\config.yaml
```

## 训练

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $python recommendation_model\train_sft.py --config recommendation_model\config.yaml
```

LoRA 参数为 rank 8、alpha 16、dropout 0.05，目标层为 q/k/v/o projection；`embed_tokens` 与 `lm_head` 随新增 SID 一起训练。训练 batch 为 4，gradient accumulation 为 2，有效 batch 为 8，BF16，3 epoch，共 588 步。

| 指标 | 实际值 |
|---|---:|
| 训练耗时 | 1,439.263 秒 |
| train loss | 0.4465676 |
| 最终 validation loss | 0.0006933 |
| LoRA 参数 | 1,081,344 |
| 总可训练参数 | 272,915,200 |

## 独立测试集评估

```powershell
& $python recommendation_model\evaluate.py `
  --config recommendation_model\config.yaml `
  --checkpoint recommendation_model\runs\minionerec-mvp-v1 `
  --device cuda
```

| 指标 | 实际值 |
|---|---:|
| Top1Accuracy | 1.0 |
| HitRate@3 | 1.0 |
| NDCG@3 | 1.0 |
| HistoryPairFlipRate | 19/20 = 0.95 |
| AdapterEffect | PASS |
| AdapterMaxCandidateLogitChange | 27.375 |
| CandidateOrderStabilityRate | 1.0（6 对） |
| FirstCandidateCopyRate | 0.590909 |
| P50 / P95 | 275.981 / 317.667 ms |
| ValidTopicRate | 1.0 |
| 候选越界/重复/排除类别/负反馈/安全顺序/未知 SID | 全部 0 |

评估使用 158 条 next-topic test 行，只对每条样例剩余的允许候选计算 SID token logits。

## 封装与离线 smoke

```powershell
& $python recommendation_model\package_artifact.py --config recommendation_model\config.yaml
& $python recommendation_model\smoke_inference.py --config recommendation_model\config.yaml --artifact artifacts\minionerec-mvp\v1 --device cuda
& $python recommendation_model\smoke_inference.py --config recommendation_model\config.yaml --artifact artifacts\minionerec-mvp\v1 --device cpu
```

| 设备 | dtype | 加载 | 冷启动排序 | 历史排序 |
|---|---|---:|---:|---:|
| RTX 4070 Laptop | bfloat16 | 21,534.783 ms | 1,149.631 ms | 418.117 ms |
| CPU | float32 | 8,483.885 ms | 2,118.345 ms | 1,907.473 ms |

两种设备的冷启动顺序均为 `hba1c_test_explanation`、`diabetes_basics`、`follow_up_checklist`；加入历史后顺序变为 `diabetes_basics`、`hba1c_test_explanation`、`follow_up_checklist`。两次 smoke 均在 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1` 下完成。

## 自动测试

| 命令 | 结果 |
|---|---|
| `python -m pip check` | No broken requirements found |
| `python -m pytest -q` | 81 passed, 7 skipped，52.88 秒 |
| `python -m pytest -q -m model` | 6 passed, 82 deselected，54.25 秒 |
| `python -m pytest -q -m integration` | 1 passed, 87 deselected，30.39 秒 |
| `python -m ruff check src tests recommendation_model scripts/validate-real.py` | PASS |
| `python -m compileall -q src tests recommendation_model scripts/validate-real.py` | PASS |
| PowerShell parser / `node --check` | PASS |

测试环境的 PostgreSQL、Neo4j、Redis 与 HAPI FHIR 均由项目 Docker Compose 启动。

## 真实 API 验收

严格主路径配置使用 eager、strict、CUDA、fallback disabled。`/ready` 返回 artifact valid、loaded、模型版本 `minionerec-mvp-direct-sid-v1`，五项依赖均为 true。

| transcript | 场景 | 结果 |
|---|---|---|
| `.runtime/real-validation-model-scenarios.json` | 冷启动 39.95 秒；2 条历史 49.23 秒；dismiss 45.25 秒 | 全部 PASS；冷启动与历史首选不同；dismiss 的 HbA1c 主题被排除 |
| `.runtime/real-validation-clinical-cases-pass.json` | STEMI 74.6 秒；肺炎过敏 60.6 秒；心衰 57.2 秒 | 全部 PASS；MiniOneRec 主排序、FHIR 与 PostgreSQL 写入成功 |
| `.runtime/real-validation-artifact-missing.json` | 缺失 adapter 路径，心衰 61.92 秒 | PASS；`rule_v1_fallback`，`fallback_reason=artifact_missing`，临床链路完成 |

内容层按每次响应记录 `deepseek_generated` 或 `catalog_fallback`；内容失败不会改写排序结果或临床分析状态。

## 依赖与本地运维收敛验证

| 检查 | 结果 |
|---|---|
| requirements 入口 | 9 个文件收敛为 API、完整本地环境、Windows CUDA 三组 intent/lock，共 6 个文件 |
| 完整依赖树 | 119 个包收敛为 102 个；移除 `datasets`、`pandas`、`pyarrow`、`sentencepiece` 等未使用依赖链 |
| 锁文件总大小 | 451,719 bytes 降至 366,850 bytes |
| 本地生命周期 | 5 个安装/部署/启停入口收敛为 `scripts/local.ps1`；setup、deploy、status、stop 实测通过 |
| 本机环境清理 | 删除 Python 3.13 旧 `.venv`，主环境移除 17 个废弃包，共释放 596,221,222 bytes |
| 严格模型回归 | 冷启动 52.5 秒、历史 49.1 秒、dismiss 49.4 秒；三项及交叉断言通过 |
| Docker/API profile | 87 个包；镜像构建与 `src.api.main` 导入通过；镜像大小 207,498,099 bytes |

## 运行边界

- 模型覆盖本项目 15 个 active 教育主题，扩大目录时需要重新生成 SID、数据并训练。
- 训练数据由合成模板、人工反事实、偏好模板和规则弱监督组成，质量指标仅用于该固定目录与测试分布。
- GPU 同时只允许一个排序请求进入模型；等待超时、OOM、加载异常和非法输出均进入固定原因的规则回退。
- adapter 二进制约 1.52 GiB，由 artifact 分发或本地重训提供，Git 仅保存轻量 manifest、tokenizer、配置与报告。
- 所有输出用于教育主题排序，临床决策与内容均需专业人员复核。
