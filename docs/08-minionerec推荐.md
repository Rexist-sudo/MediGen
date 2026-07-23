# 08 — MiniOneRec 推荐：架构、数据、训练、集成与运维

## 目录

- [1. 职责边界与在线架构](#1-职责边界与在线架构)
- [2. Direct-SID 排序与安全约束](#2-direct-sid-排序与安全约束)
- [3. Topic 目录与合成数据](#3-topic-目录与合成数据)
- [4. 模型环境、训练与产物](#4-模型环境训练与产物)
- [5. MediGen 集成与本机部署](#5-medigen-集成与本机部署)
- [6. 制品治理、目录演进与故障排查](#6-制品治理目录演进与故障排查)

---

## 1. 职责边界与在线架构

MiniOneRec 是 MediGen 教育主题推荐模块的本地排序器。五节点临床分析完成后，系统先召回审核过的 Topic，再依据结构化临床信号、用户偏好和历史交互生成最多 3 个主题的顺序。

推荐链路分为三个职责层：

| 层次 | 职责 | 主要实现 |
|---|---|---|
| 候选与安全策略 | Topic 召回、active 校验、强制安全主题、类别排除、负反馈过滤 | `TopicStore`、`CandidatePolicy` |
| 排序 | 在本轮允许候选中计算 Topic SID 顺序 | `MiniOneRecRanker`、`DirectCandidateTokenScorer` |
| 卡片正文 | 渲染目录卡片，并按配置生成扩展摘要 | `CardRenderer`、DeepSeek 内容层 |

### 在线数据流

```text
POST /api/v1/clinical/analyze
  -> LangGraph 五节点临床分析
  -> ContextBuilder：构造去标识化推荐上下文
  -> Neo4j 召回 Topic
       -> 查询异常时读取本地 TopicStore
  -> HistoryNormalizer：校验、去重、排序并截断历史
  -> CandidatePolicy
       -> active Topic
       -> mandatory_safety
       -> excluded_categories
       -> dismiss / not_helpful
       -> 最多 20 个候选
  -> RankerRouter
       -> MiniOneRecRanker
       -> RuleFallbackRanker
  -> OutputValidator
  -> CardRenderer：目录卡片
  -> DeepSeek：可选正文增强
  -> FHIR 与 PostgreSQL
  -> API 响应
```

### 输入与输出边界

模型读取：

- 主要诊断、鉴别诊断和 ICD-10 编码；
- 建议检查与方案中的药物名称；
- 审计安全状态和用户内容偏好；
- 最多 20 条有效 Topic 交互；
- 最多 20 个允许候选 Topic SID。

原始 `patient_description`、姓名、联系方式、病历号、诊断正文、处方正文和剂量文本不进入排序 prompt。模型输出空间限定为本轮允许的 Topic SID，临床诊断、处方、剂量和卡片正文由其他职责层处理。

### 两个独立降级边界

| 故障位置 | 处理方式 | 响应字段 |
|---|---|---|
| 模型关闭、产物缺失、加载失败、推理异常或输出非法 | 规则排序器接管候选顺序 | `ranking_strategy_used=rule_v1_fallback` |
| DeepSeek 超时、JSON/schema 异常、Topic 不完整或内容安全失败 | 保留排序，使用目录摘要 | `content_strategy_used=catalog_fallback` |

排序策略与正文策略分别记录。正文层异常不会改写 Topic 选择、排序结果或临床分析状态。

---

## 2. Direct-SID 排序与安全约束

### 稳定 Topic SID

每个教育 Topic 都对应一个稳定专用 token：

```text
<MED_TOPIC_0001>
<MED_TOPIC_0002>
...
<MED_TOPIC_0015>
```

映射由以下文件共同维护：

- `data/recommendation/knowledge_topics.jsonl`
- `data/recommendation/topic_token_map.json`
- `data/recommendation/catalog_manifest.json`

目录、SID 映射和 tokenizer 必须满足：

1. `topic_id` 与 SID 一一对应；
2. 已分配 SID 保持稳定；
3. 每个 SID 在 tokenizer 中编码为单个 token；
4. Topic 数量、目录哈希和映射哈希与模型 manifest 一致。

稳定 SID 将排序目标转化为离散 token，Topic 标题或目录摘要调整时无需改变既有标识。

### 历史事件

| 事件 | Token | 语义 |
|---|---|---|
| `view` | `<EV_VIEW>` | 浏览 |
| `save` | `<EV_SAVE>` | 收藏 |
| `helpful` | `<EV_HELPFUL>` | 有帮助 |
| `dismiss` | `<EV_DISMISS>` | 忽略 |
| `not_helpful` | `<EV_NOT_HELPFUL>` | 无帮助 |
| 无历史 | `<NO_HISTORY>` | 冷启动 |

历史事件经过时间归一化、去重和有效 Topic 校验后与 SID 组合。`dismiss` 和 `not_helpful` 先进入确定性候选过滤，再由模型处理剩余候选。

### Prompt 与逐位 Top-K

训练和在线推理共用 `minionerec_prompt.py`，字段顺序固定为：

```text
结构化临床上下文
用户偏好
时间排序的历史事件 + Topic SID
允许候选 SID
已选择 SID
下一 Topic SID
```

decoder 每轮只比较允许 SID 在最后位置的 token logits：

```text
第 1 轮：A、B、C、D -> B
第 2 轮：A、C、D    -> D
第 3 轮：A、C       -> A
最终顺序：B、D、A
```

已选 SID 会从下一轮候选中移除，因此结果不会重复，也不会产生目录外 Topic。

### 安全主题与输出校验

STEMI、心衰等场景可触发 `mandatory_safety` Topic。安全主题在模型排序前固定到首位，模型排列剩余位置。

`OutputValidator` 负责确认：

- Topic 存在且处于 active 状态；
- Topic 属于本轮允许候选；
- 无重复 SID、未知 SID 或目录越界；
- 无类别排除和负反馈冲突；
- 强制安全主题顺序正确；
- 返回数量符合 `top_k`。

校验失败会生成固定 `fallback_reason`，再由规则排序器输出合法顺序。

---

## 3. Topic 目录与合成数据

训练数据由审核 Topic 目录、YAML 场景模板和固定规则教师确定性生成，不调用外部语言模型，也不使用真实 PHI。

### 数据来源

| 输入 | 用途 |
|---|---|
| `knowledge_topics.jsonl` | Topic 标题、类别、目录摘要、安全提示和稳定 SID |
| `clinical_scenarios.yaml` | 诊断、编码、检查、药物和安全场景 |
| `history_patterns.yaml` | 冷启动、浏览、正反馈、负反馈和顺序敏感历史 |
| `preference_patterns.yaml` | 类别、深度、格式和阅读时长偏好 |
| 固定规则教师 | 生成受安全策略约束的目标顺序 |

### 样本结构

数据集混合两类 SFT 样本：

1. `next-topic`：给定上下文、历史、候选和已选 SID，预测下一 SID；
2. `topic-to-SID alignment`：给定审核 Topic 信息，输出对应稳定 SID。

默认任务比例为 80% 与 20%。Label masking 只对目标 SID 计算损失，prompt token 不进入监督标签。

生成器同时构造：

- 相同临床上下文、不同历史反馈的反事实样本；
- 相同候选集合、不同输入排列的稳定性样本；
- 冷启动、正反馈和负反馈样本；
- 强制安全主题样本；
- Topic 信息与 SID 对齐样本。

相互关联的反事实组、候选排列组和 Top-K 展开序列共享 `scenario_group_id`，并进入同一数据分区，避免关联样本跨区泄漏。

### 生成流程

以下命令在 `python/` 目录执行：

```powershell
$python = if (Test-Path .\.venv-minionerec\python.exe) {
  '.\.venv-minionerec\python.exe'
} else {
  '.\.venv-minionerec\Scripts\python.exe'
}

& $python recommendation_model\migrate_topic_tokens.py --check
& $python recommendation_model\build_synthetic_dataset.py `
  --config recommendation_model\config.yaml `
  --seed 42
& $python recommendation_model\validate_dataset.py `
  --config recommendation_model\config.yaml
```

数据校验覆盖 schema、Topic/SID 合法性、标签优先级、数据分区隔离、候选边界、重复项、负反馈、强制安全主题和 manifest 哈希。输出写入 `data/recommendation/synthetic/`，文件计数与 SHA-256 记录在 `data_manifest.json`。

---

## 4. 模型环境、训练与产物

### 依赖分层

| 文件 | 用途 |
|---|---|
| `requirements-app.in` / `requirements-app.lock.txt` | API 与 Docker 轻量环境 |
| `requirements.in` / `requirements.lock.txt` | 本地应用、模型、训练和维护工具 |
| `requirements-torch-cu128.in` / `requirements-torch-cu128.lock.txt` | Windows CPython 3.11、CUDA 12.8 PyTorch wheel |

训练脚本使用仓库内 `SFTDataset`。`tokenizers` 与 `safetensors` 由 Transformers/PEFT 依赖链锁定，安装过程使用 SHA-256 与 `--require-hashes`。

### 环境与基础模型

```powershell
Set-Location .\python
.\scripts\local.ps1 setup

$python = if (Test-Path .\.venv-minionerec\python.exe) {
  '.\.venv-minionerec\python.exe'
} else {
  '.\.venv-minionerec\Scripts\python.exe'
}

& $python recommendation_model\download_base_model.py `
  --model-id Qwen/Qwen2.5-0.5B `
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 `
  --output artifacts\base-models\qwen2.5-0.5b
```

基础模型使用固定 revision，下载脚本写入 manifest。后续训练和在线推理均读取本地快照。

### 核心训练配置

主配置位于 `recommendation_model/config.yaml`：

| 参数 | 值 |
|---|---:|
| Backbone | Qwen2.5-0.5B |
| 最大输入长度 | 1024 token |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q_proj`、`k_proj`、`v_proj`、`o_proj` |
| Modules to save | `embed_tokens`、`lm_head` |
| Epoch | 3 |
| Train / eval batch | 4 / 8 |
| Gradient accumulation | 2 |
| Learning rate | 0.0002 |
| Seed | 42 |
| 首选精度 | bfloat16 |

`embed_tokens` 和 `lm_head` 随 adapter 保存，用于保留新增 SID 与事件 token 的可训练参数。

### 训练与封装

```powershell
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
```

训练 run 写入 `recommendation_model/runs/minionerec-mvp-v1`。封装器读取完成的训练摘要、质量指标、adapter 和 tokenizer，生成在线加载目录：

```text
artifacts/minionerec-mvp/v1/
├── adapter/
├── tokenizer/
├── model_manifest.json
├── data_manifest.json
├── topic_token_map.json
├── training_config.yaml
├── training_summary.json
├── metrics.json
└── catalog_snapshot.sha256
```

`model_manifest.json` 记录基础模型 revision、adapter/tokenizer/catalog/dataset/config 哈希、依赖版本、训练设备、精度和输入上限。在线 loader 会在读取权重前完成兼容性校验。

---

## 5. MediGen 集成与本机部署

### RankerRouter 与模型加载

| 配置 | 行为 |
|---|---|
| `RECOMMENDATION_RANKER=auto` | 模型可用时使用 MiniOneRec，异常时按配置回退 |
| `RECOMMENDATION_RANKER=minionerec` | 指定 MiniOneRec 为主排序器 |
| `RECOMMENDATION_RANKER=rule_v1` | 使用确定性规则排序 |
| `RECOMMENDATION_RULE_FALLBACK_ENABLED=false` | 模型故障直接暴露为推荐层异常 |

常见模型故障原因包括：

- `model_disabled`
- `artifact_missing`
- `artifact_incompatible`
- `model_load_failed`
- `inference_timeout`
- `cuda_oom`
- `invalid_model_output`

API 顶层会隔离推荐异常，已经完成的临床分析、FHIR 和持久化流程继续保留。

模型 loader 的主要配置：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `MINIONEREC_ARTIFACT_PATH` | `./artifacts/minionerec-mvp/v1` | 最终产物目录 |
| `MINIONEREC_DEVICE` | `auto` | `auto`、`cuda` 或 `cpu` |
| `MINIONEREC_DTYPE` | `auto` | 根据设备选择精度 |
| `MINIONEREC_LOAD_POLICY` | `lazy` | 首次请求或启动时加载 |
| `MINIONEREC_READINESS_STRICT` | `false` | 是否要求模型已加载 |
| `MINIONEREC_INFERENCE_CONCURRENCY` | `1` | 同时进入模型的请求数 |
| `MINIONEREC_SEMAPHORE_WAIT_SECONDS` | `1` | 推理槽位等待上限 |
| `MINIONEREC_ALLOW_CPU` | `true` | CUDA 不可用时允许 CPU |

loader 校验 manifest 后设置离线模式，再加载本地基础模型、tokenizer 和 adapter。

### API 与持久化字段

推荐结果写入：

- `education_recommendations`
- `recommendation_status`
- `ranking_strategy_used`
- `content_strategy_used`
- `model_version`
- `model_ready`
- `fallback_reason`
- `ranker_inference_ms`
- `candidate_count`
- `valid_history_count`

相同信息进入 PostgreSQL 推荐 JSONB 和 `integration_trace.recommendation_ranker`。Trace 保存策略、版本、固定原因、耗时和计数，不保存 prompt、logits、原始历史、绝对模型路径或 Python 堆栈。

### 目录正文与 DeepSeek 正文

`CardRenderer` 先生成完整目录卡片，再尝试使用 DeepSeek 增强摘要：

```text
MiniOneRec 排序
  -> 目录卡片
  -> DeepSeek JSON + Pydantic schema
  -> Topic 完整性校验
  -> 动态内容安全校验
  -> 合格：deepseek_generated
  -> 异常：catalog_fallback
```

目录正文来自审核目录，包含标题、类别、摘要、排序原因、来源链接和安全提示，可直接形成完整响应。

DeepSeek 正文只允许返回已选 Topic 的 `topic_id` 与 `summary`。当 JSON 结构、Pydantic schema、Topic 覆盖或内容安全不合格时，系统使用目录正文；Topic 选择与顺序保持不变。该设计将动态文本限制在摘要增强范围内，并保证正文服务异常时仍有完整卡片。

### 本机部署

模型主路径可在 `python/.env` 中配置为：

```dotenv
RECOMMENDATION_RANKER=minionerec
RECOMMENDATION_RULE_FALLBACK_ENABLED=false
MINIONEREC_ENABLED=true
MINIONEREC_LOAD_POLICY=eager
MINIONEREC_READINESS_STRICT=true
MINIONEREC_DEVICE=auto
```

统一生命周期入口：

```powershell
Set-Location .\python

# 安装依赖、校验产物、启动数据服务和 API
.\scripts\local.ps1 deploy

# 复用已安装环境
.\scripts\local.ps1 deploy -SkipInstall

# 查看状态或管理 API
.\scripts\local.ps1 status
.\scripts\local.ps1 stop
.\scripts\local.ps1 start
```

`GET /ready` 的 `recommendation_model` 节点提供 `enabled`、`artifact_valid`、`loaded`、`status`、`device`、`dtype`、`model_version` 和 `fallback_enabled`。

规则回滚配置：

```dotenv
RECOMMENDATION_RANKER=rule_v1
MINIONEREC_ENABLED=false
```

---

## 6. 制品治理、目录演进与故障排查

### Git 与模型制品

Git 保存：

- Topic 目录、SID 映射和 catalog manifest；
- 数据模板、合成数据和 data manifest；
- 训练、评估、封装及在线集成代码；
- 依赖 intent 与 hash lock；
- 模型 manifest、配置、指标和交付说明；
- 本文档及组件 README。

本机或模型制品库保存：

- 基础模型快照；
- LoRA 权重和封装 tokenizer；
- trainer checkpoint；
- optimizer、scheduler 和 RNG 状态；
- 训练日志、虚拟环境、缓存和 `.runtime`。

对应路径由 `python/.gitignore` 排除。模型目录发布时，应将 adapter、tokenizer、manifest、Topic 映射和基础模型 revision 作为同一版本，并依据 manifest 中的 SHA-256 核验。

### Topic 目录演进

新增 Topic 时：

1. 在 `knowledge_topics.jsonl` 增加审核内容；
2. 分配新的 `<MED_TOPIC_NNNN>`，保留已有 SID；
3. 更新 Topic 映射和 catalog manifest；
4. 补充适用的临床、历史或偏好模板；
5. 重新生成数据；
6. 复用固定基础模型 revision 完成训练与封装；
7. 发布新的 artifact 版本并更新 `MINIONEREC_ARTIFACT_PATH`。

删除 Topic 时优先标记为 inactive，由候选策略过滤，以保留历史事件中的 SID 可解释性。

### 常见故障

| 现象 | 核心排查项 | 处理方向 |
|---|---|---|
| `artifact_missing` | manifest、adapter、tokenizer 路径 | 恢复完整产物目录或重新封装 |
| `artifact_incompatible` | catalog、SID map、adapter、tokenizer、revision 哈希 | 按同一模型版本整体发布 |
| `cuda_oom` | 显存占用、并发数、设备配置 | 保持并发 1，释放显存或切换 CPU |
| `rule_v1_fallback` | `fallback_reason`、`model_ready`、`/ready` | 依据固定原因检查产物、加载或推理 |
| `catalog_fallback` | DeepSeek JSON、schema、Topic 覆盖和内容安全 | 修正正文输出约束或继续使用目录摘要 |
| 封装失败 | 训练摘要、质量指标、adapter、tokenizer | 补齐同一 run 的完整文件 |

### 目录速查

```text
python/
├── data/recommendation/
│   ├── knowledge_topics.jsonl
│   ├── topic_token_map.json
│   ├── catalog_manifest.json
│   └── synthetic/
├── recommendation_model/
│   ├── config.yaml
│   ├── templates/
│   ├── schemas/
│   ├── build_synthetic_dataset.py
│   ├── validate_dataset.py
│   ├── download_base_model.py
│   ├── train_sft.py
│   ├── evaluate.py
│   └── package_artifact.py
├── artifacts/
│   ├── base-models/
│   └── minionerec-mvp/v1/
└── src/services/recommendation/
    ├── context_builder.py
    ├── history_normalizer.py
    ├── candidate_policy.py
    ├── minionerec_prompt.py
    ├── minionerec_decoder.py
    ├── model_loader.py
    ├── minionerec_ranker.py
    ├── ranker_router.py
    ├── output_validator.py
    ├── card_renderer.py
    └── service.py
```

MiniOneRec 的核心链路由稳定 Topic SID、确定性合成数据、固定基础模型 revision、LoRA SFT、候选内受约束排序、产物哈希校验和在线故障隔离组成。
