# MediGen Python

MediGen 是面向合成或去标识化病例的本地临床辅助分析工作台。FastAPI 接收叙述文本，LangGraph 执行信息提取、诊断分析、方案建议、医学编码和安全审计；随后由 Neo4j 召回教育主题，本地 MiniOneRec Direct-SID 模型完成候选排序，目录或 DeepSeek 生成卡片正文，最终写入 HAPI FHIR 与 PostgreSQL。

项目定位为软件工程、模型集成和临床工作流验证。所有分析结果均须由具备资质的专业人员复核；系统不声明 HIPAA 合规，只接收合成或已完成去标识化的数据。

## 运行组件

| 组件 | 主链路用途 |
|---|---|
| FastAPI + LangGraph + Pydantic | HTTP 接口、五节点流程、结构校验与就绪检查 |
| DeepSeek 兼容 API | 信息提取、诊断、方案、编码和可选教育正文 |
| MiniOneRec Direct-SID | Qwen2.5-0.5B + LoRA，对允许的 Topic SID token logits 逐位 Top-K |
| Neo4j 5 | 疾病、症状、照护概念和教育主题关系；诊断证据与推荐候选均执行 Cypher 查询 |
| Redis 7 | 请求限流、图谱查询缓存和教育正文缓存 |
| PostgreSQL 16 + SQLAlchemy 2 | 保存临床会话、推荐策略、模型元数据、FHIR 结果和只追加审计记录 |
| Presidio + spaCy + 本地规则 | 模型调用前扫描直接标识符 |
| `fhir.resources` + HAPI FHIR | 校验并提交 FHIR transaction Bundle |
| 本地 ICD-10 / DRG / 用药规则 | 编码补充、药物组合核查和过敏核查 |

Neo4j 使用项目维护的有限医学图谱；ICD-10、DRG、药品和教育主题也采用仓库内固定目录。

## MiniOneRec 排序链路

```text
结构化诊断/编码/检查/药物
  + 用户偏好
  + 最多 20 条交互事件
  -> Neo4j 候选
  -> active / 强制安全 / 排除类别 / 负反馈策略
  -> PromptBuilder
  -> 本地 Qwen2.5-0.5B + LoRA
  -> 允许候选的单 Token SID logits
  -> 逐位 Top-K 与输出校验
  -> 固定知识卡片
```

15 个 Topic 分配 `<MED_TOPIC_0001>` 至 `<MED_TOPIC_0015>`。排序模型只读取结构化信号与 Topic SID，不读取原始 `patient_description`，输出空间也只包含允许候选 SID。诊断、处方、剂量和医学正文由临床流程及卡片内容层处理。

响应分别记录：

- `ranking_strategy_used`：`mini_onerec_mvp` 或 `rule_v1_fallback`；
- `content_strategy_used`：`deepseek_generated` 或 `catalog_fallback`；
- `model_version`、`model_ready`、`fallback_reason`、`ranker_inference_ms`；
- `candidate_count`、`history_used`、`valid_history_count`。

模型设计、数据、训练、部署和验收见 [`docs/08-minionerec推荐.md`](../docs/08-minionerec推荐.md)，常用命令见 [`recommendation_model/README.md`](recommendation_model/README.md)，完整结果见 [`artifacts/minionerec-mvp/v1/DELIVERY_REPORT.md`](artifacts/minionerec-mvp/v1/DELIVERY_REPORT.md)。

## 本机部署

前置条件：Python 3.11、Docker Desktop、可用的 DeepSeek 兼容 API 参数；仓库提供的 GPU 依赖锁对应 CUDA 12.8。

在 `python/` 目录执行：

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
# 编辑 .env 中的 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL、DEEPSEEK_MODEL
.\scripts\local.ps1 deploy -RunTests
```

严格模型主路径配置：

```dotenv
RECOMMENDATION_RANKER=minionerec
RECOMMENDATION_RULE_FALLBACK_ENABLED=false
MINIONEREC_ENABLED=true
MINIONEREC_LOAD_POLICY=eager
MINIONEREC_READINESS_STRICT=true
MINIONEREC_DEVICE=auto
```

部署脚本执行以下步骤：

1. 创建 Python 3.11 `.venv-minionerec`；
2. 使用 `--require-hashes` 安装 CUDA PyTorch 锁和统一应用、模型、训练及验证依赖锁；
3. 加载 spaCy `en_core_web_sm`；
4. 校验本地基础模型、adapter、tokenizer 和 manifest，并做离线 smoke；
5. 启动 PostgreSQL、Neo4j、Redis 与 HAPI FHIR；
6. 启动 FastAPI，等待 `/ready` 的模型与五项依赖全部就绪。

`requirements.lock.txt` 是本地开发、训练和验收入口，`requirements-app.lock.txt` 是 Docker/API 轻量入口，`requirements-torch-cu128.lock.txt` 单独固定 Windows CUDA wheel。三个锁均由对应 `.in` 文件生成。

依赖已经安装时：

```powershell
.\scripts\local.ps1 deploy -SkipInstall
```

服务地址：

- 工作台：<http://127.0.0.1:8001/>
- Swagger：<http://127.0.0.1:8001/docs>
- liveness：<http://127.0.0.1:8001/health>
- readiness：<http://127.0.0.1:8001/ready>
- Neo4j Browser：<http://127.0.0.1:7474/>
- HAPI FHIR metadata：<http://127.0.0.1:8080/fhir/metadata>

停止 FastAPI：

```powershell
.\scripts\local.ps1 stop
```

规则回滚 profile：

```dotenv
RECOMMENDATION_RANKER=rule_v1
MINIONEREC_ENABLED=false
```

## 真实链路验收

模型专项场景：

```powershell
.\scripts\demo-real.ps1 `
  -Case none `
  -ModelScenarios `
  -RequireFallbackDisabled `
  -OutputPath .runtime\real-validation-model-scenarios.json
```

该命令提交同一糖尿病上下文的冷启动、两条历史交互和 dismiss 负反馈请求，检查模型主路径、历史计数、首选变化和负反馈排除。

三类临床病例：

```powershell
.\scripts\demo-real.ps1 `
  -Case all `
  -ExpectedContent auto `
  -RequireFallbackDisabled `
  -OutputPath .runtime\real-validation-clinical-cases.json
```

`auto` 逐响应接受 `deepseek_generated` 或 `catalog_fallback`，同时要求每张卡片的 `content_source` 与响应策略一致。排序仍须为 `mini_onerec_mvp`，强制安全主题、FHIR、PostgreSQL、Neo4j、Redis 和 Presidio 断言保持开启。

| 病例 | 核心断言 |
|---|---|
| 前壁 STEMI | `I21.0`、DRG `280`、warfarin–aspirin 相互作用、心梗警示首位 |
| 右下叶肺炎 | `J18.1`、DRG `193`、amoxicillin 与青霉素过敏风险 |
| 急性失代偿性心衰 | `I50.9`、DRG `291`、容量负荷证据、心衰警示首位 |

2026-07-23 本机验收结果：

- 冷启动 39.95 秒、历史 49.23 秒、负反馈 45.25 秒，模型专项全部通过；
- STEMI 74.6 秒、肺炎过敏 60.6 秒、心衰 57.2 秒，三病例全部通过；
- 缺失 adapter 故障注入 61.92 秒，临床分析完成，排序切换为 `rule_v1_fallback`，原因 `artifact_missing`。

## 测试

```powershell
$python = if (Test-Path .\.venv-minionerec\python.exe) {
  '.\.venv-minionerec\python.exe'
} else {
  '.\.venv-minionerec\Scripts\python.exe'
}

& $python -m pip check
& $python -m pytest -q
& $python -m pytest -q -m model
& $python -m pytest -q -m integration
& $python -m ruff check src tests recommendation_model scripts\validate-real.py
& $python -m compileall -q src tests recommendation_model scripts\validate-real.py
docker compose config --quiet
```

本次结果：主测试 81 passed / 7 skipped；真实模型 6 passed；PostgreSQL 集成 1 passed。模型测试在显式 offline 环境加载封装后的 adapter 与 tokenizer。

## 主要端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 可视化工作台 |
| POST | `/api/v1/clinical/analyze` | 五节点分析、教育推荐、FHIR 与持久化 |
| POST | `/api/v1/clinical/icd10/search` | 本地 ICD-10 搜索 |
| GET | `/api/v1/clinical/icd10/{code}` | ICD-10 与 DRG 查询 |
| POST | `/api/v1/clinical/ddi/check` | 本地药物相互作用查询 |
| GET | `/health` | 进程存活 |
| GET | `/ready` | 模型产物、加载状态和五项本地依赖 |

## 安全边界

- Presidio 与规则扫描阻断常见姓名标签、邮箱、电话、证件号等直接标识符；输入限定为合成或已去标识化病例。
- 强制安全主题、排除类别、负反馈、active 状态和最大返回数由确定性策略执行。
- API trace 仅记录策略、版本、固定原因、耗时与计数，不包含 prompt、logits、原始历史、adapter 绝对路径或 Python 堆栈。
- 本地目录与合成训练分布限定了覆盖范围，结果用于教育与工程验证并接受专业复核。
- `.env`、虚拟环境、运行日志和大型权重由 Git 忽略；API 密钥不写入验收 transcript。
