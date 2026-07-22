# MediGen Python MVP

这是一个用于软件架构演示的多节点原型：FastAPI 接收合成、去标识化文本，依次执行 Intake、Diagnosis、Treatment、Coding、Audit，再在 LangGraph 外执行本地确定性教育内容推荐。

> This software is a prototype for software architecture demonstration only. It is not a medical device and must not be used for diagnosis or treatment. Use synthetic, de-identified data only.

实现基线：`cd5636d7b2f063b7a77c6cf8b05248c0cbd6639f`。

## MVP 边界

- 默认 LLM 路径是 DeepSeek 官方兼容 API，模型 ID 固定默认为 `deepseek-v4-pro`，thinking 显式关闭。
- `fixture` 是无网络、无模型的固定合成演示后端，只用于本机验证接线；响应会明确返回 `llm_backend: "fixture"`。
- 推荐模块是 MiniOneRec-Lite-compatible contract 的 MVP 实现，实际排序器为本地确定性 `rule_v1`，没有模型训练、GPU 或自由医学正文生成。
- `/clinical/analyze` 不依赖 PostgreSQL、Redis、Neo4j、FHIR Server、GraphRAG 或 Docker Compose。
- 本项目未实现真实医学验证、真实 PHI 处理、HIPAA 合规、生产鉴权、持久化或生产部署。

## 本机启动

以下命令从 `python/` 目录运行。推荐 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-mvp.in
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

真实 DeepSeek 路径是默认值。编辑 `.env`，只需至少填写：

```env
LLM_BACKEND=deepseek
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

然后启动：

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

无 Key 的本地接线演示可在当前终端显式覆盖后端，不需要改 `.env`：

```powershell
$env:LLM_BACKEND = "fixture"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

`fixture` 不代表 DeepSeek 已验证。重新打开终端或执行 `Remove-Item Env:LLM_BACKEND` 即恢复 `.env` 中的默认值。

## 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/clinical/analyze` | 有限五节点流程 + 图外推荐 |
| POST | `/api/v1/clinical/icd10/search` | 小型本地演示表搜索 |
| GET | `/api/v1/clinical/icd10/{code}` | 小型本地演示表查询 |
| POST | `/api/v1/clinical/ddi/check` | 小型本地交互表查询 |
| GET | `/health` | 仅进程存活，不探测 DeepSeek |
| GET | `/ready` | 检查 Key 配置与本地 TopicStore，不调用模型 |

Swagger：<http://127.0.0.1:8000/docs>

示例请求：

```powershell
$body = @{
  patient_description = "56-year-old adult with increased thirst and fatigue. Prior high glucose readings. A clinician suggested checking HbA1c."
  include_recommendations = $true
  recommendation_top_k = 3
  user_preferences = @{
    preferred_categories = @("test_explanation", "disease_basics")
    preferred_depth = "beginner"
    preferred_format = "bullet_points"
    max_reading_minutes = 3
  }
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/clinical/analyze `
  -ContentType application/json `
  -Body $body
```

包含明显 email、电话、SSN、IP 或病历号前缀的输入会在任何外部 LLM 调用前返回 422。该有限规则不等于完整去标识化。

## 测试

```powershell
python -m pip install pytest
python -m compileall src
python -m pytest tests -q
```

`requirements-mvp.in` 是最小依赖意图文件。`requirements-mvp.lock.txt` 会在使用合法 Key 完成真实 DeepSeek smoke 后生成；fixture smoke 不作为真实外部 API 兼容性或版本锁定证据。

## 本次实际验证记录

验证日期：2026-07-22。验证环境：Windows、Python 3.13.5、全新临时虚拟环境。

已执行：

```text
python -m pip install -r requirements-mvp.in pytest
python -m pip check
python -m compileall -q src tests
ruff check <本次新增和修改的 Python 文件>
python -m pytest tests -q
```

结果：依赖一致性检查通过，编译通过，Ruff 检查通过，50 个测试全部通过。测试运行产生 1 条来自当前 FastAPI/Starlette TestClient 的第三方弃用警告，不影响测试结果。

另以 `LLM_BACKEND=fixture` 在 `127.0.0.1:8765` 启动真实 Uvicorn 进程并通过 HTTP 验证：

- `/health`：200 / `healthy`；
- `/ready`：200 / `ready`，TopicStore 已加载；
- 完整合成样例：200 / `completed`，五类 Pipeline 输出齐全，推荐 `ok` 且返回 3 条；
- 信息不足样例：200 / `needs_more_info`，Treatment/Coding 为空并进入 Audit；
- 禁用推荐：200 / `disabled`；
- 含 email 样例：422 / `prototype_phi_not_allowed`。
- 缺失 TopicStore 的独立实例：`/ready` 报告 `not_ready`，但完整分析仍返回 200 / `completed`、保留全部临床字段，推荐单独返回 `degraded`。

未执行：真实 DeepSeek API 调用。当前没有合法 `DEEPSEEK_API_KEY`，因此不能声称外部模型已经跑通，也未生成依赖 lock 文件。
