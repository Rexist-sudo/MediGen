# MediGen Python

MediGen 是一个面向合成或去标识化病例的本地临床辅助分析工作台。FastAPI 接收叙述文本，LangGraph 依次执行信息提取、诊断分析、方案建议、医学编码和安全审计，随后生成与病例及阅读深度相匹配的教育内容卡片。

本项目用于软件工程与工作流验证，不属于医疗器械。所有分析结果均须由具备资质的专业人员复核；系统不声明 HIPAA 合规，也不接收真实身份信息。

## 已接入的运行组件

| 组件 | 主链路中的实际用途 |
|---|---|
| FastAPI + LangGraph + Pydantic | HTTP 接口、五节点有向流程、各节点结构化校验 |
| DeepSeek 兼容 API | 信息提取、诊断、方案、编码，以及候选主题的深度化教育正文 |
| Neo4j 5 | 保存疾病、症状、照护概念和教育主题关系；诊断与教育候选均执行 Cypher 检索 |
| Redis 7 | 请求频率限制、Neo4j 查询缓存和教育正文缓存 |
| PostgreSQL 16 + SQLAlchemy 2 | 保存完整临床会话、推荐结果、FHIR 写入信息和只追加审计记录 |
| Presidio + spaCy + 本地规则 | 在模型调用前扫描输入中的直接标识符 |
| `fhir.resources` + HAPI FHIR | 校验 FHIR 数据并向本地 HAPI 服务提交 transaction Bundle |
| 本地 ICD-10 / DRG / 药物安全规则 | 对模型结果进行确定性编码补充、现用药组合核查和药物过敏核查 |

教育主题排序保留 MiniOneRec-compatible 契约，当前采用可审计的本地 `rule_v1` 排序器；MiniOneRec 上游训练与推理实现不在本项目范围内。Neo4j 中的数据是项目维护的有限本地医学图谱，未导入 UMLS、SNOMED CT 或完整药品知识库。

## 本机部署

前置条件：Python 3.11 或更高版本、Docker Desktop，以及可用的 DeepSeek 兼容 API 参数。

在 `python/` 目录执行：

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
# 编辑 .env 中的 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL 和 DEEPSEEK_MODEL
.\scripts\deploy-local.ps1 -WithTests
```

部署脚本会完成以下工作：

1. 创建 `.venv` 并安装锁定依赖与 Presidio 使用的 spaCy 模型；
2. 启动 PostgreSQL、Neo4j、Redis 和 HAPI FHIR；
3. 等待四项数据服务通过就绪检查；
4. 启动 FastAPI，并等待 `/ready` 返回全部依赖就绪。

本机已经安装依赖时可执行：

```powershell
.\scripts\deploy-local.ps1 -SkipInstall
```

服务地址：

- 可视化工作台：<http://127.0.0.1:8001/>
- Swagger：<http://127.0.0.1:8001/docs>
- 进程存活：<http://127.0.0.1:8001/health>
- 组件就绪：<http://127.0.0.1:8001/ready>
- Neo4j Browser：<http://127.0.0.1:7474/>
- HAPI FHIR metadata：<http://127.0.0.1:8080/fhir/metadata>

停止 FastAPI：

```powershell
.\scripts\stop-local.ps1
```

数据服务由 Docker Compose 管理，可按需执行 `docker compose stop`。PostgreSQL、Neo4j 和 HAPI 使用命名卷保存本地数据。

## 可视化工作台

页面与 API 同源运行，提供：

- 病例叙述输入，以及三个经过完整性筛选的合成病例快速填入按钮；
- 教育内容类别、内容深度和卡片数量偏好；
- 信息提取、诊断分析、方案建议、医学编码、安全审计和教育内容六个结果区域；
- 每个节点的真实处理时长，以及面向临床工作的节点摘要、关键依据和形成结果；
- Neo4j 证据路径、本地编码验证、药物安全结果，以及 Presidio、Redis、PostgreSQL 和 HAPI FHIR 数据链路记录；
- 桌面与窄屏自适应布局、键盘可操作的阶段标签。

快速填入病例由 [`src/web/validation-cases.json`](src/web/validation-cases.json) 统一维护。页面和真实链路验收脚本使用同一份病例目录。

### 网页演示流程

1. 打开 <http://127.0.0.1:8001/>，确认页面显示三个快速填入病例。
2. 选择“前壁心梗与用药核查”“肺炎与过敏核查”或“急性失代偿性心衰”。病例全文会写入病情描述框。
3. 选择教育内容类别、内容深度和卡片数量；首次演示建议使用“标准深度”和 3 张卡片。
4. 点击“提交分析”。模型与本地数据服务会依次执行，实际耗时随模型响应而变化。
5. 返回结果后，先核对顶部诊断候选、分析状态和全链处理时长，再依次打开六个节点：
   - 信息提取：确认症状、病史、现用药、过敏、生命体征、查体、化验和诊断检查均已归档；
   - 诊断分析：确认诊断置信度、支持证据、鉴别诊断、建议检查和 Neo4j 图谱路径；
   - 方案建议：确认药物、非药物措施、随访计划，以及相互作用或过敏警示；
   - 医学编码：确认主要 ICD-10、DRG、本地目录校验和编码置信度；
   - 质量复核：确认标识符扫描、PostgreSQL 会话和 HAPI FHIR 资源写入；
   - 教育内容：确认主题来自 Neo4j 候选，正文深度与页面选择一致。
6. 三例的主要预期分别为：`I21.0 / DRG 280` 且命中 warfarin–aspirin 相互作用；`J18.1 / DRG 193` 且命中 amoxicillin 过敏警示；`I50.9 / DRG 291` 且完整呈现容量负荷相关证据。

## 真实病例族验证

服务全部就绪后执行：

```powershell
.\scripts\demo-real.ps1 -Case all
```

脚本会逐一提交以下合成病例：

- 前壁 STEMI：包含心电图、心肌损伤标志物，以及华法林与阿司匹林并用情境；
- 右下叶肺炎：包含影像与感染指标，以及青霉素过敏和阿莫西林暴露；
- 急性失代偿性心力衰竭：包含既往 CABG、查体、BNP、胸部影像和多药治疗信息。

每例都会检查五节点结果、诊断与编码置信度门槛、预期 ICD-10、本地 DRG、Neo4j 证据、三张标准深度教育卡片、Presidio 审计、Redis 链路、PostgreSQL 会话、FHIR transaction Bundle，以及对应的药物相互作用或过敏警示。完整请求、响应和逐项失败原因写入 `.runtime/real-validation-last.json`。

2026-07-22 的本机最终验收中，三例均通过：前壁 STEMI 82.1 秒、右下叶肺炎 54.5 秒、急性失代偿性心力衰竭 65.1 秒。该耗时包含模型结构校正重试，后续运行会随模型服务状态变化。

也可只运行一个病例：

```powershell
.\scripts\demo-real.ps1 -Case stemi_interaction
```

## 主要端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 可视化工作台 |
| POST | `/api/v1/clinical/analyze` | 五节点分析、教育内容、FHIR 写入与会话持久化 |
| POST | `/api/v1/clinical/icd10/search` | 本地 ICD-10 目录搜索 |
| GET | `/api/v1/clinical/icd10/{code}` | ICD-10 与 DRG 查询 |
| POST | `/api/v1/clinical/ddi/check` | 本地药物相互作用规则查询 |
| GET | `/health` | FastAPI 进程存活检查 |
| GET | `/ready` | 模型配置、主题目录与五项本地依赖检查 |

示例请求：

```powershell
$body = @{
  patient_description = "一名 45 岁男性合成病例，发热、咳黄色痰并有右下叶实变；青霉素过敏，当前服用 metformin 和 lisinopril。"
  include_recommendations = $true
  recommendation_top_k = 3
  user_preferences = @{
    preferred_categories = @("disease_basics", "test_explanation", "medication_safety")
    preferred_depth = "standard"
  }
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8001/api/v1/clinical/analyze `
  -ContentType application/json `
  -Body $body
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src tests scripts\validate-real.py
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts\validate-real.py
docker compose config --quiet
```

单元测试使用 `fixture` 后端隔离外部模型。真实验收必须使用 `LLM_BACKEND=deepseek`，并要求 PostgreSQL、Neo4j、Redis、HAPI FHIR 和 Presidio 全部就绪。

## 安全边界

- Presidio 与规则扫描用于阻断明显的姓名标签、邮箱、电话、证件号等字段，但自动检测无法保证完整去标识化；仅提交合成或已完成去标识化的数据。
- 本地医学图谱、ICD-10、DRG 与药物规则均为有限目录，输出供专业复核使用。
- HAPI FHIR starter 在本机提供互操作验证；当前部署未配置身份认证、授权或生产安全控制。
- `.env`、`.venv` 和 `.runtime` 均由 Git 忽略。验收记录不写入 API 密钥。
