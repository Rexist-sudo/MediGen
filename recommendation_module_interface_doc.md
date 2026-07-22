# 医学知识推荐模块接口文档.md

本文档用于指导在现有 Python FastAPI 临床辅助决策系统中接入一个解耦的医学知识推荐模块。

本模块定位为 **API 层后处理**：在一次完整临床分析完成后，根据用户历史上下文及诊断结果，为用户推荐最多 3 个相关医学知识主题，并以知识卡片形式附加在原 API 回复中。

本阶段只定义可开发接口和最小接入边界，不强制确定最终推荐算法。`TopicGenerator` 可先使用 `mock` 或 `fallback_rule` 实现，后续再替换为 OneRec 服务。

---

## 1. 接入原则

### 1.1 接入位置

推荐模块只在 API 层后处理执行。

```text
POST /api/v1/clinical/analyze
  -> 原临床分析流程
  -> 得到 ClinicalState / AnalyzeResult
  -> RecommendationService.recommend_after_analysis(...)
  -> 将 education_recommendations 附加到 AnalyzeResponse
```

### 1.2 不修改部分

| 模块 | 是否修改 | 说明 |
|---|---:|---|
| `python/src/agents/` | 否 | 不新增推荐 Agent，不修改 Intake / Diagnosis / Treatment / Coding / Audit |
| `python/src/graph/clinical_pipeline.py` | 否 | 不新增 LangGraph 节点，不修改边或条件路由 |
| `python/src/graph/state.py` | 默认否 | 首期不要求在 `ClinicalState` 中增加推荐字段 |
| 诊断、治疗、编码、审计逻辑 | 否 | 推荐模块只消费结果，不影响原分析 |
| API Response | 是 | 增加推荐结果字段 |
| API Request | 可选是 | 增加推荐开关、用户历史上下文、推荐策略 |
| `python/src/models/` | 是 | 新增推荐相关 Pydantic 模型 |
| `python/src/services/` | 是 | 新增推荐服务目录 |

### 1.3 推荐功能边界

推荐模块功能：

```text
用户历史上下文 + 诊断结果
  -> 推荐上下文
  -> 最多 3 个医学知识主题
  -> 知识卡片
  -> 附加到 API 回复
```

---

## 2. 文件改动总览

### 2.1 必须新增文件

```text
python/src/models/recommendation.py
python/src/services/recommendation/__init__.py
python/src/services/recommendation/recommendation_service.py
python/src/services/recommendation/context_builder.py
python/src/services/recommendation/topic_generator.py
python/src/services/recommendation/topic_store.py
python/src/services/recommendation/card_renderer.py
python/data/recommendation/knowledge_topics.jsonl
python/tests/test_recommendation_service.py
```

### 2.2 必须修改文件

```text
python/src/api/routes.py
```

修改内容：

1. 引入推荐模型。
2. 扩展 `AnalyzeRequest`。
3. 扩展 `AnalyzeResponse`。
4. 在 `analyze_patient()` 中，`pipeline.invoke(...)` 完成后、构造响应前调用推荐服务。

### 2.3 可选修改文件

```text
python/src/config/settings.py
python/.env.example
python/tests/test_services.py
```

可选用途：

1. 增加推荐模块开关。
2. 增加 OneRec 服务 URL。
3. 增加推荐测试入口。

---

## 3. 主接口改动：`POST /api/v1/clinical/analyze`

### 3.1 接口位置

```text
python/src/api/routes.py::analyze_patient()
```

### 3.2 当前链路插入点

推荐模块插入在 `pipeline.invoke(...)` 之后。

```python
state = pipeline.invoke(
    {"raw_input": request.patient_description},
    thread_id,
)

# 新增：API 层推荐后处理
recommendation_result = None
if request.include_recommendations:
    recommendation_result = recommendation_service.recommend_after_analysis(
        user_history_context=request.user_history_context,
        diagnosis_result=state.diagnosis,
        structured_result=state,
        strategy=request.recommendation_strategy,
        top_k=request.recommendation_top_k,
    )

return AnalyzeResponse(
    patient_info=state.patient_info,
    diagnosis=state.diagnosis,
    treatment_plan=state.treatment_plan,
    coding_result=state.coding_result,
    audit_result=state.audit_result,
    errors=state.errors,
    education_recommendations=recommendation_result,
)
```

### 3.3 修改前后链路对比

修改前：

```text
analyze_patient()
  -> get_pipeline()
  -> pipeline.invoke(...)
  -> AnalyzeResponse
```

修改后：

```text
analyze_patient()
  -> get_pipeline()
  -> pipeline.invoke(...)
  -> RecommendationService.recommend_after_analysis(...)
  -> AnalyzeResponse + education_recommendations
```

---

## 4. 请求参数定义

### 4.1 `AnalyzeRequest` 新增字段

位置：

```text
python/src/api/routes.py
```

建议新增字段：

```python
class AnalyzeRequest(BaseModel):
    patient_description: str
    thread_id: Optional[str] = None

    # 新增：是否启用推荐模块
    include_recommendations: bool = True

    # 新增：用户历史上下文，可为空
    user_history_context: Optional[UserHistoryContext] = None

    # 新增：推荐策略，首期建议 fallback_rule
    recommendation_strategy: Literal["mock", "fallback_rule", "onerec"] = "fallback_rule"

    # 新增：推荐数量，上限固定为 3
    recommendation_top_k: int = Field(default=3, ge=0, le=3)
```

### 4.2 `UserHistoryContext`

位置：

```text
python/src/models/recommendation.py
```

类型定义：

```python
class UserHistoryContext(BaseModel):
    user_id: Optional[str] = None
    viewed_topics: list[str] = []
    clicked_topics: list[str] = []
    saved_topics: list[str] = []
    dismissed_topics: list[str] = []
    preferred_categories: list[str] = []
```

字段说明：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `user_id` | `Optional[str]` | 否 | `None` | 用户标识，可为空 |
| `viewed_topics` | `list[str]` | 否 | `[]` | 用户历史浏览过的知识主题 |
| `clicked_topics` | `list[str]` | 否 | `[]` | 用户点击过的知识主题 |
| `saved_topics` | `list[str]` | 否 | `[]` | 用户保存 / 收藏的主题 |
| `dismissed_topics` | `list[str]` | 否 | `[]` | 用户忽略或关闭的主题 |
| `preferred_categories` | `list[str]` | 否 | `[]` | 用户偏好的知识类别 |

### 4.3 请求 JSON 示例

```json
{
  "patient_description": "患者 56 岁，近期口渴、乏力，既往有高血糖史，医生建议检查 HbA1c。",
  "thread_id": "thread_001",
  "include_recommendations": true,
  "recommendation_strategy": "fallback_rule",
  "recommendation_top_k": 3,
  "user_history_context": {
    "user_id": "user_001",
    "viewed_topics": ["hypertension_basics"],
    "clicked_topics": ["hba1c_test_explanation"],
    "saved_topics": [],
    "dismissed_topics": [],
    "preferred_categories": ["test_explanation", "disease_basics"]
  }
}
```

---

## 5. 响应参数定义

### 5.1 `AnalyzeResponse` 新增字段

位置：

```text
python/src/api/routes.py
```

新增字段：

```python
class AnalyzeResponse(BaseModel):
    patient_info: Optional[PatientInfo] = None
    diagnosis: Optional[DifferentialDiagnosis] = None
    treatment_plan: Optional[TreatmentPlan] = None
    coding_result: Optional[CodingResult] = None
    audit_result: Optional[AuditResult] = None
    errors: list[str] = []

    # 新增：推荐结果
    education_recommendations: Optional[EducationRecommendationResult] = None
```

说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `education_recommendations` | `Optional[EducationRecommendationResult]` | 推荐模块输出；未启用或失败时可为 `None` 或状态为 `failed` |

### 5.2 `EducationRecommendationResult`

位置：

```text
python/src/models/recommendation.py
```

类型定义：

```python
class EducationRecommendationResult(BaseModel):
    recommendations: list[KnowledgeRecommendation] = Field(default_factory=list, max_length=3)
    recommendation_status: Literal["ok", "degraded", "disabled", "failed"]
    strategy: Literal["mock", "fallback_rule", "onerec"]
    warnings: list[str] = []
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `recommendations` | `list[KnowledgeRecommendation]` | 是 | 推荐知识卡片列表，长度 `0-3` |
| `recommendation_status` | `Literal["ok", "degraded", "disabled", "failed"]` | 是 | 推荐执行状态 |
| `strategy` | `Literal["mock", "fallback_rule", "onerec"]` | 是 | 实际使用的推荐策略 |
| `warnings` | `list[str]` | 否 | 非阻断警告 |

状态枚举：

| 状态 | 说明 |
|---|---|
| `ok` | 推荐成功 |
| `degraded` | 主策略失败，使用 fallback |
| `disabled` | 请求关闭推荐或配置关闭推荐 |
| `failed` | 推荐失败，未返回推荐内容 |

### 5.3 `KnowledgeRecommendation`

位置：

```text
python/src/models/recommendation.py
```

类型定义：

```python
class KnowledgeRecommendation(BaseModel):
    rank: int = Field(ge=1, le=3)
    topic_id: str
    title: str
    category: Literal[
        "disease_basics",
        "test_explanation",
        "medication_knowledge",
        "lifestyle_education",
        "follow_up_education",
        "care_process_explanation",
        "prevention_education",
    ]
    reason: str
    summary: str
    source: Optional[str] = None
    source_url: Optional[str] = None
    safety_note: str = "该内容仅用于医学知识学习，不能替代医生诊断或治疗建议。"
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `rank` | `int` | 是 | 推荐排序，范围 `1-3` |
| `topic_id` | `str` | 是 | 知识主题 ID |
| `title` | `str` | 是 | 推荐卡片标题 |
| `category` | `Literal[...]` | 是 | 推荐类别 |
| `reason` | `str` | 是 | 推荐原因，需基于诊断结果或用户历史上下文 |
| `summary` | `str` | 是 | 知识主题摘要 |
| `source` | `Optional[str]` | 否 | 来源名称 |
| `source_url` | `Optional[str]` | 否 | 来源链接 |
| `safety_note` | `str` | 是 | 固定医学安全声明 |

### 5.4 响应 JSON 示例

```json
{
  "patient_info": {},
  "diagnosis": {},
  "treatment_plan": {},
  "coding_result": {},
  "audit_result": {},
  "errors": [],
  "education_recommendations": {
    "recommendation_status": "ok",
    "strategy": "fallback_rule",
    "warnings": [],
    "recommendations": [
      {
        "rank": 1,
        "topic_id": "diabetes_basics",
        "title": "糖尿病基础知识",
        "category": "disease_basics",
        "reason": "诊断结果中包含糖尿病相关内容。",
        "summary": "帮助用户理解糖尿病的基本概念、常见检查和长期健康管理方向。",
        "source": "MedlinePlus",
        "source_url": null,
        "safety_note": "该内容仅用于医学知识学习，不能替代医生诊断或治疗建议。"
      },
      {
        "rank": 2,
        "topic_id": "hba1c_test_explanation",
        "title": "HbA1c 检查是什么？",
        "category": "test_explanation",
        "reason": "诊断结果或检查建议中包含 HbA1c。",
        "summary": "介绍 HbA1c 的基本含义，以及它为什么常用于了解血糖管理情况。",
        "source": "MedlinePlus",
        "source_url": null,
        "safety_note": "该内容仅用于医学知识学习，不能替代医生诊断或治疗建议。"
      }
    ]
  }
}
```

---

## 6. 推荐服务内部接口

### 6.1 `RecommendationService`

文件：

```text
python/src/services/recommendation/recommendation_service.py
```

接口：

```python
class RecommendationService:
    def recommend_after_analysis(
        self,
        *,
        user_history_context: Optional[UserHistoryContext],
        diagnosis_result: Optional[DifferentialDiagnosis],
        structured_result: Optional[Any] = None,
        strategy: Literal["mock", "fallback_rule", "onerec"] = "fallback_rule",
        top_k: int = 3,
    ) -> EducationRecommendationResult:
        ...
```

参数说明：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `user_history_context` | `Optional[UserHistoryContext]` | 否 | 用户历史上下文 |
| `diagnosis_result` | `Optional[DifferentialDiagnosis]` | 是 | 诊断结果，推荐模块的主要输入 |
| `structured_result` | `Optional[Any]` | 否 | 完整结构化分析结果，首期可传 `ClinicalState` |
| `strategy` | `Literal["mock", "fallback_rule", "onerec"]` | 否 | 推荐策略 |
| `top_k` | `int` | 否 | 推荐数量，上限 3 |

返回：

```python
EducationRecommendationResult
```

异常约定：

- 不向 `analyze_patient()` 抛出未处理异常。
- 推荐服务内部捕获异常并返回：

```python
EducationRecommendationResult(
    recommendations=[],
    recommendation_status="failed",
    strategy=strategy,
    warnings=[str(exc)],
)
```

---

### 6.2 `ContextBuilder`

文件：

```text
python/src/services/recommendation/context_builder.py
```

接口：

```python
class ContextBuilder:
    def build(
        self,
        *,
        user_history_context: Optional[UserHistoryContext],
        diagnosis_result: Optional[DifferentialDiagnosis],
        structured_result: Optional[Any] = None,
    ) -> RecommendationContext:
        ...
```

输出类型：

```python
class RecommendationContext(BaseModel):
    user_history_context: Optional[UserHistoryContext] = None
    diagnosis_terms: list[str] = []
    diagnosis_codes: list[str] = []
    recommended_tests: list[str] = []
    medication_names: list[str] = []
    medication_classes: list[str] = []
    symptoms: list[str] = []
    audit_safe: bool = True
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `diagnosis_terms` | `list[str]` | 诊断名称、候选诊断名称等 |
| `diagnosis_codes` | `list[str]` | ICD-10 或其他诊断编码 |
| `recommended_tests` | `list[str]` | 推荐检查名称 |
| `medication_names` | `list[str]` | 用药名称 |
| `medication_classes` | `list[str]` | 用药类别 |
| `symptoms` | `list[str]` | 与推荐相关的症状关键词 |
| `audit_safe` | `bool` | 是否适合使用完整上下文生成推荐 |

---

### 6.3 `TopicGenerator`

文件：

```text
python/src/services/recommendation/topic_generator.py
```

接口：

```python
class TopicGenerator:
    def generate(
        self,
        *,
        context: RecommendationContext,
        strategy: Literal["mock", "fallback_rule", "onerec"],
        top_k: int = 3,
    ) -> TopicGenerationResult:
        ...
```

输出类型：

```python
class TopicGenerationResult(BaseModel):
    topic_ids: list[str] = Field(default_factory=list, max_length=3)
    strategy: Literal["mock", "fallback_rule", "onerec"]
    warnings: list[str] = []
```

策略说明：

| 策略 | 说明 |
|---|---|
| `mock` | 固定返回若干测试 topic，用于接口联调 |
| `fallback_rule` | 根据诊断、检查、用药规则匹配 topic |
| `onerec` | 调用 OneRec 服务生成 topic ID，当前可先预留 |

---

### 6.4 `TopicStore`

文件：

```text
python/src/services/recommendation/topic_store.py
```

接口：

```python
class TopicStore:
    def get_topic(self, topic_id: str) -> Optional[KnowledgeTopic]:
        ...

    def validate_topic_ids(self, topic_ids: list[str]) -> list[str]:
        ...
```

知识主题类型：

```python
class KnowledgeTopic(BaseModel):
    topic_id: str
    title: str
    category: Literal[
        "disease_basics",
        "test_explanation",
        "medication_knowledge",
        "lifestyle_education",
        "follow_up_education",
        "care_process_explanation",
        "prevention_education",
    ]
    related_codes: list[str] = []
    related_terms: list[str] = []
    summary: str
    source: Optional[str] = None
    source_url: Optional[str] = None
```

数据文件：

```text
python/data/recommendation/knowledge_topics.jsonl
```

示例：

```json
{"topic_id":"diabetes_basics","title":"糖尿病基础知识","category":"disease_basics","related_codes":["E11"],"related_terms":["diabetes","糖尿病"],"summary":"帮助用户理解糖尿病的基本概念、常见检查和长期健康管理方向。","source":"MedlinePlus","source_url":null}
```

---

### 6.5 `CardRenderer`

文件：

```text
python/src/services/recommendation/card_renderer.py
```

接口：

```python
class CardRenderer:
    def render(
        self,
        *,
        topic_ids: list[str],
        context: RecommendationContext,
        topic_store: TopicStore,
    ) -> list[KnowledgeRecommendation]:
        ...
```

渲染规则：

1. 每个 `topic_id` 必须存在于 `TopicStore`。
2. 每张卡片必须包含 `rank`、`topic_id`、`title`、`category`、`reason`、`summary`、`safety_note`。
3. `reason` 应基于 `RecommendationContext` 生成。
4. 不输出治疗指令、剂量建议、停药建议、急诊建议。
5. 最多返回 3 张卡片。

---

## 7. 可选独立接口：`POST /api/v1/clinical/recommendations`

首期不是必须。如果需要支持不重新跑临床分析而单独生成推荐，可新增该接口。

### 7.1 接口位置

```text
python/src/api/routes.py
```

### 7.2 Request

```python
class RecommendationRequest(BaseModel):
    user_history_context: Optional[UserHistoryContext] = None
    diagnosis_result: Optional[DifferentialDiagnosis] = None
    structured_result: Optional[dict] = None
    strategy: Literal["mock", "fallback_rule", "onerec"] = "fallback_rule"
    top_k: int = Field(default=3, ge=0, le=3)
```

### 7.3 Response

```python
class RecommendationResponse(BaseModel):
    education_recommendations: EducationRecommendationResult
```

### 7.4 路由伪代码

```python
@router.post(
    "/clinical/recommendations",
    response_model=RecommendationResponse,
)
async def recommend_education_topics(request: RecommendationRequest):
    result = recommendation_service.recommend_after_analysis(
        user_history_context=request.user_history_context,
        diagnosis_result=request.diagnosis_result,
        structured_result=request.structured_result,
        strategy=request.strategy,
        top_k=request.top_k,
    )
    return RecommendationResponse(education_recommendations=result)
```

---

## 8. 其他模块修改点

### 8.1 `python/src/api/routes.py`

修改内容：

```text
1. import 推荐模型
2. import RecommendationService
3. 扩展 AnalyzeRequest
4. 扩展 AnalyzeResponse
5. analyze_patient() 中 pipeline.invoke 后调用推荐服务
```

推荐初始化方式：

```python
recommendation_service = RecommendationService()
```

或按现有项目风格增加：

```python
def get_recommendation_service() -> RecommendationService:
    return RecommendationService()
```

---

### 8.2 `python/src/models/`

新增文件：

```text
python/src/models/recommendation.py
```

不建议修改现有 `patient.py`、`diagnosis.py`、`treatment.py`。

---

### 8.3 `python/src/services/`

新增目录：

```text
python/src/services/recommendation/
```

不修改现有服务逻辑。

可复用现有服务输出，但不反向依赖修改：

```text
icd10_service.py       可选：通过 ICD-10 code 匹配主题
hipaa_service.py       可选：推荐 reason 中避免 PHI
```

---

### 8.4 `python/src/graph/clinical_pipeline.py`

不修改。

明确不做以下改动：

```text
不 add_node("recommendation", ...)
不 add_edge("audit", "recommendation")
不改变 END 节点
不改变 diagnosis 后条件路由
```

---

### 8.5 `python/src/graph/state.py`

首期不修改。

原因：推荐结果只附加在 API response 中，不需要参与 Agent 状态传递。

后续如果需要记录推荐结果到状态中，可再新增：

```python
education_recommendations: Optional[EducationRecommendationResult] = None
```

但该项不是首期对接要求。

---

### 8.6 `python/src/config/settings.py`

可选修改。

建议新增配置：

```python
recommendation_enabled: bool = True
recommendation_default_strategy: str = "fallback_rule"
onerec_service_url: Optional[str] = None
recommendation_top_k: int = 3
```

`.env.example` 可选新增：

```env
RECOMMENDATION_ENABLED=true
RECOMMENDATION_DEFAULT_STRATEGY=fallback_rule
ONEREC_SERVICE_URL=
RECOMMENDATION_TOP_K=3
```

---

## 9. 错误处理约定

推荐模块错误不影响原临床分析接口。

| 场景 | API 行为 |
|---|---|
| `include_recommendations=false` | `education_recommendations.recommendation_status = "disabled"` |
| 推荐服务异常 | 返回 `recommendation_status = "failed"`，主分析结果照常返回 |
| OneRec 调用失败 | 降级到 `fallback_rule`，状态为 `degraded` |
| 生成 topic 不存在 | 过滤无效 topic，必要时补 fallback topic |
| 推荐结果为空 | 返回空列表，状态为 `failed` 或 `degraded` |
| `diagnosis_result=None` | 只基于用户历史或通用主题生成；若不可生成则返回空列表 |

失败响应示例：

```json
{
  "education_recommendations": {
    "recommendations": [],
    "recommendation_status": "failed",
    "strategy": "fallback_rule",
    "warnings": ["diagnosis_result is empty"]
  }
}
```

---

## 10. 最小测试用例

新增测试文件：

```text
python/tests/test_recommendation_service.py
```

必须覆盖：

```text
1. include_recommendations=true 时，AnalyzeResponse 包含 education_recommendations
2. include_recommendations=false 时，推荐状态为 disabled
3. RecommendationService 最多返回 3 条推荐
4. topic_id 必须存在于 knowledge_topics.jsonl
5. 推荐服务异常不影响主接口返回
6. strategy=mock 可返回固定推荐
7. strategy=fallback_rule 可根据 diagnosis code 匹配推荐
8. 返回卡片不包含治疗指令、剂量调整、停药、换药等内容
```

推荐服务单元测试示例：

```python
def test_recommend_after_analysis_returns_at_most_three_cards():
    service = RecommendationService()
    result = service.recommend_after_analysis(
        user_history_context=None,
        diagnosis_result=mock_diagnosis_result,
        structured_result=None,
        strategy="fallback_rule",
        top_k=3,
    )
    assert len(result.recommendations) <= 3
    assert result.recommendation_status in {"ok", "degraded", "failed"}
```

---

## 11. 开发顺序建议

建议按以下顺序开发：

```text
1. 新增 models/recommendation.py
2. 新增 knowledge_topics.jsonl
3. 实现 TopicStore
4. 实现 ContextBuilder
5. 实现 fallback_rule TopicGenerator
6. 实现 CardRenderer
7. 实现 RecommendationService
8. 修改 routes.py 的 AnalyzeRequest / AnalyzeResponse
9. 在 analyze_patient() 中接入后处理调用
10. 增加测试
```

OneRec 接入顺序后置：

```text
1. 保留 TopicGenerator.generate(strategy="onerec") 分支
2. 定义 onerec client 调用协议
3. 后续用 OneRec 服务替换 topic_ids 生成逻辑
4. 不改 API 契约
```

---

## 12. 最终对接契约摘要

### 修改位置

```text
API 层：python/src/api/routes.py
Models 层：python/src/models/recommendation.py
Services 层：python/src/services/recommendation/
Data：python/data/recommendation/knowledge_topics.jsonl
Tests：python/tests/test_recommendation_service.py
```

### 不修改位置

```text
python/src/agents/
python/src/graph/clinical_pipeline.py
python/src/graph/state.py  # 首期不改
```

### 输入

```text
user_history_context: Optional[UserHistoryContext]
diagnosis_result: Optional[DifferentialDiagnosis]
structured_result: Optional[ClinicalState | dict]
strategy: Literal["mock", "fallback_rule", "onerec"]
top_k: int <= 3
```

### 输出

```text
education_recommendations: Optional[EducationRecommendationResult]
```

### 核心链路

```text
pipeline.invoke(...)
  -> state.diagnosis
  -> RecommendationService.recommend_after_analysis(...)
  -> EducationRecommendationResult
  -> AnalyzeResponse.education_recommendations
```

### 核心约束

```text
推荐模块只作为 API 层后处理；
不进入 LangGraph；
不新增 Agent；
不影响原诊断、治疗、编码、审计结果；
推荐最多 3 个医学知识主题；
推荐结果以知识卡片形式返回。
```
