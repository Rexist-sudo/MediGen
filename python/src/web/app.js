"use strict";

const byId = (id) => document.getElementById(id);

const elements = {
  form: byId("analysis-form"),
  description: byId("patient-description"),
  characterCount: byId("character-count"),
  validation: byId("validation-message"),
  analyzeButton: byId("analyze-button"),
  buttonLabel: document.querySelector("#analyze-button .button-label"),
  conversation: byId("conversation-log"),
  resultsEmpty: byId("results-empty"),
  loadingState: byId("loading-state"),
  resultsView: byId("results-view"),
  resultSummary: byId("result-summary"),
  resultTime: byId("result-time"),
  pipelineNav: byId("pipeline-nav"),
  stagePanel: byId("stage-panel"),
  issuePanel: byId("issue-panel"),
  elapsedTime: byId("elapsed-time"),
};

let demoCases = {};
const validationCasesPromise = fetch("/static/validation-cases.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((cases) => {
    demoCases = Object.fromEntries(cases.map((item) => [item.id, item.description]));
    return demoCases;
  });

const labels = {
  analysisStatus: {
    completed: "整链完成",
    needs_more_info: "需要补充信息",
    partial: "部分完成",
  },
  gender: {
    male: "男",
    female: "女",
    other: "其他",
    unknown: "未知",
  },
  severity: {
    mild: "轻度",
    moderate: "中度",
    severe: "重度",
    critical: "危重",
    none: "无",
    minor: "轻微",
    major: "严重",
    contraindicated: "禁忌",
  },
  category: {
    disease_basics: "疾病基础",
    test_explanation: "检查解读",
    medication_safety: "用药安全",
    lifestyle_education: "生活方式",
    follow_up_education: "随访准备",
    care_process: "就医流程",
    warning_signs: "警示信号",
  },
  checks: {
    presidio_and_rule_identifier_scan: "身份标识扫描",
    structured_section_coverage: "结果区域覆盖",
  },
  gaps: {
    patient_narrative_required: "需要病例描述",
    structured_patient_information_required: "需要可结构化的病例信息",
    intake_structured_output_unavailable: "信息提取结果异常",
    diagnosis_structured_output_unavailable: "诊断结构化结果异常",
    additional_synthetic_clinical_context_required: "需要补充更多合成临床背景",
  },
  systemCodes: {
    identifier_scan_clear: "身份标识扫描通过",
    direct_identifiers_detected: "发现需复核的身份标识符",
    direct_identifier_review_required: "复核身份标识扫描结果",
    audit_service_error: "安全审计服务异常",
    recommendation_unavailable: "教育内容服务异常",
    education_content_generation_fallback: "已采用主题基础内容",
    knowledge_graph_catalog_fallback: "已采用本地主题目录",
    knowledge_graph_candidates_empty: "已采用通用教育主题",
    audit_result_missing: "安全审计结果缺失",
    fixture_demo_only: "离线知识源",
    mandatory_safety_override: "安全主题优先展示",
    unknown_history_topic_ignored: "已忽略目录外历史主题",
    history_missing_timestamps: "部分历史缺少时间信息",
    unsafe_context: "推荐上下文采用固定安全路径",
    ranker_fallback: "本地排序器已切换到规则回退",
  },
  rankingStrategy: {
    mini_onerec_mvp: "本地 Mini-OneRec",
    rule_v1_fallback: "规则回退",
    none: "固定安全顺序",
  },
  contentStrategy: {
    deepseek_generated: "DeepSeek 教育正文",
    catalog_fallback: "主题目录正文",
    none: "无正文",
  },
  fallbackReason: {
    model_disabled: "模型配置关闭",
    artifact_missing: "模型产物缺失",
    artifact_incompatible: "模型产物与目录不匹配",
    model_load_failed: "模型加载失败",
    model_not_ready: "模型处于冷却状态",
    unsafe_context: "审计结果限制模型路径",
    no_rankable_candidates: "没有可排序候选",
    inference_failed: "模型推理失败",
    cuda_oom: "显存不足",
    invalid_model_output: "模型输出校验失败",
    concurrency_busy: "模型推理并发繁忙",
  },
};

let currentResponse = null;
let activeStage = "diagnosis";
let loadingTimer = null;
let loadingStartedAt = 0;
let pendingMessage = null;

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function append(parent, ...children) {
  children.flat().forEach((child) => {
    if (child !== null && child !== undefined) parent.append(child);
  });
  return parent;
}

function hasValue(value) {
  return value !== null && value !== undefined && value !== "";
}

function displayValue(value, fallback = "—") {
  if (!hasValue(value)) return fallback;
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function asList(value) {
  return Array.isArray(value) ? value.filter((item) => hasValue(item)) : [];
}

function translated(value, dictionary, fallback) {
  if (!hasValue(value)) return fallback || "—";
  return dictionary[value] || String(value);
}

function readableCode(value) {
  if (!hasValue(value)) return "—";
  const text = String(value);
  return labels.gaps[text] || labels.systemCodes[text] || text.replaceAll("_", " ");
}

function codingCategory(value) {
  if (!hasValue(value)) return "辅助编码";
  return ["prototype", "fixture"].includes(String(value).toLowerCase())
    ? "辅助编码"
    : String(value);
}

function auditCheckDetail(item) {
  if (item.check_name === "presidio_and_rule_identifier_scan") {
    return item.passed
      ? "Presidio 与本地规则扫描通过"
      : "发现需人工复核的身份标识类型";
  }
  if (item.check_name === "structured_section_coverage") {
    const matched = /Reviewed\s+(\d+)\s+structured pipeline sections/i.exec(item.detail || "");
    return matched ? `已覆盖 ${matched[1]} 个结构化结果区域` : displayValue(item.detail);
  }
  return displayValue(item.detail);
}

function auditTrailDetail(item) {
  const matched = /Scanned\s+(\d+)\s+generated section/i.exec(item.detail || "");
  return matched ? `已扫描 ${matched[1]} 个结果区域` : displayValue(item.detail);
}

function confidenceValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.min(1, Math.max(0, numeric));
}

function makeBadge(text, variant = "") {
  return createElement("span", `mini-badge${variant ? ` ${variant}` : ""}`, text);
}

function makeDefinitionGrid(entries) {
  const list = createElement("dl", "definition-grid");
  entries.filter(([, value]) => hasValue(value)).forEach(([term, value]) => {
    const wrapper = createElement("div");
    append(
      wrapper,
      createElement("dt", "", term),
      createElement("dd", "", displayValue(value)),
    );
    list.append(wrapper);
  });
  return list;
}

function makeTags(values, variant = "") {
  const items = asList(values);
  const wrapper = createElement("div", "tag-list");
  items.forEach((value) => {
    const tag = createElement("span", `tag${variant ? ` ${variant}` : ""}`, displayValue(value));
    wrapper.append(tag);
  });
  return wrapper;
}

function makePlainList(values) {
  const items = asList(values);
  const list = createElement("ul", "plain-list");
  items.forEach((value) => list.append(createElement("li", "", displayValue(value))));
  return list;
}

function makeBlock(title, { wide = false, badge = "" } = {}) {
  const block = createElement("section", `data-block${wide ? " wide" : ""}`);
  const heading = createElement("div", "block-heading");
  heading.append(createElement("h4", "", title));
  if (badge) heading.append(makeBadge(badge));
  const body = createElement("div", "block-body");
  append(block, heading, body);
  return { block, body };
}

function makeStage(title, description, kind = "分析服务") {
  elements.stagePanel.replaceChildren();
  const header = createElement("header", "stage-header");
  const titleGroup = createElement("div");
  append(
    titleGroup,
    createElement("h3", "", title),
    createElement("p", "", description),
  );
  append(header, titleGroup, createElement("span", "stage-kind", kind));
  const grid = createElement("div", "content-grid");
  append(elements.stagePanel, header, grid);
  return grid;
}

function stageElapsedSeconds(response, stage) {
  const value = Number(response.processing_timeline?.stages?.[stage]?.elapsed_seconds);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function supportingElapsedSeconds(response, step) {
  const value = Number(response.processing_timeline?.supporting_steps?.[step]?.elapsed_seconds);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function totalElapsedSeconds(response, fallback = 0) {
  const value = Number(response.processing_timeline?.total_seconds);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function processFacts(stage, response) {
  const patient = response.patient_info || {};
  const diagnosis = response.diagnosis || {};
  const treatment = response.treatment_plan || {};
  const coding = response.coding_result || {};
  const audit = response.audit_result || {};
  const education = response.education_recommendations || {};

  if (stage === "intake") {
    const historyCount = [
      ...asList(patient.medical_history),
      ...asList(patient.surgical_history),
      ...asList(patient.family_history),
      ...asList(patient.social_history),
    ].length;
    const vitalCount = Object.values(patient.vital_signs || {}).filter(hasValue).length;
    const abnormalCount = [
      ...asList(patient.lab_results),
      ...asList(patient.diagnostic_studies),
    ].filter((item) => item?.is_abnormal).length;
    return [
      {
        label: "主诉与症状",
        value: `${asList(patient.symptoms).length} 项症状`,
        detail: displayValue(patient.chief_complaint, "主诉信息需补充"),
      },
      {
        label: "病史与用药",
        value: `${historyCount} 项背景记录`,
        detail: `当前用药 ${asList(patient.current_medications).length} 项 · 过敏记录 ${asList(patient.allergies).length} 项`,
      },
      {
        label: "客观资料",
        value: `${vitalCount} 项生命体征`,
        detail: `查体 ${asList(patient.physical_exam).length} 项 · 化验 ${asList(patient.lab_results).length} 项 · 诊断检查 ${asList(patient.diagnostic_studies).length} 项`,
      },
      {
        label: "异常线索",
        value: `${abnormalCount} 项`,
        detail: "异常化验与诊断检查已标记，供后续诊断研判使用",
      },
    ];
  }

  if (stage === "diagnosis") {
    const primary = diagnosis.primary_diagnosis || {};
    const graphEvidence = asList(diagnosis.knowledge_graph?.evidence);
    const confidence = confidenceValue(primary.confidence);
    return [
      {
        label: "主要判断",
        value: displayValue(primary.disease_name, "诊断信息需补充"),
        detail: confidence === null ? "诊断置信度需复核" : `诊断置信度 ${Math.round(confidence * 100)}%`,
      },
      {
        label: "支持依据",
        value: `${asList(primary.evidence).length} 条临床证据`,
        detail: `Neo4j 关联证据 ${graphEvidence.length} 项`,
      },
      {
        label: "鉴别范围",
        value: `${asList(diagnosis.differential_list).length} 项鉴别诊断`,
        detail: "按症状、检查结果与危险因素进行对照研判",
      },
      {
        label: "检查安排",
        value: `${asList(diagnosis.recommended_tests).length} 项建议`,
        detail: asList(diagnosis.recommended_tests).slice(0, 2).join("；") || "检查建议需结合临床补充",
      },
    ];
  }

  if (stage === "treatment") {
    return [
      {
        label: "治疗目标",
        value: displayValue(treatment.diagnosis_addressed, "诊断信息需补充"),
        detail: "药物、非药物措施与随访计划围绕当前诊断候选整理",
      },
      {
        label: "药物建议",
        value: `${asList(treatment.medications).length} 项`,
        detail: "已整理通用名、剂量、途径、频次、疗程与注意事项",
      },
      {
        label: "用药安全",
        value: `${asList(treatment.drug_interactions).length} 项相互作用`,
        detail: `警示 ${asList(treatment.warnings).length} 项 · 本地药物规则已复核`,
      },
      {
        label: "综合干预",
        value: `${asList(treatment.non_drug_treatments).length + asList(treatment.lifestyle_recommendations).length} 项`,
        detail: hasValue(treatment.follow_up_plan) ? "随访计划已形成" : "随访计划需补充",
      },
    ];
  }

  if (stage === "coding") {
    const primary = coding.primary_icd10 || {};
    const validation = coding.local_validation || {};
    return [
      {
        label: "主要编码",
        value: displayValue(primary.code, "编码需复核"),
        detail: displayValue(primary.description, "编码描述需补充"),
      },
      {
        label: "伴随编码",
        value: `${asList(coding.secondary_icd10_codes).length} 项`,
        detail: "结合既往病史与并存情况生成编码候选",
      },
      {
        label: "病组归类",
        value: coding.drg_group?.drg_code ? `DRG ${coding.drg_group.drg_code}` : "专业复核",
        detail: displayValue(coding.drg_group?.description, "分组需结合完整住院信息确认"),
      },
      {
        label: "本地核验",
        value: validation.primary_code_matched ? "主要编码通过" : "主要编码需复核",
        detail: validation.drg_prefix_matched ? "编码目录与 DRG 前缀校验通过" : "DRG 前缀需专业复核",
      },
    ];
  }

  if (stage === "audit") {
    const trace = response.integration_trace || {};
    return [
      {
        label: "身份标识扫描",
        value: audit.demo_safe ? "通过" : "需要人工复核",
        detail: `命中类别 ${asList(audit.phi_fields_found).length} 项`,
      },
      {
        label: "结果完整性",
        value: `${asList(audit.compliance_checks).filter((item) => item?.passed).length} 项检查通过`,
        detail: `审计轨迹 ${asList(audit.audit_trail).length} 条`,
      },
      {
        label: "临床记录",
        value: trace.persistence?.session_id ? "会话已保存" : "记录状态需检查",
        detail: `PostgreSQL 临床会话 ${displayValue(trace.persistence?.clinical_sessions, 0)} 条`,
      },
      {
        label: "标准化交换",
        value: `${displayValue(trace.interoperability?.resource_count, 0)} 项 FHIR 资源`,
        detail: trace.interoperability?.bundle_id ? "HAPI FHIR 事务已完成" : "FHIR 交换状态需检查",
      },
    ];
  }

  const selectedCategories = [...new Set(asList(education.recommendations).map((item) => labels.category[item.category] || readableCode(item.category)))];
  const depth = education.recommendations?.[0]?.content_depth === "standard" ? "标准深度" : "基础深度";
  return [
    {
      label: "候选检索",
      value: `${displayValue(education.candidate_count, 0)} 个医学主题`,
      detail: education.candidate_source === "neo4j" ? "候选来自 Neo4j 医学知识图谱" : "候选来自本地医学主题库",
    },
    {
      label: "主题遴选",
      value: `${asList(education.recommendations).length} 张内容卡片`,
      detail: selectedCategories.join("、") || "教育类别由病例内容确定",
    },
    {
      label: "内容深度",
      value: depth,
      detail: "正文围绕已选主题与当前病例背景生成",
    },
    {
      label: "安全边界",
      value: `${asList(education.recommendations).filter((item) => hasValue(item.safety_note)).length} 项提示`,
      detail: "每张卡片均附就医与专业复核提示",
    },
  ];
}

const processTitles = {
  intake: "病情要素归档",
  diagnosis: "证据整合与鉴别",
  treatment: "治疗方案与用药安全",
  coding: "编码映射与分组核验",
  audit: "隐私与结果完整性复核",
  recommendations: "教育主题遴选与内容生成",
};

const processIndexes = {
  intake: "01",
  diagnosis: "02",
  treatment: "03",
  coding: "04",
  audit: "05",
  recommendations: "06",
};

function makeProcessOverview(stage, response) {
  const section = createElement("section", "data-block wide process-overview");
  const header = createElement("div", "process-overview-header");
  const identity = createElement("div", "process-identity");
  const title = createElement("div");
  append(
    title,
    createElement("span", "process-kicker", "节点工作摘要"),
    createElement("h4", "", processTitles[stage]),
  );
  append(identity, createElement("span", "process-index", processIndexes[stage]), title);

  const elapsed = stageElapsedSeconds(response, stage);
  const total = totalElapsedSeconds(response, 0);
  const share = elapsed === null || total <= 0 ? 0 : Math.min(100, Math.max(3, (elapsed / total) * 100));
  const duration = createElement("div", "process-duration");
  duration.style.setProperty("--duration-share", `${share}%`);
  duration.setAttribute("aria-label", elapsed === null ? "节点耗时无记录" : `节点耗时 ${elapsed.toFixed(1)} 秒`);
  append(
    duration,
    createElement("span", "duration-label", "节点耗时"),
    createElement("strong", "", elapsed === null ? "—" : elapsed.toFixed(1)),
    createElement("small", "", "秒"),
  );
  append(header, identity, duration);

  const facts = createElement("div", "process-facts");
  processFacts(stage, response).forEach((item) => {
    const fact = createElement("article", "process-fact");
    append(
      fact,
      createElement("span", "process-fact-label", item.label),
      createElement("strong", "", item.value),
      createElement("p", "", item.detail),
    );
    facts.append(fact);
  });
  append(section, header, facts);
  return section;
}

function makeConfidence(label, value) {
  const confidence = confidenceValue(value);
  if (confidence === null) {
    return makeDefinitionGrid([[label, "—"]]);
  }
  const row = createElement("div", "confidence-row");
  const track = createElement("div", "confidence-track");
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", label);
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  track.setAttribute("aria-valuenow", String(Math.round(confidence * 100)));
  const fill = createElement("span");
  fill.style.setProperty("--confidence", `${Math.round(confidence * 100)}%`);
  track.append(fill);
  append(
    row,
    createElement("strong", "", label),
    track,
    createElement("output", "", `${Math.round(confidence * 100)}%`),
  );
  return row;
}

function makeRecordRow(title, subtitle, detailContent) {
  const row = createElement("div", "record-row");
  const titleArea = createElement("div", "record-title");
  append(titleArea, createElement("strong", "", displayValue(title)));
  if (hasValue(subtitle)) titleArea.append(createElement("small", "", subtitle));
  const detail = createElement("div", "record-detail");
  if (detailContent instanceof Node) {
    detail.append(detailContent);
  } else if (Array.isArray(detailContent)) {
    detail.append(makeDefinitionGrid(detailContent));
  } else {
    detail.append(createElement("p", "", displayValue(detailContent)));
  }
  append(row, titleArea, detail);
  return row;
}

function appendLabeledCollection(parent, label, values, variant = "") {
  if (!asList(values).length) return false;
  const section = createElement("div", "collection-group");
  append(section, createElement("p", "collection-label", label), makeTags(values, variant));
  parent.append(section);
  return true;
}

function renderMissingStage(title, description, explanation, response, stage, kind) {
  const grid = makeStage(title, description, kind);
  if (response && stage) grid.append(makeProcessOverview(stage, response));
  const { block, body } = makeBlock("节点结果", { wide: true });
  body.append(createElement("p", "empty-value", explanation));
  grid.append(block);
}

function renderIntake(response) {
  const data = response.patient_info;
  if (!data) {
    renderMissingStage("信息提取", "将病例叙述整理为症状、病史、用药、查体和检查要素。", "病例要素整理中断，请核对本次输入后重新提交。", response, "intake", "病情要素归档");
    return;
  }

  const grid = makeStage("信息提取", "将病例叙述整理为症状、病史、用药、查体和检查要素。", "病情要素归档");
  grid.append(makeProcessOverview("intake", response));
  const overview = makeBlock("基本信息", { wide: true, badge: "合成病例" });
  overview.body.append(makeDefinitionGrid([
    ["年龄", hasValue(data.age) ? `${data.age} 岁` : "—"],
    ["性别", translated(data.gender, labels.gender, "未知")],
    ["主要诉求", data.chief_complaint],
    ["数据标识", "合成或去标识化"],
  ]));
  grid.append(overview.block);

  const symptoms = makeBlock("症状", { badge: `${asList(data.symptoms).length} 项` });
  const symptomList = createElement("div", "record-list");
  asList(data.symptoms).forEach((symptom) => {
    const duration = hasValue(symptom.duration_days) ? `${symptom.duration_days} 天` : "时长 —";
    const severity = translated(symptom.severity, labels.severity, "程度 —");
    symptomList.append(makeRecordRow(
      symptom.name,
      `${severity} · ${duration}`,
      symptom.description || "症状描述已结构化",
    ));
  });
  symptoms.body.append(symptomList.childElementCount ? symptomList : createElement("p", "empty-value", "无症状条目"));
  grid.append(symptoms.block);

  const context = makeBlock("病史与安全背景");
  appendLabeledCollection(context.body, "既往史", data.medical_history);
  appendLabeledCollection(context.body, "手术史", data.surgical_history);
  appendLabeledCollection(context.body, "家族史", data.family_history);
  appendLabeledCollection(context.body, "社会史", data.social_history);
  const allergyValues = asList(data.allergies).map((item) => {
    const reaction = item.reaction ? `（${item.reaction}）` : "";
    return `${item.substance}${reaction}`;
  });
  appendLabeledCollection(context.body, "过敏信息", allergyValues, allergyValues.length ? "tag-warning" : "");
  grid.append(context.block);

  const medication = makeBlock("当前用药", { badge: `${asList(data.current_medications).length} 项` });
  const medicationRows = createElement("div", "record-list");
  asList(data.current_medications).forEach((item) => {
    medicationRows.append(makeRecordRow(item.name, item.start_date || "起始时间 —", [
      ["剂量", item.dosage],
      ["频次", item.frequency],
    ]));
  });
  medication.body.append(medicationRows.childElementCount ? medicationRows : createElement("p", "empty-value", "当前用药：无"));
  grid.append(medication.block);

  const vitals = makeBlock("生命体征");
  const vital = data.vital_signs;
  vitals.body.append(vital ? makeDefinitionGrid([
    ["体温", hasValue(vital.temperature) ? `${vital.temperature} °C` : "—"],
    ["心率", hasValue(vital.heart_rate) ? `${vital.heart_rate} 次/分` : "—"],
    ["血压", hasValue(vital.blood_pressure_systolic) || hasValue(vital.blood_pressure_diastolic) ? `${displayValue(vital.blood_pressure_systolic, "—")}/${displayValue(vital.blood_pressure_diastolic, "—")} mmHg` : "—"],
    ["呼吸频率", hasValue(vital.respiratory_rate) ? `${vital.respiratory_rate} 次/分` : "—"],
    ["血氧", hasValue(vital.oxygen_saturation) ? `${vital.oxygen_saturation}%` : "—"],
  ]) : createElement("p", "empty-value", "生命体征：—"));
  grid.append(vitals.block);

  if (asList(data.physical_exam).length) {
    const exam = makeBlock("查体结果");
    appendLabeledCollection(exam.body, "阳性与关键发现", data.physical_exam, "tag-accent");
    grid.append(exam.block);
  }

  const labs = makeBlock("检查 / 化验", { badge: `${asList(data.lab_results).length} 项` });
  const labRows = createElement("div", "record-list");
  asList(data.lab_results).forEach((item) => {
    const value = `${displayValue(item.value)}${item.unit ? ` ${item.unit}` : ""}`;
    labRows.append(makeRecordRow(item.test_name, item.is_abnormal ? "异常" : "参考范围内", [
      ["结果", value],
      ["参考范围", item.reference_range],
    ]));
  });
  labs.body.append(labRows.childElementCount ? labRows : createElement("p", "empty-value", "检查与化验：—"));
  grid.append(labs.block);

  if (asList(data.diagnostic_studies).length) {
    const studies = makeBlock("诊断检查", { wide: true, badge: `${asList(data.diagnostic_studies).length} 项` });
    const studyRows = createElement("div", "record-list");
    asList(data.diagnostic_studies).forEach((item) => {
      studyRows.append(makeRecordRow(
        item.study_name,
        item.is_abnormal ? "异常" : "参考表现",
        item.result,
      ));
    });
    studies.body.append(studyRows);
    grid.append(studies.block);
  }
}

function renderDiagnosis(response) {
  const data = response.diagnosis;
  const grid = makeStage("诊断分析", "综合临床证据、鉴别方向、知识图谱关联和建议检查。", "证据整合与鉴别");
  grid.append(makeProcessOverview("diagnosis", response));
  if (!data || !data.primary_diagnosis) {
    const missing = makeBlock("诊断信息需补充", { wide: true, badge: "需补充信息" });
    missing.body.append(createElement("p", "empty-value", "诊断条件不足，方案建议与医学编码阶段已暂停。"));
    missing.body.append(makePlainList(asList(response.information_gaps).map(readableCode)));
    grid.append(missing.block);
    return;
  }

  const primary = data.primary_diagnosis;
  const primaryBlock = makeBlock("主要诊断候选", { wide: true, badge: primary.icd10_hint || "编码提示 —" });
  const primaryTitle = createElement("p", "primary-diagnosis", primary.disease_name);
  append(primaryBlock.body, primaryTitle, makeConfidence("诊断置信度", primary.confidence));
  appendLabeledCollection(primaryBlock.body, "支持证据", primary.evidence, "tag-accent");
  if (hasValue(primary.reasoning)) {
    const reasoning = createElement("p", "narrative-note", primary.reasoning);
    primaryBlock.body.append(reasoning);
  }
  grid.append(primaryBlock.block);

  const differentials = makeBlock("鉴别诊断", { badge: `${asList(data.differential_list).length} 项` });
  const differentialRows = createElement("div", "record-list");
  asList(data.differential_list).forEach((item) => {
    const confidence = confidenceValue(item.confidence);
    differentialRows.append(makeRecordRow(
      item.disease_name,
      `${item.icd10_hint || "编码提示 —"}${confidence === null ? "" : ` · ${Math.round(confidence * 100)}%`}`,
      item.reasoning || makeTags(item.evidence),
    ));
  });
  differentials.body.append(differentialRows.childElementCount ? differentialRows : createElement("p", "empty-value", "鉴别诊断：无"));
  grid.append(differentials.block);

  const tests = makeBlock("建议检查", { badge: `${asList(data.recommended_tests).length} 项` });
  tests.body.append(makeTags(data.recommended_tests, "tag-accent"));
  grid.append(tests.block);

  const graphData = data.knowledge_graph || {};
  const graphEvidence = asList(graphData.evidence);
  if (graphEvidence.length) {
    const graph = makeBlock("知识图谱证据", { wide: true, badge: `Neo4j · ${graphEvidence.length} 项` });
    const graphRows = createElement("div", "record-list");
    graphEvidence.forEach((item) => {
      const score = hasValue(item.graph_score) ? `图谱评分 ${item.graph_score}` : "图谱关联";
      const detail = createElement("div");
      detail.append(makeDefinitionGrid([
        ["ICD-10", item.icd10_code],
        ["匹配症状", asList(item.matched_symptoms).join("、")],
        ["关联照护概念", asList(item.care_concepts).join("、")],
      ]));
      appendLabeledCollection(detail, "图谱路径", item.evidence_paths, "tag-accent");
      graphRows.append(makeRecordRow(item.disease, score, detail));
    });
    graph.body.append(graphRows);
    grid.append(graph.block);
  }

  const notes = makeBlock("临床摘要与知识来源", { wide: true });
  if (hasValue(data.clinical_notes)) notes.body.append(createElement("p", "narrative-note", data.clinical_notes));
  appendLabeledCollection(notes.body, "知识来源标签", data.knowledge_sources);
  grid.append(notes.block);
}

function renderTreatment(response) {
  const data = response.treatment_plan;
  if (!data) {
    renderMissingStage("方案建议", "整理药物、非药物措施、用药安全、生活方式与随访安排。", response.analysis_status === "needs_more_info" ? "诊断信息需补充，本节点已结束。" : "方案结果生成中断，请结合诊断结果复核。", response, "treatment", "治疗方案与用药安全");
    return;
  }

  const grid = makeStage("方案建议", "整理药物、非药物措施、用药安全、生活方式与随访安排。", "治疗方案与用药安全");
  grid.append(makeProcessOverview("treatment", response));
  const overview = makeBlock("方案概览", { wide: true, badge: "专业复核" });
  overview.body.append(makeDefinitionGrid([
    ["对应诊断", data.diagnosis_addressed],
    ["药物条目", `${asList(data.medications).length} 项`],
    ["相互作用", `${asList(data.drug_interactions).length} 项`],
    ["证据标签", `${asList(data.evidence_references).length} 项`],
  ]));
  grid.append(overview.block);

  const medications = makeBlock("药物条目", { wide: true, badge: `${asList(data.medications).length} 项` });
  const medicationRows = createElement("div", "record-list");
  asList(data.medications).forEach((item) => {
    const details = createElement("div");
    details.append(makeDefinitionGrid([
      ["通用名", item.generic_name],
      ["剂量 / 途径", `${displayValue(item.dosage)} · ${displayValue(item.route)}`],
      ["频次", item.frequency],
      ["疗程", item.duration],
    ]));
    appendLabeledCollection(details, "禁忌", item.contraindications, asList(item.contraindications).length ? "tag-warning" : "");
    appendLabeledCollection(details, "可能不良反应", item.side_effects);
    medicationRows.append(makeRecordRow(item.drug_name, "方案条目", details));
  });
  medications.body.append(medicationRows.childElementCount ? medicationRows : createElement("p", "empty-value", "药物条目：无"));
  grid.append(medications.block);

  const interactions = makeBlock("药物相互作用", { badge: `${asList(data.drug_interactions).length} 项` });
  const interactionRows = createElement("div", "record-list");
  asList(data.drug_interactions).forEach((item) => {
    const severity = translated(item.severity, labels.severity, "分级 —");
    interactionRows.append(makeRecordRow(`${displayValue(item.drug_a)} + ${displayValue(item.drug_b)}`, severity, [
      ["相互作用", item.description],
      ["建议", item.recommendation],
    ]));
  });
  interactions.body.append(interactionRows.childElementCount ? interactionRows : createElement("p", "empty-value", "药物相互作用：无"));
  grid.append(interactions.block);

  const supportive = makeBlock("非药物与生活方式");
  appendLabeledCollection(supportive.body, "非药物建议", data.non_drug_treatments, "tag-accent");
  appendLabeledCollection(supportive.body, "生活方式", data.lifestyle_recommendations);
  grid.append(supportive.block);

  const followUp = makeBlock("随访与警示", { wide: true });
  followUp.body.append(makeDefinitionGrid([["随访计划", data.follow_up_plan]]));
  appendLabeledCollection(followUp.body, "警示", data.warnings, asList(data.warnings).length ? "tag-warning" : "");
  appendLabeledCollection(followUp.body, "证据引用标签", data.evidence_references);
  grid.append(followUp.block);
}

function renderCoding(response) {
  const data = response.coding_result;
  if (!data) {
    renderMissingStage("医学编码", "形成 ICD-10 编码候选，完成本地目录与 DRG 分组核验。", response.analysis_status === "needs_more_info" ? "诊断信息需补充，本节点已结束。" : "编码结果生成中断，请进行专业复核。", response, "coding", "编码映射与分组核验");
    return;
  }

  const grid = makeStage("医学编码", "形成 ICD-10 编码候选，完成本地目录与 DRG 分组核验。", "编码映射与分组核验");
  grid.append(makeProcessOverview("coding", response));
  const primary = data.primary_icd10 || {};
  const mainCode = makeBlock("主要 ICD-10 候选", { wide: true, badge: codingCategory(primary.category) });
  const codeLine = createElement("div", "code-line");
  append(
    codeLine,
    createElement("strong", "code-value", displayValue(primary.code)),
    createElement("span", "", displayValue(primary.description)),
  );
  append(mainCode.body, codeLine, makeConfidence("编码置信度", primary.confidence));
  grid.append(mainCode.block);

  const secondary = makeBlock("次要编码", { badge: `${asList(data.secondary_icd10_codes).length} 项` });
  const secondaryRows = createElement("div", "record-list");
  asList(data.secondary_icd10_codes).forEach((item) => {
    const conf = confidenceValue(item.confidence);
    secondaryRows.append(makeRecordRow(item.code, conf === null ? item.category : `${Math.round(conf * 100)}% · ${item.category || "分类 —"}`, item.description));
  });
  secondary.body.append(secondaryRows.childElementCount ? secondaryRows : createElement("p", "empty-value", "次要编码：无"));
  grid.append(secondary.block);

  const drg = makeBlock("DRG 分组");
  drg.body.append(data.drg_group ? makeDefinitionGrid([
    ["DRG 编码", data.drg_group.drg_code],
    ["分组描述", data.drg_group.description],
    ["权重", data.drg_group.weight],
    ["平均住院日", hasValue(data.drg_group.mean_los) ? `${data.drg_group.mean_los} 天` : "—"],
  ]) : createElement("p", "empty-value", "DRG 分组：无匹配"));
  grid.append(drg.block);

  const notes = makeBlock("编码依据", { wide: true });
  append(notes.body, makeConfidence("总体编码置信度", data.coding_confidence), createElement("p", "narrative-note", displayValue(data.coding_notes)));
  if (data.local_validation) {
    notes.body.append(makeDefinitionGrid([
      ["本地编码目录", data.local_validation.catalog],
      ["主要编码匹配", data.local_validation.primary_code_matched ? "通过" : "需专业复核"],
      ["次要编码匹配", `${displayValue(data.local_validation.secondary_codes_matched, 0)} / ${displayValue(data.local_validation.secondary_code_count, 0)}`],
      ["DRG 前缀匹配", data.local_validation.drg_prefix_matched ? "通过" : "需专业复核"],
    ]));
  }
  grid.append(notes.block);
}

function renderAudit(response) {
  const data = response.audit_result;
  if (!data) {
    renderMissingStage("质量复核", "核对身份标识、结果完整性、临床记录与标准化数据交换。", "质量复核结果生成中断，请检查本次结果记录。", response, "audit", "隐私与结果完整性复核");
    return;
  }

  const grid = makeStage("质量复核", "核对身份标识、结果完整性、临床记录与标准化数据交换。", "隐私与结果完整性复核");
  grid.append(makeProcessOverview("audit", response));
  const overview = makeBlock("审计结论", { wide: true, badge: data.demo_safe ? "扫描通过" : "需要检查" });
  overview.body.append(makeDefinitionGrid([
    ["工作区", "合成或去标识化病例"],
    ["身份标识扫描", data.demo_safe ? "通过" : "需要人工复核"],
    ["结构化结果区域", `${asList(data.audit_trail).length ? auditTrailDetail(data.audit_trail[0]) : "—"}`],
    ["总体风险标签", readableCode(data.overall_risk_level)],
  ]));
  grid.append(overview.block);

  const checks = makeBlock("检查项", { badge: `${asList(data.compliance_checks).length} 项` });
  const checkList = createElement("ul", "check-list");
  asList(data.compliance_checks).forEach((item) => {
    const row = createElement("li", `check-item${item.passed ? " is-passed" : ""}`);
    const symbol = createElement("span", "check-symbol", item.passed ? "✓" : "!");
    const detail = createElement("div");
    append(
      detail,
      createElement("strong", "", labels.checks[item.check_name] || readableCode(item.check_name)),
      createElement("small", "", auditCheckDetail(item)),
    );
    append(row, symbol, detail);
    checkList.append(row);
  });
  checks.body.append(checkList.childElementCount ? checkList : createElement("p", "empty-value", "检查项：无"));
  grid.append(checks.block);

  if (asList(data.phi_fields_found).length) {
    const phi = makeBlock("标识符扫描", { badge: "人工复核" });
    appendLabeledCollection(phi.body, "命中类型", data.phi_fields_found, "tag-warning");
    grid.append(phi.block);
  }

  const trail = makeBlock("审计轨迹", { wide: true, badge: "本次会话" });
  const trailRows = createElement("div", "record-list");
  asList(data.audit_trail).forEach((item) => {
    trailRows.append(makeRecordRow(readableCode(item.action), item.timestamp, [
      ["资源类型", readableCode(item.resource_type)],
      ["结果", readableCode(item.outcome)],
      ["处理记录", auditTrailDetail(item)],
    ]));
  });
  trail.body.append(trailRows.childElementCount ? trailRows : createElement("p", "empty-value", "审计轨迹：无"));
  appendLabeledCollection(trail.body, "复核建议", asList(data.recommendations).map(readableCode), "tag-warning");
  grid.append(trail.block);

  const trace = response.integration_trace || {};
  if (Object.keys(trace).length) {
    const integrations = makeBlock("数据留存与交换", { wide: true, badge: "已写入" });
    const rows = createElement("div", "record-list");
    const privacy = trace.privacy_scan || {};
    rows.append(makeRecordRow(privacy.provider, "输入保护", [
      ["扫描结果", privacy.result === "clear" ? "通过" : displayValue(privacy.result)],
      ["命中类别", asList(privacy.detected_categories).join("、") || "无"],
    ]));
    const graph = trace.knowledge_graph || {};
    rows.append(makeRecordRow(graph.provider, "医学知识图谱", [
      ["图规模", `${displayValue(graph.nodes, 0)} 个节点 · ${displayValue(graph.relationships, 0)} 条关系`],
      ["本次诊断证据", `${displayValue(graph.evidence_count, 0)} 项`],
      ["诊断查询缓存", graph.query_cache === "hit" ? "命中" : "已写入"],
      ["教育候选来源", graph.recommendation_candidates === "neo4j" ? "Neo4j" : "本地主题目录"],
    ]));
    const cache = trace.cache_and_rate_limit || {};
    rows.append(makeRecordRow(cache.provider, `版本 ${displayValue(cache.version)}`, [
      ["当前键数量", cache.database_keys],
      ["本分钟剩余额度", cache.rate_limit_remaining],
      ["教育正文缓存", cache.recommendation_cache === "hit" ? "命中" : (cache.recommendation_cache === "fallback" ? "基础内容" : "已写入")],
    ]));
    const ranker = trace.recommendation_ranker || {};
    const inferenceMs = Number(ranker.inference_ms);
    rows.append(makeRecordRow("本地 Mini-OneRec", translated(ranker.used_strategy, labels.rankingStrategy, "固定安全顺序"), [
      ["模型版本", ranker.model_version],
      ["模型状态", ranker.model_ready ? "就绪" : "未就绪"],
      ["回退原因", translated(ranker.fallback_reason, labels.fallbackReason, "无")],
      ["排序耗时", Number.isFinite(inferenceMs) ? `${inferenceMs.toFixed(1)} 毫秒` : "—"],
      ["候选主题", ranker.candidate_count],
      ["有效历史", ranker.valid_history_count],
    ]));
    const persistence = trace.persistence || {};
    const persistenceSeconds = supportingElapsedSeconds(response, "persistence");
    rows.append(makeRecordRow(persistence.provider, persistenceSeconds === null ? "会话与审计存储" : `会话与审计存储 · ${persistenceSeconds.toFixed(1)} 秒`, [
      ["会话编号", persistence.session_id],
      ["临床会话总数", persistence.clinical_sessions],
      ["审计记录总数", persistence.audit_records],
    ]));
    const fhir = trace.interoperability || {};
    const interoperabilitySeconds = supportingElapsedSeconds(response, "interoperability");
    rows.append(makeRecordRow(fhir.provider, interoperabilitySeconds === null ? fhir.standard : `${displayValue(fhir.standard)} · ${interoperabilitySeconds.toFixed(1)} 秒`, [
      ["Bundle 编号", fhir.bundle_id],
      ["资源类型", asList(fhir.resource_types).join("、")],
      ["资源数量", fhir.resource_count],
    ]));
    integrations.body.append(rows);
    grid.append(integrations.block);
  }
}

function renderRecommendations(response) {
  const data = response.education_recommendations || {};
  const grid = makeStage("教育内容", "依据病例相关性、医学主题候选与阅读偏好生成教育内容。", "教育主题遴选与内容生成");
  grid.append(makeProcessOverview("recommendations", response));
  const ranker = makeBlock("排序执行记录", { wide: true, badge: translated(data.ranking_strategy_used, labels.rankingStrategy, "固定安全顺序") });
  const inferenceMs = Number(data.ranker_inference_ms);
  ranker.body.append(makeDefinitionGrid([
    ["排序策略", translated(data.ranking_strategy_used, labels.rankingStrategy, "固定安全顺序")],
    ["模型版本", displayValue(data.model_version)],
    ["模型状态", data.model_ready ? "就绪" : "未就绪"],
    ["回退原因", translated(data.fallback_reason, labels.fallbackReason, "无")],
    ["排序耗时", Number.isFinite(inferenceMs) ? `${inferenceMs.toFixed(1)} 毫秒` : "—"],
    ["正文策略", translated(data.content_strategy_used, labels.contentStrategy, "主题目录正文")],
    ["候选数量", displayValue(data.candidate_count, 0)],
    ["有效历史", displayValue(data.valid_history_count, 0)],
  ]));
  grid.append(ranker.block);
  const cards = makeBlock("教育内容卡片", { wide: true, badge: `${asList(data.recommendations).length} 张` });
  const cardList = createElement("div", "recommendation-list");
  asList(data.recommendations).forEach((item, index) => {
    const card = createElement("article", "recommendation-card");
    const rank = createElement("span", "recommendation-rank", `#${item.rank || index + 1}`);
    const content = createElement("div");
    const titleRow = createElement("div", "block-heading");
    append(
      titleRow,
      createElement("h4", "", item.title),
      makeBadge(labels.category[item.category] || readableCode(item.category)),
    );
    append(
      content,
      titleRow,
      createElement("p", "reason", item.reason),
      createElement("p", "", item.summary),
      createElement("p", "source-note", displayValue(item.source_label, "内容来源：医学主题服务")),
      createElement("p", "safety-note", item.safety_note),
    );
    append(card, rank, content);
    cardList.append(card);
  });
  cards.body.append(cardList.childElementCount ? cardList : createElement("p", "empty-value", data.recommendation_status === "disabled" ? "教育内容：关闭" : "教育内容生成中断"));
  grid.append(cards.block);
}

const stageRenderers = {
  intake: renderIntake,
  diagnosis: renderDiagnosis,
  treatment: renderTreatment,
  coding: renderCoding,
  audit: renderAudit,
  recommendations: renderRecommendations,
};

function activateStage(stage, focus = false) {
  if (!currentResponse || !stageRenderers[stage]) return;
  activeStage = stage;
  let selectedButton = null;
  document.querySelectorAll(".pipeline-step").forEach((button) => {
    const selected = button.dataset.stage === stage;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected) selectedButton = button;
    if (selected && focus) button.focus();
  });
  if (selectedButton) elements.stagePanel.setAttribute("aria-labelledby", selectedButton.id);
  stageRenderers[stage](currentResponse);
}

function stageState(response, stage) {
  if (stage === "intake") return response.patient_info ? ["done", "完成"] : ["error", "无结果"];
  if (stage === "diagnosis") {
    if (response.diagnosis?.primary_diagnosis) return ["done", "完成"];
    if (response.analysis_status === "needs_more_info") return ["attention", "需补充"];
    return ["error", "无结果"];
  }
  if (stage === "treatment") {
    if (response.treatment_plan) return ["done", "完成"];
    if (response.analysis_status === "needs_more_info") return ["skipped", "已跳过"];
    return ["error", "无结果"];
  }
  if (stage === "coding") {
    if (response.coding_result) return ["done", "完成"];
    if (response.analysis_status === "needs_more_info") return ["skipped", "已跳过"];
    return ["error", "无结果"];
  }
  if (stage === "audit") return response.audit_result ? ["done", "完成"] : ["error", "无结果"];
  const recommendationStatus = response.education_recommendations?.recommendation_status;
  if (recommendationStatus === "ok") return ["done", "完成"];
  if (recommendationStatus === "disabled") return ["skipped", "关闭"];
  if (recommendationStatus === "degraded") return ["attention", "已降级"];
  return ["error", "无结果"];
}

function updatePipeline(response) {
  document.querySelectorAll(".pipeline-step").forEach((button) => {
    const [state, text] = stageState(response, button.dataset.stage);
    const elapsed = stageElapsedSeconds(response, button.dataset.stage);
    button.dataset.state = state;
    button.querySelector("b").textContent = elapsed === null ? text : `${text} · ${elapsed.toFixed(1)} 秒`;
  });
}

function diagnosisName(response) {
  return response.diagnosis?.primary_diagnosis?.disease_name || "诊断候选 —";
}

function renderSummary(response, elapsedSeconds) {
  elements.resultSummary.replaceChildren();
  const lead = createElement("div", "summary-lead");
  append(
    lead,
    createElement("small", "", response.analysis_status === "needs_more_info" ? "当前结论" : "主要诊断候选"),
    createElement("strong", "", response.analysis_status === "needs_more_info" ? "需要补充病例信息" : diagnosisName(response)),
  );

  const status = createElement("div", "summary-metric");
  append(status, createElement("small", "", "分析状态"), createElement("strong", "", translated(response.analysis_status, labels.analysisStatus, "未知")));
  const serverElapsed = totalElapsedSeconds(response, elapsedSeconds);
  const backend = createElement("div", "summary-metric");
  append(backend, createElement("small", "", "全链处理时长"), createElement("strong", "", `${serverElapsed.toFixed(1)} 秒`));
  const recommendations = createElement("div", "summary-metric");
  append(recommendations, createElement("small", "", "教育推荐"), createElement("strong", "", `${asList(response.education_recommendations?.recommendations).length} 张卡片`));
  append(elements.resultSummary, lead, status, backend, recommendations);
  elements.resultTime.textContent = `本次耗时 ${serverElapsed.toFixed(1)} 秒`;
  elements.resultTime.hidden = false;
}

function renderIssues(response) {
  const gaps = asList(response.information_gaps).map(readableCode);
  const warnings = asList(response.warnings).map(readableCode);
  const errors = asList(response.errors).map(readableCode);
  const allIssues = [
    ...gaps.map((value) => `需补充：${value}`),
    ...warnings.map((value) => `提示：${value}`),
    ...errors.map((value) => `处理错误：${value}`),
  ];
  elements.issuePanel.replaceChildren();
  if (!allIssues.length) {
    elements.issuePanel.hidden = true;
    return;
  }
  const heading = createElement("div", "issue-heading");
  append(heading, createElement("h3", "", "需要关注的信息"), makeBadge(`${allIssues.length} 项`));
  append(elements.issuePanel, heading, makePlainList(allIssues));
  elements.issuePanel.hidden = false;
}

function responseMessage(response) {
  const recommendationCount = asList(response.education_recommendations?.recommendations).length;
  if (response.analysis_status === "needs_more_info") {
    const gaps = asList(response.information_gaps).slice(0, 3).map(readableCode);
    return `病例信息需补充，方案建议与医学编码已停止。${gaps.length ? `需补充项：${gaps.join("；")}。` : "请补充症状与背景。"}`;
  }
  if (response.analysis_status === "partial") {
    return `分析已返回部分结果。主要诊断候选为“${diagnosisName(response)}”。请检查右侧阶段状态与系统提示。`;
  }
  const code = response.coding_result?.primary_icd10?.code;
  return `分析已完成。主要诊断候选为“${diagnosisName(response)}”${code ? `，ICD-10 候选编码为 ${code}` : ""}；已匹配 ${recommendationCount} 张教育内容卡片。请在右侧查看各阶段结果。`;
}

function addMessage(role, text, meta = "", pending = false) {
  const article = createElement("article", `message message-${role}${pending ? " message-pending" : ""}`);
  const avatar = createElement("span", "avatar", role === "assistant" ? "M" : "你");
  avatar.setAttribute("aria-hidden", "true");
  const content = createElement("div", "message-content");
  if (pending) {
    append(content, createElement("i"), createElement("i"), createElement("i"));
    content.setAttribute("aria-label", "正在生成回复");
  } else {
    content.append(createElement("p", "", text));
    if (meta) content.append(createElement("span", "message-meta", meta));
  }
  append(article, avatar, content);
  elements.conversation.append(article);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  return article;
}

function selectedValues(containerId) {
  return [...document.querySelectorAll(`#${containerId} input:checked`)].map((input) => input.value);
}

function optionalValue(id) {
  const value = byId(id).value;
  return value === "" ? null : value;
}

function buildPayload(description) {
  const preferredCategories = selectedValues("preferred-categories");
  const excludedCategories = selectedValues("excluded-categories");
  const preferences = {
    preferred_categories: preferredCategories,
    excluded_categories: excludedCategories,
    preferred_depth: optionalValue("preferred-depth"),
  };
  return {
    patient_description: description,
    include_recommendations: byId("include-recommendations").checked,
    recommendation_top_k: Number(document.querySelector('input[name="top-k"]:checked').value),
    user_preferences: preferences,
  };
}

function setLoading(isLoading) {
  elements.analyzeButton.disabled = isLoading;
  elements.analyzeButton.classList.toggle("is-loading", isLoading);
  elements.buttonLabel.textContent = isLoading ? "正在分析" : "提交分析";
  if (isLoading) {
    elements.resultsEmpty.hidden = true;
    elements.resultsView.hidden = true;
    elements.loadingState.hidden = false;
    elements.resultTime.hidden = true;
    loadingStartedAt = performance.now();
    elements.elapsedTime.textContent = "0.0";
    loadingTimer = window.setInterval(() => {
      elements.elapsedTime.textContent = ((performance.now() - loadingStartedAt) / 1000).toFixed(1);
    }, 100);
  } else {
    elements.loadingState.hidden = true;
    if (loadingTimer !== null) window.clearInterval(loadingTimer);
    loadingTimer = null;
  }
}

function renderResponse(response, elapsedSeconds) {
  currentResponse = response;
  elements.pipelineNav.hidden = false;
  elements.resultSummary.hidden = false;
  elements.resultsView.hidden = false;
  renderSummary(response, elapsedSeconds);
  updatePipeline(response);
  renderIssues(response);
  activeStage = response.diagnosis ? "diagnosis" : (response.patient_info ? "intake" : "audit");
  activateStage(activeStage);
}

function errorExplanation(status, body) {
  const code = body?.detail?.code;
  if (status === 422 && code === "prototype_phi_not_allowed") {
    const detected = asList(body.detail.detected_types).join("、") || "身份字段";
    return `输入被本地保护规则拒绝：检测到 ${detected}。请删除真实或类似真实的身份信息后重试。`;
  }
  if (status === 422) return "输入字段未通过校验，请确认描述不少于 10 个字符，且偏好参数在允许范围内。";
  if (status === 503 || code === "llm_not_configured") return "模型服务配置未就绪，请联系系统管理员。";
  if (code === "infrastructure_unavailable") return `本地数据服务连接异常：${asList(body?.detail?.services).join("、")}。`;
  if (code === "integration_write_failed") return "分析结果写入 PostgreSQL 或 HAPI FHIR 时发生错误。";
  if (status === 429 || code === "rate_limit_exceeded") return "请求频率已达到本分钟上限，请稍后重试。";
  if (status === 502 || code === "llm_upstream_unavailable") return "上游模型暂时不可用或请求超时，请稍后重试。";
  if (code === "llm_request_rejected") return "模型服务拒绝了本次请求，请检查模型服务配置。";
  if (code === "pipeline_internal_error") return "分析服务发生内部错误，请查看系统日志。";
  return `请求未完成（HTTP ${status || "网络错误"}）。请确认本地服务仍在运行。`;
}

function renderError(message) {
  currentResponse = null;
  elements.resultSummary.hidden = true;
  elements.pipelineNav.hidden = true;
  elements.issuePanel.hidden = true;
  elements.stagePanel.replaceChildren();
  const wrapper = createElement("div", "error-state");
  append(
    wrapper,
    createElement("span", "error-icon", "!"),
    createElement("h3", "", "本次分析未完成"),
    createElement("p", "", message),
  );
  elements.stagePanel.append(wrapper);
  elements.resultsView.hidden = false;
}

async function parseResponseBody(response) {
  try {
    return await response.json();
  } catch (_error) {
    return null;
  }
}

async function submitAnalysis(event) {
  event.preventDefault();
  const description = elements.description.value.trim();
  elements.validation.hidden = true;
  if (description.length < 10) {
    elements.validation.textContent = "请至少输入 10 个字符的合成病例描述。";
    elements.validation.hidden = false;
    elements.description.focus();
    return;
  }

  addMessage("user", description, "合成病例输入");
  pendingMessage = addMessage("assistant", "", "", true);
  setLoading(true);
  const startedAt = performance.now();

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 360000);
  try {
    const response = await fetch("/api/v1/clinical/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(buildPayload(description)),
      signal: controller.signal,
    });
    const body = await parseResponseBody(response);
    if (!response.ok || !body) {
      throw { requestError: true, status: response.status, body };
    }
    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    setLoading(false);
    pendingMessage?.remove();
    pendingMessage = null;
    addMessage("assistant", responseMessage(body), `${translated(body.analysis_status, labels.analysisStatus, "已返回")} · ${elapsedSeconds.toFixed(1)} 秒`);
    renderResponse(body, elapsedSeconds);
  } catch (error) {
    setLoading(false);
    pendingMessage?.remove();
    pendingMessage = null;
    const message = error?.name === "AbortError"
      ? "请求已超过 6 分钟，当前页面已结束本次请求。请稍后检查服务状态。"
      : errorExplanation(error?.status, error?.body);
    addMessage("assistant", message, "请求未完成");
    renderError(message);
  } finally {
    window.clearTimeout(timeout);
  }
}

function setupCategoryExclusion() {
  ["preferred-categories", "excluded-categories"].forEach((containerId) => {
    byId(containerId).addEventListener("change", (event) => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement) || !input.checked) return;
      const otherId = containerId === "preferred-categories" ? "excluded-categories" : "preferred-categories";
      const counterpart = [...document.querySelectorAll(`#${otherId} input`)].find((item) => item.value === input.value);
      if (counterpart) counterpart.checked = false;
    });
  });
}

elements.description.addEventListener("input", () => {
  elements.characterCount.textContent = String(elements.description.value.length);
  elements.validation.hidden = true;
});

document.querySelectorAll("[data-case]").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      await validationCasesPromise;
    } catch (_error) {
      elements.validation.textContent = "病例目录加载异常，请刷新页面后重试。";
      elements.validation.hidden = false;
      return;
    }
    elements.description.value = demoCases[button.dataset.case] || "";
    elements.characterCount.textContent = String(elements.description.value.length);
    elements.validation.hidden = true;
    elements.description.focus();
  });
});

document.querySelectorAll(".pipeline-step").forEach((button, index, buttons) => {
  button.addEventListener("click", () => activateStage(button.dataset.stage));
  button.addEventListener("keydown", (event) => {
    if (!currentResponse || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let nextIndex = index;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + buttons.length) % buttons.length;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = buttons.length - 1;
    activateStage(buttons[nextIndex].dataset.stage, true);
  });
});

elements.form.addEventListener("submit", submitAnalysis);
setupCategoryExclusion();
