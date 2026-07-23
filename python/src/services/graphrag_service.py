"""Neo4j-backed medical graph retrieval used by diagnosis and education."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
from neo4j import GraphDatabase, RoutingControl

from ..config.settings import get_settings
from .recommendation.topic_store import TopicStore
from .redis_service import RedisService, get_redis_service

logger = structlog.get_logger(__name__)


DISEASES = [
    {"name": "Pneumonia", "aliases": ["pneumonia", "肺炎", "community acquired pneumonia"], "code": "J18.1", "description": "Lobar pneumonia, unspecified organism", "care": ["呼吸状态评估", "抗感染方案评估"]},
    {"name": "Acute myocardial infarction", "aliases": ["acute mi", "myocardial infarction", "heart attack", "心肌梗死", "stemi"], "code": "I21.0", "description": "ST elevation myocardial infarction of anterior wall", "care": ["紧急再灌注评估", "抗栓与出血风险评估"]},
    {"name": "Type 2 diabetes mellitus", "aliases": ["type 2 diabetes", "diabetes", "2 型糖尿病", "糖尿病"], "code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "care": ["血糖监测", "并发症风险评估"]},
    {"name": "Heart failure", "aliases": ["heart failure", "心力衰竭", "心衰"], "code": "I50.9", "description": "Heart failure, unspecified", "care": ["容量状态评估", "心功能随访"]},
    {"name": "Hypothyroidism", "aliases": ["hypothyroidism", "甲状腺功能减退", "甲减"], "code": "E03.9", "description": "Hypothyroidism, unspecified", "care": ["甲状腺功能复查"]},
    {"name": "Appendicitis", "aliases": ["appendicitis", "阑尾炎"], "code": "K35.80", "description": "Unspecified acute appendicitis", "care": ["腹部外科评估"]},
    {"name": "Influenza", "aliases": ["influenza", "flu", "流感"], "code": "J11.1", "description": "Influenza with respiratory manifestations", "care": ["呼吸道感染评估"]},
    {"name": "Asthma", "aliases": ["asthma", "哮喘"], "code": "J45.909", "description": "Unspecified asthma, uncomplicated", "care": ["气道状态评估"]},
    {"name": "Pulmonary embolism", "aliases": ["pulmonary embolism", "肺栓塞"], "code": "I26.99", "description": "Pulmonary embolism without acute cor pulmonale", "care": ["血栓风险与影像评估"]},
    {"name": "Gastroesophageal reflux disease", "aliases": ["gerd", "gastroesophageal reflux", "胃食管反流"], "code": "K21.0", "description": "Gastro-esophageal reflux disease", "care": ["消化系统评估"]},
]

SYMPTOM_LINKS = {
    "fever": (["发热", "高热"], [("Pneumonia", 0.82), ("Influenza", 0.72)]),
    "cough": (["咳嗽", "咳痰", "黄痰"], [("Pneumonia", 0.86), ("Influenza", 0.64), ("Asthma", 0.42)]),
    "shortness of breath": (["dyspnea", "dyspnea on exertion", "orthopnea", "呼吸困难", "气促", "气短"], [("Pneumonia", 0.70), ("Heart failure", 0.76), ("Pulmonary embolism", 0.74), ("Asthma", 0.66)]),
    "chest pain": (["胸痛", "胸骨后疼痛", "压榨性胸痛"], [("Acute myocardial infarction", 0.94), ("Pulmonary embolism", 0.65), ("Gastroesophageal reflux disease", 0.35)]),
    "diaphoresis": (["出汗", "大汗", "冷汗"], [("Acute myocardial infarction", 0.78)]),
    "nausea": (["恶心", "呕吐"], [("Acute myocardial infarction", 0.48), ("Appendicitis", 0.44)]),
    "fatigue": (["乏力", "疲劳"], [("Type 2 diabetes mellitus", 0.52), ("Heart failure", 0.46), ("Hypothyroidism", 0.60)]),
    "increased thirst": (["口渴", "多饮", "烦渴"], [("Type 2 diabetes mellitus", 0.90)]),
    "edema": (["水肿", "下肢水肿"], [("Heart failure", 0.84)]),
    "right lower abdominal pain": (["right lower quadrant abdominal pain", "rlq pain", "右下腹痛", "右下腹疼痛"], [("Appendicitis", 0.92)]),
}


def _cache_key(prefix: str, payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"medigen:{prefix}:{digest}"


def _topic_path() -> Path:
    settings = get_settings()
    path = Path(settings.recommendation_topic_path)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _load_topic_seed() -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    with _topic_path().open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            topic = json.loads(raw)
            topic["source_url"] = topic.get("source_url") or ""
            topics.append(topic)
    return topics


def _topic_catalog_hash_prefix() -> str:
    return TopicStore.from_jsonl(_topic_path()).catalog_sha256()[:12]


class GraphRAGService:
    """Execute parameterized Cypher against the local Neo4j medical graph."""

    def __init__(
        self,
        use_neo4j: bool = False,
        *,
        cache: RedisService | None = None,
    ) -> None:
        self.use_neo4j = use_neo4j
        self.cache = cache
        self._driver = None
        self._database = get_settings().neo4j_database

    def connect(self) -> None:
        if not self.use_neo4j or self._driver is not None:
            return
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=3,
        )
        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            raise
        self._driver = driver
        logger.info("graphrag.neo4j_connected", database=self._database)

    def initialize(self) -> None:
        if not self.use_neo4j:
            return
        self.connect()
        self._execute_write(
            "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE"
        )
        self._execute_write(
            "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE"
        )
        self._execute_write(
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:EducationTopic) REQUIRE n.topic_id IS UNIQUE"
        )
        self._execute_write(
            "CREATE CONSTRAINT test_name IF NOT EXISTS FOR (n:ClinicalTest) REQUIRE n.name IS UNIQUE"
        )
        self._execute_write(
            "CREATE CONSTRAINT medication_name IF NOT EXISTS FOR (n:Medication) REQUIRE n.name IS UNIQUE"
        )
        self._execute_write(
            """
            UNWIND $diseases AS row
            MERGE (d:Disease {name: row.name})
            SET d.aliases = row.aliases, d.icd10_code = row.code,
                d.description = row.description
            WITH d, row
            UNWIND row.care AS care_name
            MERGE (c:CareConcept {name: care_name})
            MERGE (d)-[:MANAGED_WITH]->(c)
            """,
            diseases=DISEASES,
        )
        symptom_rows: list[dict[str, Any]] = []
        for name, (aliases, links) in SYMPTOM_LINKS.items():
            for disease, weight in links:
                symptom_rows.append(
                    {
                        "name": name,
                        "aliases": aliases,
                        "disease": disease,
                        "weight": weight,
                    }
                )
        self._execute_write(
            """
            UNWIND $rows AS row
            MERGE (s:Symptom {name: row.name})
            SET s.aliases = row.aliases
            WITH s, row
            MATCH (d:Disease {name: row.disease})
            MERGE (s)-[r:INDICATES]->(d)
            SET r.weight = row.weight, r.source = 'curated_local_seed'
            """,
            rows=symptom_rows,
        )
        topics = _load_topic_seed()
        self._execute_write(
            """
            UNWIND $topics AS row
            MERGE (t:EducationTopic {topic_id: row.topic_id})
            SET t += row
            """,
            topics=topics,
        )
        self._execute_write(
            """
            MATCH (t:EducationTopic)
            UNWIND t.related_codes AS code_prefix
            MATCH (d:Disease)
            WHERE toUpper(d.icd10_code) STARTS WITH toUpper(code_prefix)
            MERGE (d)-[:HAS_EDUCATION]->(t)
            """
        )
        self._execute_write(
            """
            MATCH (t:EducationTopic)
            UNWIND t.related_tests AS test_name
            MERGE (test:ClinicalTest {name: toLower(test_name)})
            MERGE (test)-[:HAS_EDUCATION]->(t)
            """
        )
        self._execute_write(
            """
            MATCH (t:EducationTopic)
            UNWIND t.related_medications AS medication_name
            MERGE (med:Medication {name: toLower(medication_name)})
            MERGE (med)-[:HAS_EDUCATION]->(t)
            """
        )
        logger.info("graphrag.seed_complete", topic_count=len(topics))

    def is_ready(self) -> bool:
        if not self.use_neo4j:
            return False
        try:
            self.connect()
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.warning("graphrag.readiness_failed", error_type=type(exc).__name__)
            return False

    def _execute_write(self, query: str, **parameters: Any) -> None:
        if self._driver is None:
            raise RuntimeError("neo4j driver is not connected")
        self._driver.execute_query(
            query,
            parameters_=parameters,
            database_=self._database,
        )

    def _execute_read(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        if self._driver is None:
            raise RuntimeError("neo4j driver is not connected")
        records, _, _ = self._driver.execute_query(
            query,
            parameters_=parameters,
            database_=self._database,
            routing_=RoutingControl.READ,
        )
        return [record.data() for record in records]

    def find_diseases_with_trace(self, symptoms: list[str]) -> dict[str, Any]:
        clean = sorted({str(item).strip() for item in symptoms if str(item).strip()})
        key = _cache_key("graph:diseases", clean)
        if self.cache:
            cached = self.cache.get_json(key)
            if isinstance(cached, list):
                return {"records": cached, "cache_status": "hit"}

        if not self.use_neo4j:
            records = self._offline_disease_lookup(clean)
            return {"records": records, "cache_status": "offline"}

        self.connect()
        rows = self._execute_read(
            """
            MATCH (s:Symptom)-[r:INDICATES]->(d:Disease)
            WHERE any(input IN $symptoms WHERE
                toLower(input) CONTAINS toLower(s.name)
                OR toLower(s.name) CONTAINS toLower(input)
                OR any(alias IN s.aliases WHERE
                    toLower(input) CONTAINS toLower(alias)
                    OR toLower(alias) CONTAINS toLower(input)))
            OPTIONAL MATCH (d)-[:MANAGED_WITH]->(care:CareConcept)
            RETURN d.name AS disease,
                   d.icd10_code AS icd10_code,
                   d.description AS icd10_description,
                   collect(DISTINCT s.name) AS matched_symptoms,
                   round(sum(DISTINCT r.weight) * 100) / 100 AS graph_score,
                   collect(DISTINCT care.name) AS care_concepts
            ORDER BY graph_score DESC, disease ASC
            LIMIT 8
            """,
            symptoms=clean,
        )
        records = []
        for row in rows:
            row["symptom_match_count"] = len(row.get("matched_symptoms", []))
            row["evidence_paths"] = [
                f"{symptom} → {row['disease']}"
                for symptom in row.get("matched_symptoms", [])
            ] + [
                f"{row['disease']} → {care}"
                for care in row.get("care_concepts", [])
            ]
            records.append(row)
        if self.cache:
            self.cache.set_json(key, records)
        return {"records": records, "cache_status": "miss"}

    def find_diseases_by_symptoms(self, symptoms: list[str]) -> list[dict[str, Any]]:
        return self.find_diseases_with_trace(symptoms)["records"]

    def find_education_topics(self, context: dict[str, list[str]]) -> dict[str, Any]:
        normalized = {key: sorted({str(v).strip() for v in values if str(v).strip()}) for key, values in context.items()}
        namespace = f"graph:topics:v2:{_topic_catalog_hash_prefix()}"
        key = _cache_key(namespace, normalized)
        if self.cache:
            cached = self.cache.get_json(key)
            if isinstance(cached, list):
                return {"topics": cached, "cache_status": "hit"}
        if not self.use_neo4j:
            return {"topics": _load_topic_seed(), "cache_status": "offline"}

        self.connect()
        rows = self._execute_read(
            """
            MATCH (t:EducationTopic {status: 'active'})
            OPTIONAL MATCH (d:Disease)-[:HAS_EDUCATION]->(t)
            OPTIONAL MATCH (test:ClinicalTest)-[:HAS_EDUCATION]->(t)
            OPTIONAL MATCH (med:Medication)-[:HAS_EDUCATION]->(t)
            WITH t,
                 collect(DISTINCT d.icd10_code) AS disease_codes,
                 collect(DISTINCT d.name) + reduce(names = [], x IN collect(DISTINCT d.aliases) | names + coalesce(x, [])) AS disease_terms,
                 collect(DISTINCT test.name) AS test_terms,
                 collect(DISTINCT med.name) AS medication_terms
            WHERE t.general_fallback = true
               OR any(code IN $codes WHERE any(candidate IN disease_codes WHERE toUpper(code) STARTS WITH toUpper(split(candidate, '.')[0])))
               OR any(term IN $diagnoses WHERE any(candidate IN disease_terms WHERE toLower(term) CONTAINS toLower(candidate) OR toLower(candidate) CONTAINS toLower(term)))
               OR any(term IN $tests WHERE any(candidate IN test_terms WHERE toLower(term) CONTAINS candidate OR candidate CONTAINS toLower(term)))
               OR any(term IN $medications WHERE any(candidate IN medication_terms WHERE toLower(term) CONTAINS candidate OR candidate CONTAINS toLower(term)))
            RETURN properties(t) AS topic
            ORDER BY t.priority DESC, t.topic_id ASC
            """,
            codes=normalized.get("diagnosis_codes", []),
            diagnoses=normalized.get("diagnosis_terms", []),
            tests=normalized.get("recommended_tests", []),
            medications=normalized.get("medication_names", []),
        )
        topics = [row["topic"] for row in rows]
        for topic in topics:
            if not topic.get("source_url"):
                topic["source_url"] = None
        if self.cache:
            self.cache.set_json(key, topics)
        return {"topics": topics, "cache_status": "miss"}

    def get_icd10(self, disease_name: str) -> dict[str, str] | None:
        if not self.use_neo4j:
            for disease in DISEASES:
                if disease["name"].casefold() == disease_name.casefold():
                    return {"code": disease["code"], "desc": disease["description"]}
            return None
        self.connect()
        rows = self._execute_read(
            """
            MATCH (d:Disease)
            WHERE toLower(d.name) = toLower($name)
               OR any(alias IN d.aliases WHERE toLower(alias) = toLower($name))
            RETURN d.icd10_code AS code, d.description AS desc
            LIMIT 1
            """,
            name=disease_name,
        )
        return rows[0] if rows else None

    def stats(self) -> dict[str, Any]:
        self.connect()
        rows = self._execute_read(
            """
            MATCH (n)
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS relationships
            """
        )
        values = rows[0] if rows else {"nodes": 0, "relationships": 0}
        return {"provider": "Neo4j", **values}

    @staticmethod
    def _offline_disease_lookup(symptoms: list[str]) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}
        for raw in symptoms:
            normalized = raw.casefold().replace("_", " ")
            for name, (aliases, links) in SYMPTOM_LINKS.items():
                if not any(term.casefold() in normalized or normalized in term.casefold() for term in [name, *aliases]):
                    continue
                for disease_name, weight in links:
                    item = scores.setdefault(disease_name, {"score": 0.0, "matched": []})
                    item["score"] += weight
                    item["matched"].append(name)
        disease_by_name = {item["name"]: item for item in DISEASES}
        result = []
        for disease_name, values in sorted(scores.items(), key=lambda item: (-item[1]["score"], item[0])):
            disease = disease_by_name[disease_name]
            result.append(
                {
                    "disease": disease_name,
                    "symptom_match_count": len(set(values["matched"])),
                    "icd10_code": disease["code"],
                    "icd10_description": disease["description"],
                    "matched_symptoms": sorted(set(values["matched"])),
                    "graph_score": round(values["score"], 2),
                    "care_concepts": disease["care"],
                }
            )
        return result

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


@lru_cache(maxsize=1)
def get_graphrag_service() -> GraphRAGService:
    settings = get_settings()
    if settings.llm_backend == "fixture":
        return GraphRAGService(use_neo4j=False)
    return GraphRAGService(use_neo4j=True, cache=get_redis_service())
