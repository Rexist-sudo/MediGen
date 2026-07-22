-- Clinical Decision Support System — Database Initialization

CREATE TABLE IF NOT EXISTS audit_logs (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id         VARCHAR(128) NOT NULL DEFAULT 'system',
    action          VARCHAR(64) NOT NULL,
    resource_type   VARCHAR(64) NOT NULL,
    resource_id     VARCHAR(256) DEFAULT '',
    detail          TEXT DEFAULT '',
    outcome         VARCHAR(32) DEFAULT 'success',
    ip_address      VARCHAR(45) DEFAULT ''
);

CREATE INDEX idx_audit_logs_timestamp ON audit_logs (timestamp);
CREATE INDEX idx_audit_logs_user_id ON audit_logs (user_id);
CREATE INDEX idx_audit_logs_resource ON audit_logs (resource_type, resource_id);

CREATE TABLE IF NOT EXISTS clinical_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id       VARCHAR(128) NOT NULL,
    raw_input       TEXT,
    patient_info    JSONB,
    diagnosis       JSONB,
    treatment_plan  JSONB,
    coding_result   JSONB,
    audit_result    JSONB,
    recommendation_result JSONB,
    knowledge_context JSONB,
    fhir_export     JSONB,
    analysis_status VARCHAR(32),
    llm_backend     VARCHAR(32),
    errors          JSONB DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE clinical_sessions
    ADD COLUMN IF NOT EXISTS recommendation_result JSONB,
    ADD COLUMN IF NOT EXISTS knowledge_context JSONB,
    ADD COLUMN IF NOT EXISTS fhir_export JSONB,
    ADD COLUMN IF NOT EXISTS analysis_status VARCHAR(32),
    ADD COLUMN IF NOT EXISTS llm_backend VARCHAR(32);

CREATE INDEX idx_sessions_thread ON clinical_sessions (thread_id);
CREATE INDEX idx_sessions_created ON clinical_sessions (created_at);

CREATE OR REPLACE FUNCTION reject_audit_log_mutation()
RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_logs is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_logs_append_only ON audit_logs;
CREATE TRIGGER audit_logs_append_only
BEFORE UPDATE OR DELETE ON audit_logs
FOR EACH ROW EXECUTE FUNCTION reject_audit_log_mutation();

COMMENT ON TABLE audit_logs IS 'Append-only local audit trail for synthetic or de-identified cases.';
COMMENT ON TABLE clinical_sessions IS 'Clinical pipeline results for synthetic or de-identified cases.';
