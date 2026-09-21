-- ATBG database schema (Postgres)
-- Run automatically by docker-compose on first container start.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Conversations ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     TEXT UNIQUE,                 -- id from the source dataset, if any
    source          TEXT NOT NULL,                -- e.g. 'scam-dialogue', 'user_upload'
    channel         TEXT NOT NULL DEFAULT 'unknown', -- email/sms/whatsapp/telegram/chat/voice_transcript/screenshot/unknown
    language        TEXT NOT NULL DEFAULT 'unknown', -- en/hi/ta/mixed/other/unknown
    is_attack       BOOLEAN,                      -- null = unknown/unlabelled
    scam_type       TEXT,                         -- phishing/bec/romance/investment/tech_support/job_task/sextortion/lottery_419/other/null
    is_synthetic    BOOLEAN NOT NULL DEFAULT FALSE,
    split           TEXT,                         -- train/validation/test/adversarial/temporal_holdout/null
    label_source    TEXT NOT NULL DEFAULT 'none',  -- none/weak/gold
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_source ON conversations(source);
CREATE INDEX IF NOT EXISTS idx_conversations_split ON conversations(split);

-- ── Turns ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turns (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id     UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_index          INT NOT NULL,             -- 1-based order within the conversation
    speaker             TEXT NOT NULL,             -- attacker/victim/unknown
    text_redacted       TEXT NOT NULL,
    "timestamp"         TIMESTAMPTZ,               -- NULL if unavailable — never invented
    behavior_state      TEXT,                      -- one of the 12 states, null if unlabelled
    behavior_confidence REAL,
    label_source        TEXT NOT NULL DEFAULT 'none', -- none/weak/gold
    escalation_flag     BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (conversation_id, turn_index)
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id);

-- ── Artifacts (extracted from turn text) ────────────────────────
CREATE TABLE IF NOT EXISTS artifacts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    turn_id         UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    artifact_type   TEXT NOT NULL,   -- url/domain/ip/email/phone/upi/crypto_wallet/file_hash
    value           TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_artifacts_turn ON artifacts(turn_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_value ON artifacts(value);

-- ── Threat-intelligence lookups (cached, per artifact) ──────────
CREATE TABLE IF NOT EXISTS threat_intel_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    artifact_id     UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    provider        TEXT NOT NULL,   -- virustotal/abuseipdb/urlhaus/safe_browsing/rdap
    malicious       BOOLEAN,
    raw_response    JSONB,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ti_artifact ON threat_intel_results(artifact_id);

-- ── Model predictions (one row per analysis run, at a given prefix) ──
CREATE TABLE IF NOT EXISTS predictions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id     UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    up_to_turn_index    INT NOT NULL,          -- prediction made using turns 1..N
    risk_score          REAL NOT NULL,
    current_state        TEXT,
    next_state_pred      TEXT,
    next_state_prob      REAL,
    eta_seconds          REAL,
    explanation_json      JSONB,
    model_version         TEXT NOT NULL DEFAULT 'untrained',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_conversation ON predictions(conversation_id);
