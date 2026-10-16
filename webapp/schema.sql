CREATE TABLE IF NOT EXISTS project (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL CHECK (length(trim(name)) > 0),
    description TEXT    NOT NULL DEFAULT '',
    options     TEXT    NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS source_file (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES project (id) ON DELETE CASCADE,
    original_name TEXT    NOT NULL,
    stored_name   TEXT    NOT NULL UNIQUE,
    size_bytes    INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256        TEXT    NOT NULL,
    uploaded_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS source_file_project ON source_file (project_id);

CREATE TABLE IF NOT EXISTS source_table (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id      INTEGER NOT NULL REFERENCES source_file (id) ON DELETE CASCADE,
    position     INTEGER NOT NULL CHECK (position >= 0),
    name         TEXT    NOT NULL,
    format       TEXT    NOT NULL,
    columns      TEXT    NOT NULL,
    details      TEXT    NOT NULL DEFAULT '{}',
    key_column   TEXT,
    split_column TEXT,
    UNIQUE (file_id, position),
    CHECK (key_column IS NULL OR split_column IS NULL OR key_column <> split_column)
);

CREATE TABLE IF NOT EXISTS source (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id     INTEGER NOT NULL REFERENCES source_table (id) ON DELETE CASCADE,
    part         TEXT    NOT NULL DEFAULT '',
    name         TEXT    NOT NULL,
    record_count INTEGER NOT NULL CHECK (record_count >= 0),
    column_count INTEGER NOT NULL CHECK (column_count >= 0),
    completeness REAL    NOT NULL CHECK (completeness BETWEEN 0 AND 1),
    priority     INTEGER NOT NULL DEFAULT 0,
    included     INTEGER NOT NULL DEFAULT 1 CHECK (included IN (0, 1)),
    profile      TEXT    NOT NULL DEFAULT '[]',
    UNIQUE (table_id, part)
);

CREATE TABLE IF NOT EXISTS attribute (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES project (id) ON DELETE CASCADE,
    position   INTEGER NOT NULL CHECK (position >= 0),
    name       TEXT    NOT NULL CHECK (length(trim(name)) > 0),
    type       TEXT    NOT NULL,
    confidence REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    role       TEXT    NOT NULL DEFAULT 'attribute' CHECK (role IN ('attribute', 'ignored', 'reference')),
    resolution TEXT,
    UNIQUE (project_id, name)
);

CREATE TABLE IF NOT EXISTS attribute_column (
    attribute_id INTEGER NOT NULL REFERENCES attribute (id) ON DELETE CASCADE,
    source_id    INTEGER NOT NULL REFERENCES source (id) ON DELETE CASCADE,
    column_name  TEXT    NOT NULL,
    PRIMARY KEY (attribute_id, source_id),
    UNIQUE (source_id, column_name)
);

CREATE TABLE IF NOT EXISTS pair_review (
    project_id   INTEGER NOT NULL REFERENCES project (id) ON DELETE CASCADE,
    left_source  TEXT    NOT NULL,
    left_key     TEXT    NOT NULL,
    right_source TEXT    NOT NULL,
    right_key    TEXT    NOT NULL,
    is_match     INTEGER NOT NULL CHECK (is_match IN (0, 1)),
    reviewed_at  TEXT    NOT NULL,
    PRIMARY KEY (project_id, left_source, left_key, right_source, right_key)
);

CREATE TABLE IF NOT EXISTS run (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES project (id) ON DELETE CASCADE,
    status      TEXT    NOT NULL CHECK (status IN ('queued', 'running', 'finished', 'failed')),
    stage       TEXT    NOT NULL DEFAULT '',
    options     TEXT    NOT NULL,
    error       TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS run_project ON run (project_id);

CREATE TABLE IF NOT EXISTS report (
    run_id             INTEGER PRIMARY KEY REFERENCES run (id) ON DELETE CASCADE,
    source_count       INTEGER NOT NULL,
    record_count       INTEGER NOT NULL,
    attribute_count    INTEGER NOT NULL,
    entity_count       INTEGER NOT NULL,
    merged_records     INTEGER NOT NULL,
    candidate_pairs    INTEGER NOT NULL,
    matched_pairs      INTEGER NOT NULL,
    possible_pairs     INTEGER NOT NULL,
    reviewed_pairs     INTEGER NOT NULL,
    unified_cells      INTEGER NOT NULL,
    unrecognized_cells INTEGER NOT NULL,
    conflicting_cells  INTEGER NOT NULL,
    filled_cells       INTEGER NOT NULL,
    imputed_cells      INTEGER NOT NULL,
    index_before       REAL    NOT NULL,
    index_after        REAL    NOT NULL,
    seconds            REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS quality_score (
    run_id    INTEGER NOT NULL REFERENCES run (id) ON DELETE CASCADE,
    criterion TEXT    NOT NULL,
    weight    REAL    NOT NULL,
    before    REAL    NOT NULL,
    after     REAL    NOT NULL,
    PRIMARY KEY (run_id, criterion)
);

CREATE TABLE IF NOT EXISTS evaluation (
    run_id        INTEGER PRIMARY KEY REFERENCES run (id) ON DELETE CASCADE,
    reference     TEXT    NOT NULL,
    true_entities INTEGER NOT NULL,
    true_pairs    INTEGER NOT NULL,
    found_pairs   INTEGER NOT NULL,
    correct_pairs INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS report_file (
    run_id     INTEGER NOT NULL REFERENCES run (id) ON DELETE CASCADE,
    kind       TEXT    NOT NULL,
    file_name  TEXT    NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    PRIMARY KEY (run_id, kind)
);
