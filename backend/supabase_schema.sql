-- Supabase Schema for Orchestry
-- Run this in Supabase SQL Editor

-- Training Jobs Table
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    current_episode INTEGER DEFAULT 0,
    total_episodes INTEGER NOT NULL,
    average_reward FLOAT,
    error_message TEXT,
    anthropic_api_key TEXT -- Encrypted in production
);

-- Training Results Table
CREATE TABLE training_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES training_jobs(id) ON DELETE CASCADE,
    episodes JSONB NOT NULL,
    rewards JSONB NOT NULL,
    learned_behaviors JSONB,
    summary JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- User Usage Table (for billing)
CREATE TABLE user_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    month TEXT NOT NULL, -- Format: '2025-01'
    episodes_used INTEGER DEFAULT 0,
    api_calls INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, month)
);

-- Indexes
CREATE INDEX idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_results_job_id ON training_results(job_id);
CREATE INDEX idx_user_usage_user_month ON user_usage(user_id, month);

-- Row Level Security (RLS) - Users can only see their own data
ALTER TABLE training_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_usage ENABLE ROW LEVEL SECURITY;

-- Note: You'll need to configure RLS policies based on your auth setup
-- Example policy (adjust based on your Clerk integration):
-- CREATE POLICY "Users can view own jobs" ON training_jobs
--     FOR SELECT USING (user_id = current_user_id());
