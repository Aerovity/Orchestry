# Orchestry Backend API

FastAPI backend for the Orchestry MARL platform.

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Configure `.env` with your credentials:
- **Clerk**: Get keys from https://dashboard.clerk.com
- **Supabase**: Get keys from https://supabase.com/dashboard
- **Anthropic**: Get key from https://console.anthropic.com

4. Set up Supabase database:
- Go to Supabase SQL Editor
- Run the SQL in `supabase_schema.sql`

5. Run the server:
```bash
python run.py
```

API will be available at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

## API Endpoints

- `POST /api/v1/training/jobs` - Create training job
- `GET /api/v1/training/jobs` - List user's jobs
- `GET /api/v1/training/jobs/{id}` - Get job details
- `GET /api/v1/training/jobs/{id}/results` - Get results
- `POST /api/v1/training/jobs/{id}/cancel` - Cancel job
