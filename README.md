# DataDoctor AI

AI-powered Data Quality & SQL/PySpark Debugging Copilot.

## Production setup

DataDoctor AI is a Streamlit app with Google OIDC authentication. Raw uploaded rows are profiled in memory and are not sent to OpenAI. AI requests use bounded, metadata-only context.

### Streamlit Secrets

Configure these in **Streamlit Cloud → Manage app → Settings → Secrets**. Never commit real credentials to GitHub.

```toml
[auth]
redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
cookie_secret = "YOUR_RANDOM_SECRET"
client_id = "YOUR_GOOGLE_CLIENT_ID"
client_secret = "YOUR_GOOGLE_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENAI_MODEL = "gpt-4.1-mini"

[app]
# Optional: restrict access to approved Google accounts.
# If omitted/empty, any successfully authenticated Google account can use the app.
allowed_emails = ["you@example.com"]

# Optional resource/cost controls.
max_ai_calls_per_hour = 10
ai_cooldown_seconds = 3
max_rows = 1000000
max_columns = 500
```

### Google OAuth

Create a Google OAuth 2.0 Web application and set the authorized redirect URI to:

`https://YOUR-APP.streamlit.app/oauth2callback`

For a private deployment, configure `allowed_emails` with the approved accounts. Do not place Google client secrets or the OpenAI API key in source control.

## Security notes

- Authentication is handled by Streamlit OIDC; the application does not implement passwords or its own OAuth token exchange.
- OpenAI credentials remain server-side in Streamlit Secrets.
- Uploaded files are processed in memory and are not written to disk by the profiler.
- `sample_values` are explicitly excluded from AI context.
- AI context is bounded by size and column/finding count.
- AI failures return a generic user-safe message rather than provider exception details.
- AI calls have a process-local per-user hourly limit and cooldown to reduce accidental OpenAI spend. This is a lightweight application guard, not a replacement for provider-level billing limits.

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local OIDC development, create `.streamlit/secrets.toml` using the same structure as above and use `http://localhost:8501/oauth2callback` as the local redirect URI. Keep that file out of source control.
