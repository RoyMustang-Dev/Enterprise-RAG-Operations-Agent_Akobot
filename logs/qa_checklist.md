# QA Checklist (V2)

Generated: 2026-03-19 10:18:23

## Email Connect Link Flow (V2)
- Step: Send a query that includes an email address and a send request.
- Expected: `email_action.connect_url` is returned.
- Result: **PASS** (connect link present in `logs/_latest_v2_chat_test.json`).

## V2 Chat (RAG + File)
- Step: Upload a file and ask for a summary.
- Expected: Summary response + sources + persisted chat history.
- Result: **PASS** (response contains `sources` and `chat_history`).

## V2 Provider Selection
- Step: Run V2 chat with Modelslab configured.
- Expected: `primary_provider=modelslab` and `active_model=modelslab`.
- Result: **PASS**.

## Deterministic Email Trigger
- Step: Include email address in query.
- Expected: `get_email_or_send` tool is called automatically.
- Result: **PASS** (tools_used contains `get_email_or_send`).
