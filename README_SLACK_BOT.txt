Case 4 – Slack Policy Bot
=========================

This update contains only slack_bot.py and this instruction file. It does NOT
replace app.py, rag_utils.py, policy_vectors.json or your existing .env file.

1) In your Slack workspace, open #case4-bartal-testing, click the channel name,
   and use the Agents & apps / Integrations tab to add "Case 4 Policy Bot".

2) Copy slack_bot.py into the root of Case4-PolicyAssistant, next to app.py.

3) Make sure your existing local .env contains all three keys (NEVER share them):
       GEMINI_API_KEY=...
       SLACK_APP_TOKEN=xapp-...
       SLACK_BOT_TOKEN=xoxb-...

4) In VS Code, open a NEW terminal in the Case4-PolicyAssistant folder, then run:
       .\.venv\Scripts\python.exe -m pip install slack-bolt
       .\.venv\Scripts\python.exe slack_bot.py

5) In #case4-bartal-testing, send a message mentioning the BOT (select it from
   Slack's @ autocomplete), for example:
       @Case 4 Policy Bot How many days can I work abroad?

Expected: a concise answer stating the 30-day limit with approval, plus
'Relevant policy: Work From Abroad Policy'.

Keep the Slack bot terminal running while testing. The website is independent.

Later, after successful testing, add slack_bot.py to the GitHub repository
and optionally add slack-bolt to requirements.txt for reproducibility.
Do not commit .env, xapp or xoxb tokens.
