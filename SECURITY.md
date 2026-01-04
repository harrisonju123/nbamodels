# Security Notes

## API Key Management

**IMPORTANT**: Never commit API keys to version control.

### Setup Instructions

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your actual API keys

3. Verify `.env` is in `.gitignore` (it should be)

### If API Keys Were Exposed

If you accidentally committed API keys:

1. **Rotate the keys immediately** by generating new ones from:
   - Ball Don't Lie: https://www.balldontlie.io/
   - The Odds API: https://the-odds-api.com/

2. Remove from git history:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   ```

3. Force push (if repo is shared):
   ```bash
   git push origin --force --all
   ```

### Current Status

✅ `.env` is in `.gitignore`
✅ `.env.example` template created
✅ No `.env` found in git history (verified)

**Action Required**: Consider rotating API keys as a precaution if this repo was ever public or shared.
