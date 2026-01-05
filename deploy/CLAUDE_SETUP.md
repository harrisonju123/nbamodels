# Claude Code CLI on Droplet

## Quick Install

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Pull latest deployment scripts
cd /root/nbamodels
git pull

# Run installation script
chmod +x deploy/install_claude.sh
./deploy/install_claude.sh
```

---

## Authentication

After installation, authenticate Claude Code:

```bash
claude auth login
```

This will:
1. Show you a URL to visit in your browser
2. Ask you to sign in with your Anthropic account
3. Give you an authentication code
4. Paste the code back in the terminal

---

## Using Claude Code on the Droplet

### Start an Interactive Session

```bash
cd /root/nbamodels
claude
```

Then you can chat with Claude and ask it to:
- Debug issues
- Check logs
- Edit files
- Run commands
- Analyze errors

### Run a Single Command

```bash
# Check why dashboard won't start
claude "check why the dashboard service is failing"

# Fix a specific error
claude "fix the ModuleNotFoundError in the logs"

# Analyze performance
claude "check if the API is responding correctly"
```

### Useful Commands for Your Deployment

```bash
# Debug service issues
claude "check the dashboard service logs and tell me what's wrong"

# Verify deployment
claude "verify all services are running correctly"

# Check database
claude "show me the latest bets in the database"

# Monitor resources
claude "check if we're running out of memory or disk space"
```

---

## Why Use Claude on the Server?

1. **Debug in real-time** - Claude can see actual logs and errors
2. **Quick fixes** - Make changes without local git workflow
3. **Server context** - Claude knows the exact server state
4. **Learn commands** - Claude shows you the Linux commands to use

---

## Example Session

```bash
root@droplet:~/nbamodels# claude

You: why is the dashboard not loading?

Claude: Let me check the service logs...
[Runs: journalctl -u nba-dashboard -n 50]

I see the issue - ModuleNotFoundError for 'nba_api'.
Let me check if it's in requirements.txt...
[Runs: grep nba_api requirements.txt]

It's there but not installed. Let me install it:
[Runs: pip install nba_api]

Now let me restart the service:
[Runs: systemctl restart nba-dashboard]

Done! The dashboard should now be working.

You: exit
```

---

## Tips

1. **Always cd to project directory first**
   ```bash
   cd /root/nbamodels
   claude
   ```

2. **Use for debugging, not just coding**
   - Check why cron jobs aren't running
   - Analyze error logs
   - Monitor resource usage
   - Verify configurations

3. **Exit sessions when done**
   - Type `exit` or press Ctrl+D
   - This keeps your session history clean

4. **Check Claude status**
   ```bash
   claude auth status
   ```

---

## Troubleshooting

### "command not found: claude"

```bash
# Check if installed
npm list -g @anthropic-ai/claude-code

# If not installed, run:
npm install -g @anthropic-ai/claude-code
```

### "Not authenticated"

```bash
claude auth login
```

### "Permission denied"

```bash
# Make sure you're running as root
whoami

# Or use sudo
sudo claude
```

---

## Cost Note

Claude Code CLI usage counts toward your Anthropic API usage. Each request costs the same as API calls:
- Sonnet: ~$3 per million input tokens
- For debugging/deployment tasks, costs are typically minimal (<$1/month)

---

## Next Steps After Installation

1. Authenticate: `claude auth login`
2. Test it: `claude "show me the current directory contents"`
3. Debug deployment: `claude "check if all services are running"`
4. Use it regularly for server management!
