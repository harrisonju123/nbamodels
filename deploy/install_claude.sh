#!/bin/bash
# Install Claude Code CLI on Ubuntu Droplet

set -e

echo "=== Installing Claude Code CLI ==="
echo

echo "1. Installing Node.js (required for Claude Code)..."
# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
echo "✓ Node.js installed"
echo "Node version: $(node --version)"
echo "NPM version: $(npm --version)"
echo

echo "2. Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code
echo "✓ Claude Code CLI installed"
echo

echo "3. Verifying installation..."
claude --version
echo

echo "=== Installation Complete! ==="
echo
echo "Next steps:"
echo "1. Authenticate Claude Code:"
echo "   claude auth login"
echo
echo "2. This will give you a URL to visit in your browser"
echo "3. Sign in with your Anthropic account"
echo "4. Copy the authentication code back to the terminal"
echo
echo "After authentication, you can use Claude Code:"
echo "  - Start a session: claude"
echo "  - Run a command: claude 'fix the dashboard error'"
echo "  - Exit session: Type 'exit' or Ctrl+D"
echo
