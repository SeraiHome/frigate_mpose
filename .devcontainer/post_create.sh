#!/bin/bash

set -euxo pipefail

# MediaPipe Tasks requires OpenGL ES 2.0 (libGLESv2.so.2) for its native
# C bindings.  The production Docker image ships this via mesa, but the
# devcontainer base image may not include it.
sudo apt-get update -qq && sudo apt-get install -y -qq libegl1 libgles2 libgl1-mesa-glx >/dev/null 2>&1 || true

# Ensure git treats the mounted workspace as safe (ownership differs on Windows hosts)
git config --global --add safe.directory /workspace/frigate 2>/dev/null || true

# Windows host mounts files with CRLF; tell git to handle the conversion
# transparently (CRLF in working tree ↔ LF in repo) to prevent phantom diffs
git config core.autocrlf true

# Setup user-level npm global directory (avoids EACCES errors with global installs)
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
if ! grep -q 'npm-global' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.bashrc"
fi

# Cleanup the old github host key
if [[ -f ~/.ssh/known_hosts ]]; then
  # Add new github host key
  sed -i -e '/AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31\/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi\/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ==/d' ~/.ssh/known_hosts
  curl -L https://api.github.com/meta | jq -r '.ssh_keys | .[]' | \
    sed -e 's/^/github.com /' >> ~/.ssh/known_hosts
fi

# Frigate normal container runs as root, so it have permission to create
# the folders. But the devcontainer runs as the host user, so we need to
# create the folders and give the host user permission to write to them.
sudo mkdir -p /media/frigate
sudo chown -R "$(id -u):$(id -g)" /media/frigate

# When started as a service, LIBAVFORMAT_VERSION_MAJOR is defined in the
# s6 service file. For dev, where frigate is started from an interactive
# shell, we define it in .bashrc instead.
echo 'export LIBAVFORMAT_VERSION_MAJOR=$("$(python3 /usr/local/ffmpeg/get_ffmpeg_path.py)" -version | grep -Po "libavformat\W+\K\d+")' >> "$HOME/.bashrc"

# Setup user-level npm global directory (avoids EACCES on global installs)
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
if ! grep -q 'npm-global' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.bashrc"
fi
export PATH="$HOME/.npm-global/bin:$PATH"

# Install Claude Code CLI (the VSCode extension delegates to this)
npm install -g @anthropic-ai/claude-code

# Setup user-level npm global directory (avoids EACCES on global installs)
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
if ! grep -q 'npm-global' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.bashrc"
fi
export PATH="$HOME/.npm-global/bin:$PATH"

# Install Claude Code CLI (the VSCode extension delegates to this)
npm install -g @anthropic-ai/claude-code

if [[ -f Makefile ]] && git rev-parse --git-dir &>/dev/null; then
  make version
else
  echo "Skipping 'make version' — Makefile or git history not available" >&2
fi

cd web

npm install

npm run build
