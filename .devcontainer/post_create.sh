#!/bin/bash

set -euxo pipefail

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
if command -v sudo &>/dev/null; then
  sudo mkdir -p /media/frigate
  sudo chown -R "$(id -u):$(id -g)" /media/frigate
else
  mkdir -p /media/frigate 2>/dev/null || true
fi

# When started as a service, LIBAVFORMAT_VERSION_MAJOR is defined in the
# s6 service file. For dev, where frigate is started from an interactive
# shell, we define it in .bashrc instead.
echo 'export LIBAVFORMAT_VERSION_MAJOR=$("$(python3 /usr/local/ffmpeg/get_ffmpeg_path.py)" -version | grep -Po "libavformat\W+\K\d+")' >> "$HOME/.bashrc"

if [[ -f Makefile ]] && git rev-parse --git-dir &>/dev/null; then
  make version
else
  echo "Skipping 'make version' — Makefile or git history not available" >&2
fi

cd web

npm install

npm run build
