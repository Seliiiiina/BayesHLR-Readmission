#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <path-to-tex-file>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
input_path="$1"

if [[ "$input_path" = /* ]]; then
  tex_file="$input_path"
else
  tex_file="$repo_root/$input_path"
fi

if [[ ! -f "$tex_file" ]]; then
  echo "TeX file not found: $tex_file" >&2
  exit 1
fi

tex_dir="$(cd "$(dirname "$tex_file")" && pwd)"
tex_name="$(basename "$tex_file")"
repo_tectonic="$repo_root/.texenv/bin/tectonic"

cd "$tex_dir"

if command -v latexmk >/dev/null 2>&1; then
  exec latexmk -pdf -interaction=nonstopmode -synctex=1 "$tex_name"
fi

export XDG_CACHE_HOME="$repo_root/.cache"
export XDG_CONFIG_HOME="$repo_root/.config"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"

if [[ -x "$repo_tectonic" ]]; then
  exec "$repo_tectonic" --synctex --keep-logs "$tex_name"
fi

if command -v tectonic >/dev/null 2>&1; then
  exec tectonic --synctex --keep-logs "$tex_name"
fi

if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -synctex=1 "$tex_name"
  pdflatex -interaction=nonstopmode -synctex=1 "$tex_name"
  exit 0
fi

echo "No LaTeX compiler found in PATH." >&2
echo "Install TeX Live/Tectonic or load the cluster's TeX module, then try again." >&2
exit 1
