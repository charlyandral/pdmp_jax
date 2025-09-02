#!/usr/bin/env bash
set -euo pipefail

EXTRA=""

runs=(
  "0.4.31::"                                      # no flag
  "0.4.33::"                                      # no flag
  "0.4.33::--xla_cpu_use_thunk_runtime=false"     # with flag
  "0.5.3::"
  "0.5.3::--xla_cpu_use_thunk_runtime=false"
  "0.6.2::"
  "0.6.2::--xla_cpu_use_thunk_runtime=false"
  "0.7.0::"
  # "0.7.1::"
)

for entry in "${runs[@]}"; do
  ver="${entry%%::*}"
  flag="${entry##*::}"
  spec="jax${EXTRA:+[$EXTRA]}==${ver}"

  echo "==============================="
  echo "JAX $ver   XLA_FLAGS=${flag:-<unset>}"
  echo "==============================="

  if [ -n "$flag" ]; then
    XLA_FLAGS="$flag" uv run --no-project --with "$spec" python benchmark.py
  else
    uv run --no-project --with "$spec" python benchmark.py
  fi
done