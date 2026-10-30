#!/usr/bin/env sh
# Byggkontroller för motorn och appen (PLAN.md, Fas 0 och 2).
# Kör från repots rot: sh scripts/check.sh
set -e

echo "== tester (app-API; motorns tester ligger i ../RustyFEM tests)"
cargo test --release -p rustyfem-session
(cd "../RustyFEM tests" && cargo test --release)

echo "== motorn utan cli/dynlib (appens feature-uppsättning)"
cargo build --release -p rustyfem --lib --no-default-features

echo "== typkontroll för alla mål"
rustup target add aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin x86_64-unknown-linux-gnu >/dev/null
for t in aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin x86_64-unknown-linux-gnu; do
  CARGO_TARGET_DIR=target/cross cargo check -q -p rustyfem-session --target "$t"
  echo "   $t ok"
done

echo "== gränssnittet"
(cd web && npm ci --silent && npx tsc && npx vite build --logLevel warn)
echo "Alla kontroller gick igenom"
