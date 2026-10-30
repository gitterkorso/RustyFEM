#!/usr/bin/env sh
# Build checks for the engine and the app.
# Run from the repository root: sh scripts/check.sh
set -e

echo "== tests (app API; the engine tests live in ../RustyFEM tests)"
cargo test --release -p rustyfem-session
(cd "../RustyFEM tests" && cargo test --release)

echo "== engine without cli/dynlib (the app's feature set)"
cargo build --release -p rustyfem --lib --no-default-features

echo "== type check for all targets"
rustup target add aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin x86_64-unknown-linux-gnu >/dev/null
for t in aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin x86_64-unknown-linux-gnu; do
  CARGO_TARGET_DIR=target/cross cargo check -q -p rustyfem-session --target "$t"
  echo "   $t ok"
done

echo "== user interface"
(cd web && npm ci --silent && npx tsc && npx vite build --logLevel warn)
echo "All checks passed"
