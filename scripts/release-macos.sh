#!/usr/bin/env sh
# Builds a universal macOS DMG, signs it with Developer ID, notarizes it
# and uploads it to the GitHub release for the current version.
# Push the tag first so the release exists (the Windows job creates it).
# Run from the repository root: sh scripts/release-macos.sh
#
# The App Store Connect API key stays outside the repository:
#   ~/.appstoreconnect/rustyfem.env   APPLE_API_ISSUER=... and APPLE_API_KEY=...
#   ~/.appstoreconnect/private_keys/AuthKey_<APPLE_API_KEY>.p8
set -e

env="$HOME/.appstoreconnect/rustyfem.env"
[ -f "$env" ] || { echo "Missing $env (APPLE_API_ISSUER and APPLE_API_KEY)."; exit 1; }
set -a
. "$env"
set +a

export PATH="$HOME/.cargo/bin:$PATH"
export APPLE_SIGNING_IDENTITY="Developer ID Application: Westin Digital AB (MQTBFQ9Q83)"
export APPLE_API_KEY_PATH="$HOME/.appstoreconnect/private_keys/AuthKey_$APPLE_API_KEY.p8"

version=$(sed -n 's/^  "version": "\(.*\)",$/\1/p' app/src-tauri/tauri.conf.json)
tag="v$version"
dmg="target/universal-apple-darwin/release/bundle/dmg/RustyFEM_${version}_universal.dmg"

gh release view "$tag" >/dev/null || { echo "No release $tag on GitHub yet: push the tag first."; exit 1; }

echo "== build, sign and notarize the app ($tag)"
rustup target add aarch64-apple-darwin x86_64-apple-darwin >/dev/null
(cd web && npm run tauri build -- --bundles dmg --target universal-apple-darwin)

# Tauri notarizes the .app but not the .dmg around it.
echo "== notarize the DMG"
xcrun notarytool submit "$dmg" --key "$APPLE_API_KEY_PATH" \
  --key-id "$APPLE_API_KEY" --issuer "$APPLE_API_ISSUER" --wait
xcrun stapler staple "$dmg"
spctl -a -t install "$dmg"

echo "== upload to $tag"
gh release upload "$tag" "$dmg" --clobber
