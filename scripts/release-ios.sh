#!/usr/bin/env sh
# Builds a signed App Store IPA for iOS; with --upload it is also sent to
# App Store Connect (TestFlight). Raise CFBundleVersion in
# app/src-tauri/gen/apple/project.yml (and Info.plist) before each upload.
# Run from the repository root: sh scripts/release-ios.sh [--upload]
#
# `tauri ios build` does not recognise the iOS platform in Xcode 27, so this
# script does its steps itself: the frontend, the Rust library and then
# xcodebuild with the Rust build phase skipped (SKIP_RUST_BUILD).
#
# The App Store Connect API key and the team ID stay outside the repository:
#   ~/.appstoreconnect/rustyfem.env   APPLE_API_ISSUER=..., APPLE_API_KEY=...
#                                     and APPLE_TEAM_ID=...
#   ~/.appstoreconnect/private_keys/AuthKey_<APPLE_API_KEY>.p8
set -e

env="$HOME/.appstoreconnect/rustyfem.env"
[ -f "$env" ] || { echo "Missing $env (APPLE_API_ISSUER, APPLE_API_KEY and APPLE_TEAM_ID)."; exit 1; }
set -a
. "$env"
set +a
[ -n "$APPLE_TEAM_ID" ] || { echo "APPLE_TEAM_ID is missing from $env."; exit 1; }

export PATH="$HOME/.cargo/bin:$PATH"
key="$HOME/.appstoreconnect/private_keys/AuthKey_$APPLE_API_KEY.p8"
apple=app/src-tauri/gen/apple
out=target/ios-release
auth="-allowProvisioningUpdates -authenticationKeyPath $key -authenticationKeyID $APPLE_API_KEY -authenticationKeyIssuerID $APPLE_API_ISSUER"

echo "== frontend"
(cd web && npm run build)

echo "== Rust library"
rustup target add aarch64-apple-ios >/dev/null
# custom-protocol embeds the frontend instead of loading the dev server
cargo build --manifest-path app/src-tauri/Cargo.toml --lib --release \
  --target aarch64-apple-ios --features tauri/custom-protocol
mkdir -p "$apple/Externals/arm64/release"
cp target/aarch64-apple-ios/release/librustyfem_app_lib.a "$apple/Externals/arm64/release/libapp.a"

echo "== archive"
rm -rf "$out"
xcodebuild -project "$apple/rustyfem-app.xcodeproj" -scheme rustyfem-app_iOS \
  -configuration release -destination "generic/platform=iOS" \
  -archivePath "$out/RustyFEM.xcarchive" $auth SKIP_RUST_BUILD=1 DEVELOPMENT_TEAM="$APPLE_TEAM_ID" archive

echo "== export"
xcodebuild -exportArchive -archivePath "$out/RustyFEM.xcarchive" \
  -exportOptionsPlist "$apple/ExportOptions.plist" -exportPath "$out" $auth
ipa="$out/RustyFEM.ipa"
echo "$ipa"

if [ "$1" = "--upload" ]; then
  echo "== upload to App Store Connect"
  xcrun altool --upload-app -f "$ipa" -t ios --apiKey "$APPLE_API_KEY" --apiIssuer "$APPLE_API_ISSUER"
fi
