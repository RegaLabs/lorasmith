#!/usr/bin/env bash
# Builds the release binary for the machine this runs on and packs it with the README and
# licenses into dist/loraforge-<target>.tar.gz, or .zip on Windows.
set -euo pipefail
cd "$(dirname "$0")/.."

target=$(rustc -vV | sed -n 's/^host: //p')
name=loraforge-$target
cargo build --release --locked

rm -rf "dist/$name"
mkdir -p "dist/$name"
cp README.md LICENSE-APACHE LICENSE-MIT "dist/$name/"
cd dist
if [[ $target == *windows* ]]; then
    cp ../target/release/loraforge.exe "$name/"
    rm -f "$name.zip"
    7z a "$name.zip" "$name"
else
    cp ../target/release/loraforge "$name/"
    tar -czf "$name.tar.gz" "$name"
fi
rm -r "$name"
