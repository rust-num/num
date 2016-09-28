#!/bin/sh

set -ex

cargo bench --verbose

cargo test --verbose --manifest-path=macros/Cargo.toml
cargo test --verbose --manifest-path=derive/Cargo.toml

# Build test for the serde feature
cargo build --verbose --features "serde"

# Downgrade serde and build test the 0.7.0 channel as well
cargo update -p serde --precise 0.7.0
cargo build --verbose --features "serde"
