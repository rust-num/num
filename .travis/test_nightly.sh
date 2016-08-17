#!/bin/sh

set -ex

cargo bench --verbose

cargo test --verbose --manifest-path=macros/Cargo.toml

# Build test for the serde feature
cargo build --verbose --features "serde"
