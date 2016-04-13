#!/bin/sh

set -ex

cargo bench --verbose

cargo test --verbose --manifest-path=macros/Cargo.toml
