#!/bin/sh

set -ex

cargo bench --verbose

cargo test --verbose --manifest-path=num-macros/Cargo.toml
