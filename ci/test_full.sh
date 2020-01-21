#!/bin/bash

set -ex

echo Testing num on rustc ${TRAVIS_RUST_VERSION}

case "$TRAVIS_RUST_VERSION" in
  1.31.*) STD_FEATURES="serde" ;;
  *) STD_FEATURES="serde rand" ;;
esac

case "$TRAVIS_RUST_VERSION" in
  1.3[1-5].*) ;;
  *) ALLOC_FEATURES="serde rand" ;;
esac

# num should build and test everywhere.
cargo build --verbose
cargo test --verbose

# It should build with minimal features too.
cargo build --no-default-features
cargo test --no-default-features

# Each isolated feature should also work everywhere.
# (but still with "std", else bigint breaks)
for feature in $STD_FEATURES; do
  cargo build --verbose --no-default-features --features="std $feature"
  cargo test --verbose --no-default-features --features="std $feature"
done

# test all supported features together
cargo build --features="std $STD_FEATURES"
cargo test --features="std $STD_FEATURES"

if test -n "${ALLOC_FEATURES:+true}"; then
  # It should build with minimal features too.
  cargo build --no-default-features --features="alloc"
  cargo test --no-default-features --features="alloc"

  # Each isolated feature should also work everywhere.
  for feature in $ALLOC_FEATURES; do
    cargo build --verbose --no-default-features --features="alloc $feature"
    cargo test --verbose --no-default-features --features="alloc $feature"
  done

  # test all supported features together
  cargo build --no-default-features --features="alloc $ALLOC_FEATURES"
  cargo test --no-default-features --features="alloc $ALLOC_FEATURES"
fi
