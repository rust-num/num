#!/bin/sh
# Use rustup to locally run the same suite of tests as .travis.yml.
# (You should first install/update 1.8.0, 1.15.0, beta, and nightly.)

set -ex

export TRAVIS_RUST_VERSION
for TRAVIS_RUST_VERSION in 1.8.0 1.15.0 beta nightly; do
    run="rustup run $TRAVIS_RUST_VERSION"
    if [ "$TRAVIS_RUST_VERSION" = 1.8.0 ]; then
      # libc 0.2.34 started using #[deprecated]
      $run cargo generate-lockfile
      $run cargo update --package libc --precise 0.2.33 || :
    fi
    $run cargo build --verbose
    $run $PWD/ci/test_full.sh
    $run cargo doc
done
