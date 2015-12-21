#!/bin/sh
# Use multirust to locally run the same suite of tests as .travis.yml.
# (You should first install/update 1.0.0, beta, and nightly.)

set -ex

for toolchain in 1.0.0 beta nightly; do
    run="multirust run $toolchain"
    $run cargo build --verbose
    $run cargo test --verbose
    $run .travis/test_features.sh
    if [ $toolchain = nightly ]; then
        $run .travis/test_nightly.sh
    fi
    $run cargo doc
done
