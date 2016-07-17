#!/bin/sh
# Use rustup to locally run the same suite of tests as .travis.yml.
# (You should first install/update 1.0.0, beta, and nightly.)

set -ex

for toolchain in 1.0.0 beta nightly; do
    run="rustup run $toolchain"
    $run cargo build --verbose
    $run /usr/bin/env make test
    $run $PWD/.travis/test_features.sh
    if [ $toolchain = nightly ]; then
        $run $PWD/.travis/test_nightly.sh
    fi
    $run cargo doc
done
