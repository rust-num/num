#!/bin/sh

set -ex

for feature in '' bigint rational complex; do
  cargo build --verbose --no-default-features --features="$feature"
done
