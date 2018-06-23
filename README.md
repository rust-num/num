# num

[![crate](https://img.shields.io/crates/v/num.svg)](https://crates.io/crates/num)
[![documentation](https://docs.rs/num/badge.svg)](https://docs.rs/num)
![minimum rustc 1.15](https://img.shields.io/badge/rustc-1.15+-red.svg)
[![Travis status](https://travis-ci.org/rust-num/num.svg?branch=master)](https://travis-ci.org/rust-num/num)

A collection of numeric types and traits for Rust.

This includes new types for big integers, rationals, and complex numbers,
new traits for generic programming on numeric properties like `Integer`,
and generic range iterators.

`num` is a meta-crate, re-exporting items from these sub-crates:

- [`num-bigint`](https://github.com/rust-num/num-bigint)
  [![crate](https://img.shields.io/crates/v/num-bigint.svg)](https://crates.io/crates/num-bigint)

- [`num-complex`](https://github.com/rust-num/num-complex)
  [![crate](https://img.shields.io/crates/v/num-complex.svg)](https://crates.io/crates/num-complex)

- [`num-integer`](https://github.com/rust-num/num-integer)
  [![crate](https://img.shields.io/crates/v/num-integer.svg)](https://crates.io/crates/num-integer)

- [`num-iter`](https://github.com/rust-num/num-iter)
  [![crate](https://img.shields.io/crates/v/num-iter.svg)](https://crates.io/crates/num-iter)

- [`num-rational`](https://github.com/rust-num/num-rational)
  [![crate](https://img.shields.io/crates/v/num-rational.svg)](https://crates.io/crates/num-rational)

- [`num-traits`](https://github.com/rust-num/num-traits)
  [![crate](https://img.shields.io/crates/v/num-traits.svg)](https://crates.io/crates/num-traits)

There is also a `proc-macro` crate for deriving some numeric traits:

- [`num-derive`](https://github.com/rust-num/num-derive)
  [![crate](https://img.shields.io/crates/v/num-derive.svg)](https://crates.io/crates/num-derive)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
num = "0.2"
```

and this to your crate root:

```rust
extern crate num;
```

## Features

This crate can be used without the standard library (`#![no_std]`) by disabling
the default `std` feature. Use this in `Cargo.toml`:

```toml
[dependencies.num]
version = "0.2"
default-features = false
```

The `num-bigint` crate is only available when `std` is enabled, and the other
sub-crates may have limited functionality when used without `std`.

Implementations for `i128` and `u128` are only available with Rust 1.26 and
later.  The build script automatically detects this, but you can make it
mandatory by enabling the `i128` crate feature.

The `rand` feature enables randomization traits in `num-bigint` and
`num-complex`.

The `serde` feature enables serialization for types in `num-bigint`,
`num-complex`, and `num-rational`.

The `num` meta-crate no longer supports features to toggle the inclusion of
the individual sub-crates.  If you need such control, you are recommended to
directly depend on your required crates instead.

## Releases

Release notes are available in [RELEASES.md](RELEASES.md).

## Compatibility

The `num` crate as a whole is tested for rustc 1.15 and greater.

The `num-traits`, `num-integer`, and `num-iter` crates are individually tested
for rustc 1.8 and greater, if you require such older compatibility.
