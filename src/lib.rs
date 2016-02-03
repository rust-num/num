// Copyright 2014-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A collection of numeric types and traits for Rust.
//!
//! This includes new types for big integers, rationals, and complex numbers,
//! new traits for generic programming on numeric properties like `Integer`,
//! and generic range iterators.
//!
//! ## Example
//!
//! This example uses the BigRational type and [Newton's method][newt] to
//! approximate a square root to arbitrary precision:
//!
//! ```
//! extern crate num;
//! # #[cfg(all(feature = "bigint", feature="rational"))]
//! # mod test {
//!
//! use num::FromPrimitive;
//! use num::bigint::BigInt;
//! use num::rational::{Ratio, BigRational};
//!
//! # pub
//! fn approx_sqrt(number: u64, iterations: usize) -> BigRational {
//!     let start: Ratio<BigInt> = Ratio::from_integer(FromPrimitive::from_u64(number).unwrap());
//!     let mut approx = start.clone();
//!
//!     for _ in 0..iterations {
//!         approx = (&approx + (&start / &approx)) /
//!             Ratio::from_integer(FromPrimitive::from_u64(2).unwrap());
//!     }
//!
//!     approx
//! }
//! # }
//! # #[cfg(not(all(feature = "bigint", feature="rational")))]
//! # mod test { pub fn approx_sqrt(n: u64, _: usize) -> u64 { n } }
//! # use test::approx_sqrt;
//!
//! fn main() {
//!     println!("{}", approx_sqrt(10, 4)); // prints 4057691201/1283082416
//! }
//!
//! ```
//!
//! [newt]: https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Babylonian_method
#![doc(html_logo_url = "http://rust-num.github.io/num/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://rust-num.github.io/num/favicon.ico",
       html_root_url = "http://rust-num.github.io/num/",
       html_playground_url = "http://play.rust-lang.org/")]

#[cfg(feature = "rustc-serialize")]
extern crate rustc_serialize;

// Some of the tests of non-RNG-based functionality are randomized using the
// RNG-based functionality, so the RNG-based functionality needs to be enabled
// for tests.
#[cfg(any(feature = "rand", all(feature = "bigint", test)))]
extern crate rand;

#[cfg(feature = "bigint")]
pub use bigint::{BigInt, BigUint};
#[cfg(feature = "rational")]
pub use rational::Rational;
#[cfg(all(feature = "rational", feature="bigint"))]
pub use rational::BigRational;
#[cfg(feature = "complex")]
pub use complex::Complex;
pub use integer::Integer;
pub use iter::{range, range_inclusive, range_step, range_step_inclusive};
pub use traits::{Num, Zero, One, Signed, Unsigned, Bounded,
                 Saturating, CheckedAdd, CheckedSub, CheckedMul, CheckedDiv,
                 PrimInt, Float, ToPrimitive, FromPrimitive, NumCast, cast};

#[cfg(test)] use std::hash;

use std::ops::{Mul};

#[cfg(feature = "bigint")]
pub mod bigint;
pub mod complex;
pub mod integer;
pub mod iter;
pub mod traits;
#[cfg(feature = "rational")]
pub mod rational;

/// Returns the additive identity, `0`.
#[inline(always)] pub fn zero<T: Zero>() -> T { Zero::zero() }

/// Returns the multiplicative identity, `1`.
#[inline(always)] pub fn one<T: One>() -> T { One::one() }

/// Computes the absolute value.
///
/// For `f32` and `f64`, `NaN` will be returned if the number is `NaN`
///
/// For signed integers, `::MIN` will be returned if the number is `::MIN`.
#[inline(always)]
pub fn abs<T: Signed>(value: T) -> T {
    value.abs()
}

/// The positive difference of two numbers.
///
/// Returns zero if `x` is less than or equal to `y`, otherwise the difference
/// between `x` and `y` is returned.
#[inline(always)]
pub fn abs_sub<T: Signed>(x: T, y: T) -> T {
    x.abs_sub(&y)
}

/// Returns the sign of the number.
///
/// For `f32` and `f64`:
///
/// * `1.0` if the number is positive, `+0.0` or `INFINITY`
/// * `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
/// * `NaN` if the number is `NaN`
///
/// For signed integers:
///
/// * `0` if the number is zero
/// * `1` if the number is positive
/// * `-1` if the number is negative
#[inline(always)] pub fn signum<T: Signed>(value: T) -> T { value.signum() }

/// Raises a value to the power of exp, using exponentiation by squaring.
///
/// # Example
///
/// ```rust
/// use num;
///
/// assert_eq!(num::pow(2i8, 4), 16);
/// assert_eq!(num::pow(6u8, 3), 216);
/// ```
#[inline]
pub fn pow<T: Clone + One + Mul<T, Output = T>>(mut base: T, mut exp: usize) -> T {
    if exp == 0 { return T::one() }

    while exp & 1 == 0 {
        base = base.clone() * base;
        exp >>= 1;
    }
    if exp == 1 { return base }

    let mut acc = base.clone();
    while exp > 1 {
        exp >>= 1;
        base = base.clone() * base;
        if exp & 1 == 1 {
            acc = acc * base.clone();
        }
    }
    acc
}

/// Raises a value to the power of exp, returning `None` if an overflow occured.
///
/// Otherwise same as the `pow` function.
///
/// # Example
///
/// ```rust
/// use num;
///
/// assert_eq!(num::checked_pow(2i8, 4), Some(16));
/// assert_eq!(num::checked_pow(7i8, 8), None);
/// assert_eq!(num::checked_pow(7u32, 8), Some(5_764_801));
/// ```
#[inline]
pub fn checked_pow<T: Clone + One + CheckedMul>(mut base: T, mut exp: usize) -> Option<T> {
    if exp == 0 { return Some(T::one()) }

    macro_rules! optry {
        ( $ expr : expr ) => {
            if let Some(val) = $expr { val } else { return None }
        }
    }

    while exp & 1 == 0 {
        base = optry!(base.checked_mul(&base));
        exp >>= 1;
    }
    if exp == 1 { return Some(base) }

    let mut acc = base.clone();
    while exp > 1 {
        exp >>= 1;
        base = optry!(base.checked_mul(&base));
        if exp & 1 == 1 {
            acc = optry!(acc.checked_mul(&base));
        }
    }
    Some(acc)
}

#[cfg(test)]
fn hash<T: hash::Hash>(x: &T) -> u64 {
    use std::hash::Hasher;
    let mut hasher = hash::SipHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}
