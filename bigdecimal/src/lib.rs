// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A Big decimal
//!
//! A `BigDecimal` is represented as a vector of `BigDigit`s.
//!
//! Common numerical operations are overloaded, so we can treat them
//! the same way we treat other numbers.
//!
//! ## Example
//!
//! ```rust
//! extern crate num_bigdecimal;
//! extern crate num_traits;
//!
//! # fn main() {
//! use num_bigdecimal::BigDecimal;
//! use num_traits::{Zero, One};
//! use std::mem::replace;
//!
//! // Calculate large fibonacci numbers.
//! fn fib(n: usize) -> BigDecimal {
//!     let mut f0: BigUint = Zero::zero();
//!     let mut f1: BigUint = One::one();
//!     for _ in 0..n {
//!         let f2 = f0 + &f1;
//!         // This is a low cost way of swapping f0 with f1 and f1 with f2.
//!         f0 = replace(&mut f1, f2);
//!     }
//!     f0
//! }
//!
//! // This is a very large number.
//! println!("fib(1000) = {}", fib(1000));
//! # }
//! ```
//!
//! It's easy to generate large random numbers:
//!
//! ```rust
//! extern crate rand;
//! extern crate num_bigint as bigint;
//!
//! # #[cfg(feature = "rand")]
//! # fn main() {
//! use bigint::{ToBigInt, RandBigInt};
//!
//! let mut rng = rand::thread_rng();
//! let a = rng.gen_bigint(1000);
//!
//! let low = -10000.to_bigint().unwrap();
//! let high = 10000.to_bigint().unwrap();
//! let b = rng.gen_bigint_range(&low, &high);
//!
//! // Probably an even larger number.
//! println!("{}", a * b);
//! # }
//!
//! # #[cfg(not(feature = "rand"))]
//! # fn main() {
//! # }
//! ```

#[cfg(any(feature = "rand", test))]
extern crate rand;
#[cfg(feature = "rustc-serialize")]
extern crate rustc_serialize;
#[cfg(feature = "serde")]
extern crate serde;

extern crate num_bigint as bigint;
extern crate num_integer as integer;
extern crate num_traits as traits;

use std::default::Default;
use std::error::Error;
use std::num::ParseFloatError;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::str::{self, FromStr};
use std::fmt;
use std::cmp::Ordering::Equal;
use std::cmp::max;
use bigint::{BigInt, ParseBigIntError};
use traits::{Num, Zero, One};

macro_rules! forward_val_val_binop {
    (impl $imp:ident for $res:ty, $method:ident) => {
        impl $imp<$res> for $res {
            type Output = $res;

            #[inline]
            fn $method(self, other: $res) -> $res {
                // forward to val-ref
                $imp::$method(self, &other)
            }
        }
    }
}

macro_rules! forward_ref_val_binop {
    (impl $imp:ident for $res:ty, $method:ident) => {
        impl<'a> $imp<$res> for &'a $res {
            type Output = $res;

            #[inline]
            fn $method(self, other: $res) -> $res {
                // forward to ref-ref
                $imp::$method(self, &other)
            }
        }
    }
}

macro_rules! forward_val_ref_binop {
    (impl $imp:ident for $res:ty, $method:ident) => {
        impl<'a> $imp<&'a $res> for $res {
            type Output = $res;

            #[inline]
            fn $method(self, other: &$res) -> $res {
                // forward to ref-ref
                $imp::$method(&self, other)
            }
        }
    }
}

macro_rules! forward_ref_ref_binop {
    (impl $imp:ident for $res:ty, $method:ident) => {
        impl<'a, 'b> $imp<&'b $res> for &'a $res {
            type Output = $res;

            #[inline]
            fn $method(self, other: &$res) -> $res {
                // forward to val-ref
                $imp::$method(self.clone(), other)
            }
        }
    }
}

// Forward everything to ref-ref, when reusing storage is not helpful
macro_rules! forward_all_binop_to_ref_ref {
    (impl $imp:ident for $res:ty, $method:ident) => {
        forward_val_val_binop!(impl $imp for $res, $method);
        forward_val_ref_binop!(impl $imp for $res, $method);
        forward_ref_val_binop!(impl $imp for $res, $method);
    };
}

/// A big decimal type.
///
/// A `BigUint`-typed value `BigUint { data: vec!(a, b, c) }` represents a number
/// `(a + b * big_digit::BASE + c * big_digit::BASE^2)`.
#[derive(Clone, Debug, Hash)]
#[cfg_attr(feature = "rustc-serialize", derive(RustcEncodable, RustcDecodable))]
pub struct BigDecimal {
    int_val: BigInt,
    scale: i64,
}

impl BigDecimal {
    /// Creates and initializes a `BigUint`.
    ///
    /// The digits are in little-endian base 2^32.
    #[inline]
    pub fn new(digits: BigInt, scale: i64) -> BigDecimal {
        BigDecimal { int_val: digits, scale: scale }
    }

    /// Creates and initializes a `BigDecimal`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_bigdecimal::{BigDecimal};
    /// use num_traits::Zero;
    ///
    /// assert_eq!(BigDecimal::parse_bytes(b"1234.12", 10), BigDecimal::zero());
    /// assert_eq!(BigDecimal::parse_bytes(b"ABCD.1", 16), BigDecimal::zero());
    /// assert_eq!(BigDecimal::parse_bytes(b"G", 16), None);
    /// ```
    #[inline]
    pub fn parse_bytes(buf: &[u8], radix: u32) -> Option<BigDecimal> {
        str::from_utf8(buf).ok().and_then(|s| BigDecimal::from_str_radix(s, radix).ok())
    }

    pub fn set_scale(&self, new_scale: i64) -> BigDecimal {
        if self.scale == new_scale {
            return self.clone();
        }

        if self.int_val.is_zero() {
            return BigDecimal::new(BigInt::zero(), new_scale);
        }

        if new_scale > self.scale {
            let raise = new_scale - self.scale;
            println!("raise: {}", raise);
            let exp = (10 as i64).pow(raise as u32);
            let new = BigDecimal::new(self.int_val.clone() * BigInt::from(exp), new_scale);
            println!("new: {}", new);
            return new;
        }

        //todo implement
        self.clone()
    }
}

#[derive(Debug, PartialEq)]
pub enum ParseBigDecimalError {
    ParseDecimal(ParseFloatError),
    Other,
}

impl fmt::Display for ParseBigDecimalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &ParseBigDecimalError::ParseDecimal(ref e) => e.fmt(f),
            &ParseBigDecimalError::Other => "failed to parse provided string".fmt(f),
        }
    }
}

impl Error for ParseBigDecimalError {
    fn description(&self) -> &str {
        "failed to parse bigint/biguint"
    }
}

impl From<ParseFloatError> for ParseBigDecimalError {
    fn from(err: ParseFloatError) -> ParseBigDecimalError {
        ParseBigDecimalError::ParseDecimal(err)
    }
}

impl From<ParseBigIntError> for ParseBigDecimalError {
    fn from(_: ParseBigIntError) -> ParseBigDecimalError {
        // ParseBigDecimalError::ParseDecimal(err)
        ParseBigDecimalError::Other
    }
}

impl FromStr for BigDecimal {
    type Err = ParseBigDecimalError;

    #[inline]
    fn from_str(s: &str) -> Result<BigDecimal, ParseBigDecimalError> {
        BigDecimal::from_str_radix(s, 10)
    }
}

impl PartialEq for BigDecimal {
    #[inline]
    fn eq(&self, other: &BigDecimal) -> bool {
        match self.int_val.cmp(&other.int_val) {
            Equal => true,
            _ => return false,
        };
        match self.scale.cmp(&other.scale) {
            Equal => true,
            _ => false,
        }
    }
}

impl Default for BigDecimal {
    #[inline]
    fn default() -> BigDecimal {
        Zero::zero()
    }
}

impl Zero for BigDecimal {
    #[inline]
    fn zero() -> BigDecimal {
        BigDecimal::new(BigInt::zero(), 0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.int_val.is_zero()
    }
}

impl One for BigDecimal {
    #[inline]
    fn one() -> BigDecimal {
        BigDecimal::new(BigInt::one(), 0)
    }
}

impl<'a, 'b> Add<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn add(self, other: &BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.clone().set_scale(scale);
        let right = other.clone().set_scale(scale);

        BigDecimal::new(left.int_val + right.int_val, left.scale)
    }
}

impl<'a> Add<BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn add(self, other: BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.clone().set_scale(scale);
        let right = other.set_scale(scale);

        BigDecimal::new(left.int_val + right.int_val, left.scale)
    }
}

impl<'a> Add<&'a BigDecimal> for BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn add(self, other: &BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.set_scale(scale);
        let right = other.clone().set_scale(scale);

        BigDecimal::new(left.int_val + right.int_val, left.scale)
    }
}

impl Add<BigDecimal> for BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn add(self, other: BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.set_scale(scale);
        let right = other.set_scale(scale);

        BigDecimal::new(left.int_val + right.int_val, left.scale)
    }
}

impl<'a, 'b> Sub<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn sub(self, other: &BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.clone().set_scale(scale);
        let right = other.clone().set_scale(scale);

        BigDecimal::new(left.int_val - right.int_val, left.scale)
    }
}

impl<'a> Sub<BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn sub(self, other: BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.clone().set_scale(scale);
        let right = other.set_scale(scale);

        BigDecimal::new(left.int_val - right.int_val, left.scale)
    }
}

impl<'a> Sub<&'a BigDecimal> for BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn sub(self, other: &BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.set_scale(scale);
        let right = other.clone().set_scale(scale);

        BigDecimal::new(left.int_val - right.int_val, left.scale)
    }
}

impl Sub<BigDecimal> for BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn sub(self, other: BigDecimal) -> BigDecimal {
        let scale = max(self.scale, other.scale);
        let left = self.set_scale(scale);
        let right = other.set_scale(scale);

        BigDecimal::new(left.int_val - right.int_val, left.scale)
    }
}

forward_all_binop_to_ref_ref!(impl Mul for BigDecimal, mul);

impl<'a, 'b> Mul<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn mul(self, other: &BigDecimal) -> BigDecimal {
        let scale = self.scale + other.scale;

        BigDecimal::new(self.int_val.clone() * other.int_val.clone(), scale)
    }
}

forward_all_binop_to_ref_ref!(impl Div for BigDecimal, div);

impl<'a, 'b> Div<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn div(self, other: &BigDecimal) -> BigDecimal {
        let scale = self.scale - other.scale;

        BigDecimal::new(self.int_val.clone() / other.int_val.clone(), scale)
    }
}

forward_all_binop_to_ref_ref!(impl Rem for BigDecimal, rem);

impl<'a, 'b> Rem<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn rem(self, _: &BigDecimal) -> BigDecimal {
        // let (_, r) = self.div_rem(other);
        // return r;
        self.clone()
    }
}

impl fmt::Display for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad_integral(true, "", &self.int_val.to_str_radix(10))
    }
}

impl Num for BigDecimal {
    type FromStrRadixErr = ParseBigDecimalError;

    /// Creates and initializes a BigDecimal.
    #[inline]
    fn from_str_radix(s: &str, radix: u32) -> Result<BigDecimal, ParseBigDecimalError> {
        let scale = match s.find('.') {
            Some(i) => (s.len() as i64) - (i as i64) - 1,
            None => 0
        };

        let bi = try!(BigInt::from_str_radix(s, radix));
        Ok(BigDecimal::new(bi, scale as i64))
    }
}

#[cfg(test)]
mod bigdecimal_tests {
    use super::{BigDecimal};
    use bigint::BigInt;
    use traits::{Num};

    /// Assert that an op works for all val/ref combinations
    macro_rules! assert_op {
        ($left:ident $op:tt $right:ident == $expected:expr) => {
            assert_eq!((&$left) $op (&$right), $expected);
            assert_eq!((&$left) $op $right.clone(), $expected);
            assert_eq!($left.clone() $op (&$right), $expected);
            assert_eq!($left.clone() $op $right.clone(), $expected);
        };
    }

    #[test]
    fn test_add() {
        let a = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 2);
        let b = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 3);
        let c = BigDecimal::new(BigInt::from_str_radix("13574", 10).unwrap(), 3);
        // 12.34 + 1.234 = 13.574
        assert_eq!(a + b, c)
    }

    #[test]
    fn test_sub() {
        let a = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 2);
        let b = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 3);
        let c = BigDecimal::new(BigInt::from_str_radix("11106", 10).unwrap(), 3);
        //  12.34
        //-  1.234
        //  11.106
        assert_eq!(a - b, c)
    }

    #[test]
    fn test_mul() {
        let a = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 2);
        let b = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 3);
        let c = BigDecimal::new(BigInt::from_str_radix("1522756", 10).unwrap(), 5);
        //  12.34
        //*  1.234
        //  15.22756
        assert_eq!(a * b, c)
    }

    #[test]
    fn test_div() {
        let a = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 2);
        let b = BigDecimal::new(BigInt::from_str_radix("1233", 10).unwrap(), 3);
        let c = BigDecimal::new(BigInt::from_str_radix("100081103", 10).unwrap(), 1);
        //  12.34
        //*  1.233
        //  10.0081103
        assert_eq!(a / b, c)
    }
}
