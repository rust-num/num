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
//! BigDecimal allows storing any real number to arbitrary precision; which
//! avoids common floating point errors (such as 0.1 + 0.2 ≠ 0.3) at the
//! cost of complexity.
//!
//! Internally, `BigDecimal` uses a `BigInt` object, paired with a 64-bit
//! integer which determines the position of the decimal point. Therefore,
//! the precision *is not* actually arbitrary, but limitied to 2^63 decimal
//! places.
//!
//! Common numerical operations are overloaded, so we can treat them
//! the same way we treat other numbers.
//!
//! It is not recommended to convert a floating point number to a decimal
//! directly, as the floating point representation may be unexpected.
//!

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
use bigint::{BigInt, BigUint, ParseBigIntError, Sign};
use traits::{Num, Zero, One};
// use num_traits::identities::Zero;


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
#[derive(Clone, Debug, Hash)]
#[cfg_attr(feature = "rustc-serialize", derive(RustcEncodable, RustcDecodable))]
pub struct BigDecimal {
    int_val: BigInt,
    scale: i64,
}

impl BigDecimal {
    /// Creates and initializes a `BigDecimal`.
    ///
    #[inline]
    pub fn new(digits: BigInt, scale: i64) -> BigDecimal {
        BigDecimal {
            int_val: digits,
            scale: scale,
        }
    }

    /// Creates and initializes a `BigDecimal`.
    ///
    /// # Examples
    ///
    /// ```
    /// // assert_eq!(BigDecimal::parse_bytes(b"0", 16), BigDecimal::zero());
    /// // assert_eq!(BigDecimal::parse_bytes(b"f", 16), BigDecimal::parse_bytes(b"16", 10));
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
            // println!("raise: {}", raise);
            let exp = (10 as i64).pow(raise as u32);
            let new = BigDecimal::new(self.int_val.clone() * BigInt::from(exp), new_scale);
            // println!("new: {}", new);
            return new;
        }

        // todo implement
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

forward_all_binop_to_ref_ref!(impl Add for BigDecimal, add);

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

forward_all_binop_to_ref_ref!(impl Sub for BigDecimal, sub);

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
        assert!(2 <= radix && radix <= 36, "The radix must be within 2...36");

        // interpret as characters
        let mut chars = s.chars().peekable();

        // get the value of the first character
        let firstchar = match chars.peek() {
            Some(&c) => c,

            // empty string returns zero
            None => return Ok(BigDecimal::zero()),
        };

        // is this a negative number
        let sign = if firstchar == '-' {
            Sign::Minus
        } else {
            Sign::Plus
        };

        // number of leading characters to ignore
        let offset = match firstchar {
            '+' | '-' => 1,
            '0'...'9' | '.' => 0,
            _ => panic!("Unexpected beginning character"),
        };

        // storage buffer
        let mut buff = Vec::with_capacity(s.len());

        // this optional decimal point
        let mut decimal_found = None;

        // keep track of double underscores
        let mut prev_underscore = false;
        for (i, c) in chars.skip(offset).enumerate() {

            if c == '_' && prev_underscore {
                panic!("Malformed string (multiple _ characters)");
            } else if c == '_' {
                prev_underscore = true;
                continue;
            } else {
                prev_underscore = false;
            }

            //
            if c == '.' {
                if decimal_found == None {
                    decimal_found = Some(i);
                    continue;
                } else {
                    panic!("Multiple decimal points");
                }
            }

            // foo
            let d = match c {
                '0'...'9' | 'a'...'f' | 'A'...'F' => c as u8,
                // Some(digit) => buff.push(digit),
                _ => panic!("Unexpected character {:?}", c),
            };

            // println!("{}", c.to_digit());
            buff.push(d);
        }

        let scale = match decimal_found {
            Some(val) => (s.len() as u64) - (val as u64) - 1,
            None => 0,
        };

        let big_uint = match BigUint::parse_bytes(&buff, radix) {
            Some(x) => x,
            None => BigUint::zero(),
        };


        let big_int = BigInt::from_biguint(sign, big_uint);
        return Ok(BigDecimal {
            int_val: big_int,
            scale: scale as i64,
        });
    }
}

#[cfg(test)]
mod bigdecimal_tests {
    use super::BigDecimal;
    use bigint::BigInt;
    use traits::{Num, ToPrimitive};

    use std::str::FromStr;

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
        //  -1.234
        //  11.106
        assert_eq!(a - b, c)
    }

    #[test]
    fn test_mul() {
        let a = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 2);
        let b = BigDecimal::new(BigInt::from_str_radix("1234", 10).unwrap(), 3);
        let c = BigDecimal::new(BigInt::from_str_radix("1522756", 10).unwrap(), 5);
        //  12.34 * 1.234 = 15.22756
        assert_eq!(a * b, c)
    }

    #[test]
    fn test_div() {
        // x / y == z
        for &(x, y, z) in [("1", "2", "0.5")].iter() {

            let a = BigDecimal::from_str_radix(x, 10).unwrap();
            let b = BigDecimal::from_str_radix(y, 10).unwrap();
            // let c = BigDecimal::from_str_radix(z, 10).unwrap();
            //  12.34 ÷ 1.233 = 10.0081103
            // assert_eq!(a / b, c)
        }
    }

    #[test]
    fn test_from_str() {
        let vals = vec![
            ("1331.107", 1331107, 3),
            ("1.0", 10, 1),
            ("0.00123", 123, 5),
            ("-123", -123, 0),
            ("-1230", -1230, 0),
        ];

        for &(source, val, scale) in vals.iter() {
            let x = BigDecimal::from_str(source).unwrap();
            assert_eq!(x.int_val.to_i32().unwrap(), val);
            assert_eq!(x.scale, scale);
        }
    }
}
