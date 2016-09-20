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
use std::cmp::max;
use bigint::{BigInt, BigUint, ParseBigIntError, Sign};
use traits::{Num, Zero, One, FromPrimitive};
use integer::Integer;


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
    fn eq(&self, rhs: &BigDecimal) -> bool {
        // println!("{}E{} =?= {}E{}",
        //          self.int_val,
        //          self.scale,
        //          rhs.int_val,
        //          rhs.scale);

        // difference in scale between the two decimals
        let scale_diff = (self.scale - rhs.scale).abs() as u32;

        // the scale as a bigint number power of ten
        // e.g. diff == 3 -> scale_bigint == 1000
        let scale_bigint = if scale_diff <= 9 {
            BigInt::from_u64(10u64.pow(scale_diff)).unwrap()
        } else {
            let billion = &BigInt::from_u64(10u64.pow(9)).unwrap();

            let mut scale_diff = scale_diff;
            let mut tmp_bigint = BigInt::one();
            while scale_diff > 9 {
                tmp_bigint = tmp_bigint * billion;
                scale_diff -= 9;
            }

            tmp_bigint * BigInt::from_u64(10u64.pow(scale_diff)).unwrap()
        };

        // fix scale and test equality
        let result = if self.scale > rhs.scale {
            let shifted_int = &rhs.int_val * scale_bigint;
            shifted_int == self.int_val
        } else if self.scale < rhs.scale {
            let shifted_int = &self.int_val * scale_bigint;
            shifted_int == rhs.int_val
        } else {
            self.int_val == rhs.int_val
        };

        return result;
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
    #[allow(non_snake_case)]
    fn div(self, other: &BigDecimal) -> BigDecimal {
        let scale = self.scale - other.scale;
        let ref num = self.int_val;
        let ref den = other.int_val;
        let (quotient, remainder) = num.div_rem(&den);

        // no remainder - quotient is final solution
        if remainder == BigInt::zero() {
            return BigDecimal::new(quotient, scale);
        }

        let BIG_TEN = &BigInt::from_i8(10).unwrap();
        let mut remainder = remainder * BIG_TEN;
        let mut quotient = quotient;

        let MAX_ITERATIONS = 100;
        let mut iteration_count = 0;
        while remainder != BigInt::zero() && iteration_count < MAX_ITERATIONS {
            let (q, r) = remainder.div_rem(&den);
            quotient = quotient * BIG_TEN + q;
            remainder = r * BIG_TEN;

            iteration_count += 1;
        }
        let scale = scale + iteration_count;
        BigDecimal::new(quotient, scale)
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
        let mut chars = s.chars();

        // get the value of the first character
        let firstchar = match chars.next() {
            Some(c) => c,

            // empty string returns zero
            None => return Ok(BigDecimal::zero()),
        };

        // is this a negative number
        let sign = if firstchar == '-' {
            Sign::Minus
        } else {
            Sign::Plus
        };

        // storage buffer
        let mut digit_vec = Vec::with_capacity(s.len());

        // optional decimal point
        let mut decimal_found = if firstchar == '.' {
            Some(0)
        } else if '0' <= firstchar && firstchar <= '9' {
            digit_vec.push(firstchar as u8);
            None
        } else {
            None
        };

        for c in chars.by_ref() {
            // skip underscores
            if c == '_' {
                continue;
            }

            // found decimal point
            if c == '.' {
                if decimal_found == None {
                    decimal_found = Some(digit_vec.len());
                    continue;
                } else {
                    panic!("Multiple decimal points");
                }
            }

            // found exponential marker - stop reading values
            if c == 'e' || c == 'E' {
                break;
            }

            // get the byte value of the character and store in buff
            let d = match c {
                '0'...'9' | 'a'...'f' | 'A'...'F' => c as u8,
                _ => panic!("Unexpected character {:?}", c),
            };

            digit_vec.push(d);
        }

        // determine scale from number of digits stored
        let scale = match decimal_found {
            Some(val) => (digit_vec.len() as i64) - (val as i64),
            None => 0,
        };

        // modify scale by examining remaining characters (exponent)
        let remaining_chars = chars.as_str();

        let scale = if remaining_chars == "" {
            scale
        } else {
            scale - i64::from_str(remaining_chars).unwrap()
        };

        let big_uint = match BigUint::parse_bytes(&digit_vec, radix) {
            Some(x) => x,
            None => BigUint::zero(),
        };

        let big_int = BigInt::from_biguint(sign, big_uint);
        return Ok(BigDecimal {
            int_val: big_int,
            scale: scale,
        });
    }
}

#[cfg(test)]
mod bigdecimal_tests {
    use super::BigDecimal;
    use traits::ToPrimitive;
    use std::str::FromStr;

    #[test]
    fn test_add() {
        let vals = vec![
            ("12.34", "1.234", "13.574"),
            ("12.34", "-1.234", "11.106"),
            ("1234e6", "1234e-6", "1234000000.001234"),
            ("18446744073709551616.0", "1", "18446744073709551617"),
        ];

        for &(x, y, z) in vals.iter() {

            let a = BigDecimal::from_str(x).unwrap();
            let b = BigDecimal::from_str(y).unwrap();
            let c = BigDecimal::from_str(z).unwrap();

            let s = a + b;
            assert_eq!(s, c);
        }
    }

    #[test]
    fn test_sub() {
        let vals = vec![
            ("12.34", "1.234", "11.106"),
            ("12.34", "-1.234", "13.574"),
            ("1234e6", "1234e-6", "1233999999.998766"),
        ];

        for &(x, y, z) in vals.iter() {

            let a = BigDecimal::from_str(x).unwrap();
            let b = BigDecimal::from_str(y).unwrap();
            let c = BigDecimal::from_str(z).unwrap();

            let d = a - b;
            assert_eq!(d, c);
        }
    }

    #[test]
    fn test_mul() {

        let vals = vec![
            ("2", "1", "2"),
            ("12.34", "1.234", "15.22756"),
            ("2e1", "1", "20"),
            ("3", ".333333", "0.999999"),
            ("2389472934723", "209481029831", "500549251119075878721813"),
            ("1e-450", "1e500", ".1e51"),
        ];

        for &(x, y, z) in vals.iter() {

            let a = BigDecimal::from_str(x).unwrap();
            let b = BigDecimal::from_str(y).unwrap();
            let c = BigDecimal::from_str(z).unwrap();

            let p = a * b;
            assert_eq!(p, c);
        }
    }

    #[test]
    fn test_div() {
        let vals = vec![
            ("2", "1", "2"),
            ("2e1", "1", "20"),
            ("1", "2", "0.5"),
            ("1", "2e-2", "5e1"),
            ("5", "4", "1.25"),
            ("5", "4", "125e-2"),
            ("100", "5", "20"),
            ("-50", "5", "-10"),
            ("200", "5", "40."),
            ("1", "3", ".3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333"),
            ("12.34", "1.233", "10.008110300081103000811030008110300081103000811030008110300081103000811030008110300081103000811030008"),
        ];

        for &(x, y, z) in vals.iter() {

            let a = BigDecimal::from_str(x).unwrap();
            let b = BigDecimal::from_str(y).unwrap();
            let c = BigDecimal::from_str(z).unwrap();

            let q = a / b;
            assert_eq!(q, c)
        }
    }

    #[test]
    fn test_equal() {
        let vals = vec![
            ("2", ".2e1"),
            ("0e1", "0.0"),
            ("0e0", "0.0"),
            ("0e-0", "0.0"),
            ("-0901300e-3", "-901.3"),
            ("-0.901300e+3", "-901.3"),
            ("-0e-1", "-0.0"),
            ("2123121e1231", "212.3121e1235"),
        ];
        for &(x, y) in vals.iter() {
            let a = BigDecimal::from_str(x).unwrap();
            let b = BigDecimal::from_str(y).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_not_equal() {
        let vals = vec![
            ("2", ".2e2"),
            ("1e45", "1e-900"),
            ("1e+900", "1e-900"),
        ];
        for &(x, y) in vals.iter() {
            let a = BigDecimal::from_str(x).unwrap();
            let b = BigDecimal::from_str(y).unwrap();
            assert!(a != b, "{} == {}", a, b);
        }
    }

    #[test]
    fn test_from_str() {
        let vals = vec![
            ("1331.107", 1331107, 3),
            ("1.0", 10, 1),
            ("2e1", 2, -1),
            ("0.00123", 123, 5),
            ("-123", -123, 0),
            ("-1230", -1230, 0),
            ("12.3", 123, 1),
            ("123e-1", 123, 1),
            ("1.23e+1", 123, 1),
            ("1.23E+3", 123, -1),
            ("1.23E-8", 123, 10),
            ("-1.23E-10", -123, 12),
        ];

        for &(source, val, scale) in vals.iter() {
            let x = BigDecimal::from_str(source).unwrap();
            assert_eq!(x.int_val.to_i32().unwrap(), val);
            assert_eq!(x.scale, scale);
        }
    }

}
