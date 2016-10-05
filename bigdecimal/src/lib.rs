// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A Big Decimal
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

use std::fmt;
use integer::Integer;
use std::error::Error;
use std::default::Default;
use std::str::{self, FromStr};
use bigint::{BigInt, ParseBigIntError};
use std::ops::{Add, Div, Mul, Rem, Sub};
use traits::{Num, Zero, One, FromPrimitive};
use std::num::{ParseFloatError, ParseIntError};

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


#[inline]
fn ten_to_the(pow: u64) -> BigInt {
    if pow < 20 {
        BigInt::from(10u64.pow(pow as u32))
    } else {
        let (half, rem) = pow.div_rem(&16);

        let mut x = ten_to_the(half);

        for _ in 0..4 {
            x = &x * &x;
        }

        if rem == 0 { x } else { x * ten_to_the(rem) }
    }
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

    /// Return a new BigDecimal object equivalent to self, with internal
    /// scaling set to the number specified.
    /// If the new_scale is lower than the current value, digits will
    /// be dropped.
    ///
    #[inline]
    pub fn with_scale(&self, new_scale: i64) -> BigDecimal {

        if self.int_val.is_zero() {
            return BigDecimal::new(BigInt::zero(), new_scale);
        }

        if new_scale > self.scale {
            let scale_diff = new_scale - self.scale;
            let int_val = &self.int_val * ten_to_the(scale_diff as u64);
            return BigDecimal::new(int_val, new_scale);

        } else if new_scale < self.scale {
            let scale_diff = self.scale - new_scale;
            let int_val = &self.int_val / ten_to_the(scale_diff as u64);
            return BigDecimal::new(int_val, new_scale);

        } else {
            return self.clone();
        }

    }
}

#[derive(Debug, PartialEq)]
pub enum ParseBigDecimalError {
    ParseDecimal(ParseFloatError),
    ParseInt(ParseIntError),
    ParseBigInt(ParseBigIntError),
    Empty,
    Other(String),
}

impl fmt::Display for ParseBigDecimalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &ParseBigDecimalError::ParseDecimal(ref e) => e.fmt(f),
            &ParseBigDecimalError::ParseInt(ref e) => e.fmt(f),
            &ParseBigDecimalError::ParseBigInt(ref e) => e.fmt(f),
            &ParseBigDecimalError::Empty => "Failed to parse empty string".fmt(f),
            &ParseBigDecimalError::Other(ref reason) => reason[..].fmt(f),
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

impl From<ParseIntError> for ParseBigDecimalError {
    fn from(err: ParseIntError) -> ParseBigDecimalError {
        ParseBigDecimalError::ParseInt(err)
    }
}

impl From<ParseBigIntError> for ParseBigDecimalError {
    fn from(err: ParseBigIntError) -> ParseBigDecimalError {
        ParseBigDecimalError::ParseBigInt(err)
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

        // fix scale and test equality
        if self.scale > rhs.scale {
            let scaled_int_val = &rhs.int_val * ten_to_the((self.scale - rhs.scale) as u64);
            return self.int_val == scaled_int_val;

        } else if self.scale < rhs.scale {
            let scaled_int_val = &self.int_val * ten_to_the((rhs.scale - self.scale) as u64);
            return scaled_int_val == rhs.int_val;

        } else {
            return self.int_val == rhs.int_val;
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
    fn add(self, rhs: &BigDecimal) -> BigDecimal {
        if self.scale < rhs.scale {
            let scaled = self.with_scale(rhs.scale);
            BigDecimal::new(scaled.int_val + &rhs.int_val, rhs.scale)

        } else if self.scale > rhs.scale {
            let scaled = rhs.with_scale(self.scale);
            BigDecimal::new(&self.int_val + scaled.int_val, self.scale)

        } else {
            BigDecimal::new(&self.int_val + &rhs.int_val, self.scale)
        }
    }
}

forward_all_binop_to_ref_ref!(impl Sub for BigDecimal, sub);

impl<'a, 'b> Sub<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn sub(self, rhs: &BigDecimal) -> BigDecimal {
        if self.scale < rhs.scale {
            let scaled = self.with_scale(rhs.scale);
            BigDecimal::new(scaled.int_val - &rhs.int_val, rhs.scale)

        } else if self.scale > rhs.scale {
            let scaled = rhs.with_scale(self.scale);
            BigDecimal::new(&self.int_val - scaled.int_val, self.scale)

        } else {
            BigDecimal::new(&self.int_val - &rhs.int_val, self.scale)
        }
    }
}

forward_all_binop_to_ref_ref!(impl Mul for BigDecimal, mul);

impl<'a, 'b> Mul<&'b BigDecimal> for &'a BigDecimal {
    type Output = BigDecimal;

    #[inline]
    fn mul(self, rhs: &BigDecimal) -> BigDecimal {
        let scale = self.scale + rhs.scale;
        BigDecimal::new(&self.int_val * &rhs.int_val, scale)
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
        if radix != 10 {
            return Err(ParseBigDecimalError::Other(String::from("The radix for decimal MUST be \
                                                                 10")));
        }

        let exp_separator: &[_] = &['e', 'E'];

        // split slice into base and exponent parts
        let (base_part, exponent_value) = match s.find(exp_separator) {
            // exponent defaults to 0 if (e|E) not found
            None => (s, 0),

            // split and parse exponent field
            Some(loc) => {
                // slice up to `loc` and 1 after to skip the 'e' char
                let (base, exp) = (&s[..loc], &s[loc + 1..]);

                // special consideration for rust 1.0.0 which would not
                // parse a leading '+'
                let exp = match exp.chars().next() {
                    Some('+') => &exp[1..],
                    _ => exp,
                };

                (base, try!(i64::from_str(exp)))
            }
        };

        // TEMPORARY: Test for emptiness - remove once BigInt supports similar error
        if base_part == "" {
            return Err(ParseBigDecimalError::Empty);
        }

        // split decimal into a digit string and decimal-point offset
        let (digits, decimal_offset): (String, _) = match base_part.find('.') {
            // No dot! pass directly to BigInt
            None => (base_part.to_string(), 0),

            // decimal point found - necessary copy into new string buffer
            Some(loc) => {
                // split into leading and trailing digits
                let (lead, trail) = (&base_part[..loc], &base_part[loc + 1..]);

                // copy all leading characters into 'digits' string
                let mut digits = String::from(lead);

                // copy all trailing characters after '.' into the digits string
                digits.extend(trail.chars());

                (digits, trail.len() as i64)
            }
        };

        let scale = decimal_offset - exponent_value;
        let big_int = try!(BigInt::from_str_radix(&digits, radix));

        return Ok(BigDecimal::new(big_int, scale));
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
            ("1234e-6", "1234e6", "1234000000.001234"),
            ("18446744073709551616.0", "1", "18446744073709551617"),
            ("184467440737e3380", "0", "184467440737e3380"),
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

    #[test]
    #[should_panic(expected = "InvalidDigit")]
    fn test_bad_string_nan() {
        BigDecimal::from_str("hello").unwrap();
    }
    #[test]
    #[should_panic(expected = "Empty")]
    fn test_bad_string_empty() {
        BigDecimal::from_str("").unwrap();
    }
    #[test]
    #[should_panic(expected = "InvalidDigit")]
    fn test_bad_string_invalid_char() {
        BigDecimal::from_str("12z3.12").unwrap();
    }
    #[test]
    #[should_panic(expected = "InvalidDigit")]
    fn test_bad_string_nan_exponent() {
        BigDecimal::from_str("123.123eg").unwrap();
    }
    #[test]
    #[should_panic(expected = "Empty")]
    fn test_bad_string_empty_exponent() {
        BigDecimal::from_str("123.123E").unwrap();
    }
    #[test]
    #[should_panic(expected = "InvalidDigit")]
    fn test_bad_string_multiple_decimal_points() {
        BigDecimal::from_str("123.12.45").unwrap();
    }
    #[test]
    #[should_panic(expected = "Empty")]
    fn test_bad_string_only_decimal() {
        BigDecimal::from_str(".").unwrap();
    }
    #[test]
    #[should_panic(expected = "Empty")]
    fn test_bad_string_only_decimal_and_exponent() {
        BigDecimal::from_str(".e4").unwrap();
    }
}
