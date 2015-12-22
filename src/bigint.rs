// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A Big integer (signed version: `BigInt`, unsigned version: `BigUint`).
//!
//! A `BigUint` is represented as a vector of `BigDigit`s.
//! A `BigInt` is a combination of `BigUint` and `Sign`.
//!
//! Common numerical operations are overloaded, so we can treat them
//! the same way we treat other numbers.
//!
//! ## Example
//!
//! ```rust
//! use num::{BigUint, Zero, One};
//! use std::mem::replace;
//!
//! // Calculate large fibonacci numbers.
//! fn fib(n: usize) -> BigUint {
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
//! ```
//!
//! It's easy to generate large random numbers:
//!
//! ```rust
//! extern crate rand;
//! extern crate num;
//! # fn main() {
//! use num::bigint::{ToBigInt, RandBigInt};
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
//! ```

use Integer;

use std::default::Default;
use std::error::Error;
use std::iter::repeat;
use std::num::ParseIntError;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Shl, Shr, Sub};
use std::str::{self, FromStr};
use std::fmt;
use std::cmp::Ordering::{self, Less, Greater, Equal};
use std::{f32, f64};
use std::{u8, i64, u64};

use rand::Rng;

use traits::{ToPrimitive, FromPrimitive};
use traits::Float;

use {Num, Unsigned, CheckedAdd, CheckedSub, CheckedMul, CheckedDiv, Signed, Zero, One};
use self::Sign::{Minus, NoSign, Plus};

/// A `BigDigit` is a `BigUint`'s composing element.
pub type BigDigit = u32;

/// A `DoubleBigDigit` is the internal type used to do the computations.  Its
/// size is the double of the size of `BigDigit`.
pub type DoubleBigDigit = u64;

pub const ZERO_BIG_DIGIT: BigDigit = 0;

#[allow(non_snake_case)]
pub mod big_digit {
    use super::BigDigit;
    use super::DoubleBigDigit;

    // `DoubleBigDigit` size dependent
    pub const BITS: usize = 32;

    pub const BASE: DoubleBigDigit = 1 << BITS;
    const LO_MASK: DoubleBigDigit = (-1i32 as DoubleBigDigit) >> BITS;

    #[inline]
    fn get_hi(n: DoubleBigDigit) -> BigDigit { (n >> BITS) as BigDigit }
    #[inline]
    fn get_lo(n: DoubleBigDigit) -> BigDigit { (n & LO_MASK) as BigDigit }

    /// Split one `DoubleBigDigit` into two `BigDigit`s.
    #[inline]
    pub fn from_doublebigdigit(n: DoubleBigDigit) -> (BigDigit, BigDigit) {
        (get_hi(n), get_lo(n))
    }

    /// Join two `BigDigit`s into one `DoubleBigDigit`
    #[inline]
    pub fn to_doublebigdigit(hi: BigDigit, lo: BigDigit) -> DoubleBigDigit {
        (lo as DoubleBigDigit) | ((hi as DoubleBigDigit) << BITS)
    }
}

/*
 * Generic functions for add/subtract/multiply with carry/borrow:
 */

// Add with carry:
#[inline]
fn adc(a: BigDigit, b: BigDigit, carry: &mut BigDigit) -> BigDigit {
    let (hi, lo) = big_digit::from_doublebigdigit(
        (a as DoubleBigDigit) +
        (b as DoubleBigDigit) +
        (*carry as DoubleBigDigit));

    *carry = hi;
    lo
}

// Subtract with borrow:
#[inline]
fn sbb(a: BigDigit, b: BigDigit, borrow: &mut BigDigit) -> BigDigit {
    let (hi, lo) = big_digit::from_doublebigdigit(
        big_digit::BASE
        + (a as DoubleBigDigit)
        - (b as DoubleBigDigit)
        - (*borrow as DoubleBigDigit));
    /*
       hi * (base) + lo == 1*(base) + ai - bi - borrow
       => ai - bi - borrow < 0 <=> hi == 0
       */
    *borrow = if hi == 0 { 1 } else { 0 };
    lo
}

#[inline]
fn mac_with_carry(a: BigDigit, b: BigDigit, c: BigDigit, carry: &mut BigDigit) -> BigDigit {
    let (hi, lo) = big_digit::from_doublebigdigit(
        (a as DoubleBigDigit) +
        (b as DoubleBigDigit) * (c as DoubleBigDigit) +
        (*carry as DoubleBigDigit));
    *carry = hi;
    lo
}

/// Divide a two digit numerator by a one digit divisor, returns quotient and remainder:
///
/// Note: the caller must ensure that both the quotient and remainder will fit into a single digit.
/// This is _not_ true for an arbitrary numerator/denominator.
///
/// (This function also matches what the x86 divide instruction does).
#[inline]
fn div_wide(hi: BigDigit, lo: BigDigit, divisor: BigDigit) -> (BigDigit, BigDigit) {
    debug_assert!(hi < divisor);

    let lhs = big_digit::to_doublebigdigit(hi, lo);
    let rhs = divisor as DoubleBigDigit;
    ((lhs / rhs) as BigDigit, (lhs % rhs) as BigDigit)
}
/// A big unsigned integer type.
///
/// A `BigUint`-typed value `BigUint { data: vec!(a, b, c) }` represents a number
/// `(a + b * big_digit::BASE + c * big_digit::BASE^2)`.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Hash)]
pub struct BigUint {
    data: Vec<BigDigit>
}

impl PartialEq for BigUint {
    #[inline]
    fn eq(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Equal => true, _ => false }
    }
}
impl Eq for BigUint {}

impl PartialOrd for BigUint {
    #[inline]
    fn partial_cmp(&self, other: &BigUint) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn cmp_slice(a: &[BigDigit], b: &[BigDigit]) -> Ordering {
    debug_assert!(a.last() != Some(&0));
    debug_assert!(b.last() != Some(&0));

    let (a_len, b_len) = (a.len(), b.len());
    if a_len < b_len { return Less; }
    if a_len > b_len { return Greater;  }

    for (&ai, &bi) in a.iter().rev().zip(b.iter().rev()) {
        if ai < bi { return Less; }
        if ai > bi { return Greater; }
    }
    return Equal;
}

impl Ord for BigUint {
    #[inline]
    fn cmp(&self, other: &BigUint) -> Ordering {
        cmp_slice(&self.data[..], &other.data[..])
    }
}

impl Default for BigUint {
    #[inline]
    fn default() -> BigUint { Zero::zero() }
}

impl fmt::Display for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_str_radix(10))
    }
}

impl FromStr for BigUint {
    type Err = ParseBigIntError;

    #[inline]
    fn from_str(s: &str) -> Result<BigUint, ParseBigIntError> {
        BigUint::from_str_radix(s, 10)
    }
}

// Read bitwise digits that evenly divide BigDigit
fn from_bitwise_digits_le(v: &[u8], bits: usize) -> BigUint {
    debug_assert!(!v.is_empty() && bits <= 8 && big_digit::BITS % bits == 0);
    debug_assert!(v.iter().all(|&c| (c as BigDigit) < (1 << bits)));

    let digits_per_big_digit = big_digit::BITS / bits;

    let data = v.chunks(digits_per_big_digit).map(|chunk| {
        chunk.iter().rev().fold(0u32, |acc, &c| (acc << bits) | c as BigDigit)
    }).collect();

    BigUint::new(data)
}

// Read bitwise digits that don't evenly divide BigDigit
fn from_inexact_bitwise_digits_le(v: &[u8], bits: usize) -> BigUint {
    debug_assert!(!v.is_empty() && bits <= 8 && big_digit::BITS % bits != 0);
    debug_assert!(v.iter().all(|&c| (c as BigDigit) < (1 << bits)));

    let big_digits = (v.len() * bits + big_digit::BITS - 1) / big_digit::BITS;
    let mut data = Vec::with_capacity(big_digits);

    let mut d = 0;
    let mut dbits = 0;
    for &c in v {
        d |= (c as DoubleBigDigit) << dbits;
        dbits += bits;
        if dbits >= big_digit::BITS {
            let (hi, lo) = big_digit::from_doublebigdigit(d);
            data.push(lo);
            d = hi as DoubleBigDigit;
            dbits -= big_digit::BITS;
        }
    }

    if dbits > 0 {
        debug_assert!(dbits < big_digit::BITS);
        data.push(d as BigDigit);
    }

    BigUint::new(data)
}

// Read little-endian radix digits
fn from_radix_digits_be(v: &[u8], radix: u32) -> BigUint {
    debug_assert!(!v.is_empty() && !radix.is_power_of_two());
    debug_assert!(v.iter().all(|&c| (c as u32) < radix));

    // Estimate how big the result will be, so we can pre-allocate it.
    let bits = (radix as f64).log2() * v.len() as f64;
    let big_digits = (bits / big_digit::BITS as f64).ceil();
    let mut data = Vec::with_capacity(big_digits as usize);

    let (base, power) = get_radix_base(radix);
    debug_assert!(base < (1 << 32));
    let base = base as BigDigit;

    let r = v.len() % power;
    let i = if r == 0 { power } else { r };
    let (head, tail) = v.split_at(i);

    let first = head.iter().fold(0, |acc, &d| acc * radix + d as BigDigit);
    data.push(first);

    debug_assert!(tail.len() % power == 0);
    for chunk in tail.chunks(power) {
        if data.last() != Some(&0) {
            data.push(0);
        }

        let mut carry = 0;
        for d in data.iter_mut() {
            *d = mac_with_carry(0, *d, base, &mut carry);
        }
        debug_assert!(carry == 0);

        let n = chunk.iter().fold(0, |acc, &d| acc * radix + d as BigDigit);
        add2(&mut data, &[n]);
    }

    BigUint::new(data)
}

impl Num for BigUint {
    type FromStrRadixErr = ParseBigIntError;

    /// Creates and initializes a `BigUint`.
    fn from_str_radix(s: &str, radix: u32) -> Result<BigUint, ParseBigIntError> {
        assert!(2 <= radix && radix <= 36, "The radix must be within 2...36");
        if s.is_empty() {
            // create ParseIntError::Empty
            let e = u64::from_str_radix(s, radix).unwrap_err();
            return Err(e.into());
        }

        // First normalize all characters to plain digit values
        let mut v = Vec::with_capacity(s.len());
        for b in s.bytes() {
            let d = match b {
                b'0' ... b'9' => b - b'0',
                b'a' ... b'z' => b - b'a' + 10,
                b'A' ... b'Z' => b - b'A' + 10,
                _ => u8::MAX,
            };
            if d < radix as u8 {
                v.push(d);
            } else {
                // create ParseIntError::InvalidDigit
                let e = u64::from_str_radix(&s[v.len()..], radix).unwrap_err();
                return Err(e.into());
            }
        }

        let res = if radix.is_power_of_two() {
            // Powers of two can use bitwise masks and shifting instead of multiplication
            let bits = radix.trailing_zeros() as usize;
            v.reverse();
            if big_digit::BITS % bits == 0 {
                from_bitwise_digits_le(&v, bits)
            } else {
                from_inexact_bitwise_digits_le(&v, bits)
            }
        } else {
            from_radix_digits_be(&v, radix)
        };
        Ok(res)
    }
}

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

macro_rules! forward_val_val_binop_commutative {
    (impl $imp:ident for $res:ty, $method:ident) => {
        impl $imp<$res> for $res {
            type Output = $res;

            #[inline]
            fn $method(self, other: $res) -> $res {
                // forward to val-ref, with the larger capacity as val
                if self.data.capacity() >= other.data.capacity() {
                    $imp::$method(self, &other)
                } else {
                    $imp::$method(other, &self)
                }
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

macro_rules! forward_ref_val_binop_commutative {
    (impl $imp:ident for $res:ty, $method:ident) => {
        impl<'a> $imp<$res> for &'a $res {
            type Output = $res;

            #[inline]
            fn $method(self, other: $res) -> $res {
                // reverse, forward to val-ref
                $imp::$method(other, self)
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

macro_rules! forward_ref_ref_binop_commutative {
    (impl $imp:ident for $res:ty, $method:ident) => {
        impl<'a, 'b> $imp<&'b $res> for &'a $res {
            type Output = $res;

            #[inline]
            fn $method(self, other: &$res) -> $res {
                // forward to val-ref, choosing the larger to clone
                if self.data.len() >= other.data.len() {
                    $imp::$method(self.clone(), other)
                } else {
                    $imp::$method(other.clone(), self)
                }
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

// Forward everything to val-ref, so LHS storage can be reused
macro_rules! forward_all_binop_to_val_ref {
    (impl $imp:ident for $res:ty, $method:ident) => {
        forward_val_val_binop!(impl $imp for $res, $method);
        forward_ref_val_binop!(impl $imp for $res, $method);
        forward_ref_ref_binop!(impl $imp for $res, $method);
    };
}

// Forward everything to val-ref, commutatively, so either LHS or RHS storage can be reused
macro_rules! forward_all_binop_to_val_ref_commutative {
    (impl $imp:ident for $res:ty, $method:ident) => {
        forward_val_val_binop_commutative!(impl $imp for $res, $method);
        forward_ref_val_binop_commutative!(impl $imp for $res, $method);
        forward_ref_ref_binop_commutative!(impl $imp for $res, $method);
    };
}

forward_all_binop_to_val_ref_commutative!(impl BitAnd for BigUint, bitand);

impl<'a> BitAnd<&'a BigUint> for BigUint {
    type Output = BigUint;

    #[inline]
    fn bitand(self, other: &BigUint) -> BigUint {
        let mut data = self.data;
        for (ai, &bi) in data.iter_mut().zip(other.data.iter()) {
            *ai &= bi;
        }
        data.truncate(other.data.len());
        BigUint::new(data)
    }
}

forward_all_binop_to_val_ref_commutative!(impl BitOr for BigUint, bitor);

impl<'a> BitOr<&'a BigUint> for BigUint {
    type Output = BigUint;

    fn bitor(self, other: &BigUint) -> BigUint {
        let mut data = self.data;
        for (ai, &bi) in data.iter_mut().zip(other.data.iter()) {
            *ai |= bi;
        }
        if other.data.len() > data.len() {
            let extra = &other.data[data.len()..];
            data.extend(extra.iter().cloned());
        }
        BigUint::new(data)
    }
}

forward_all_binop_to_val_ref_commutative!(impl BitXor for BigUint, bitxor);

impl<'a> BitXor<&'a BigUint> for BigUint {
    type Output = BigUint;

    fn bitxor(self, other: &BigUint) -> BigUint {
        let mut data = self.data;
        for (ai, &bi) in data.iter_mut().zip(other.data.iter()) {
            *ai ^= bi;
        }
        if other.data.len() > data.len() {
            let extra = &other.data[data.len()..];
            data.extend(extra.iter().cloned());
        }
        BigUint::new(data)
    }
}

impl Shl<usize> for BigUint {
    type Output = BigUint;

    #[inline]
    fn shl(self, rhs: usize) -> BigUint { (&self) << rhs }
}

impl<'a> Shl<usize> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn shl(self, rhs: usize) -> BigUint {
        let n_unit = rhs / big_digit::BITS;
        let n_bits = rhs % big_digit::BITS;
        self.shl_unit(n_unit).shl_bits(n_bits)
    }
}

impl Shr<usize> for BigUint {
    type Output = BigUint;

    #[inline]
    fn shr(self, rhs: usize) -> BigUint { (&self) >> rhs }
}

impl<'a> Shr<usize> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn shr(self, rhs: usize) -> BigUint {
        let n_unit = rhs / big_digit::BITS;
        let n_bits = rhs % big_digit::BITS;
        self.shr_unit(n_unit).shr_bits(n_bits)
    }
}

impl Zero for BigUint {
    #[inline]
    fn zero() -> BigUint { BigUint::new(Vec::new()) }

    #[inline]
    fn is_zero(&self) -> bool { self.data.is_empty() }
}

impl One for BigUint {
    #[inline]
    fn one() -> BigUint { BigUint::new(vec!(1)) }
}

impl Unsigned for BigUint {}

forward_all_binop_to_val_ref_commutative!(impl Add for BigUint, add);

// Only for the Add impl:
#[must_use]
#[inline]
fn __add2(a: &mut [BigDigit], b: &[BigDigit]) -> BigDigit {
    let mut b_iter = b.iter();
    let mut carry = 0;

    for ai in a.iter_mut() {
        if let Some(bi) = b_iter.next() {
            *ai = adc(*ai, *bi, &mut carry);
        } else if carry != 0 {
            *ai = adc(*ai, 0, &mut carry);
        } else {
            break;
        }
    }

    debug_assert!(b_iter.next() == None);
    carry
}

/// /Two argument addition of raw slices:
/// a += b
///
/// The caller _must_ ensure that a is big enough to store the result - typically this means
/// resizing a to max(a.len(), b.len()) + 1, to fit a possible carry.
fn add2(a: &mut [BigDigit], b: &[BigDigit]) {
    let carry = __add2(a, b);

    debug_assert!(carry == 0);
}

/*
 * We'd really prefer to avoid using add2/sub2 directly as much as possible - since they make the
 * caller entirely responsible for ensuring a's vector is big enough, and that the result is
 * normalized, they're rather error prone and verbose:
 *
 * We could implement the Add and Sub traits for BigUint + BigDigit slices, like below - this works
 * great, except that then it becomes the module's public interface, which we probably don't want:
 *
 * I'm keeping the code commented out, because I think this is worth revisiting:

impl<'a> Add<&'a [BigDigit]> for BigUint {
    type Output = BigUint;

    fn add(mut self, other: &[BigDigit]) -> BigUint {
        if self.data.len() < other.len() {
            let extra = other.len() - self.data.len();
            self.data.extend(repeat(0).take(extra));
        }

        let carry = __add2(&mut self.data[..], other);
        if carry != 0 {
            self.data.push(carry);
        }

        self
    }
}
 */

impl<'a> Add<&'a BigUint> for BigUint {
    type Output = BigUint;

    fn add(mut self, other: &BigUint) -> BigUint {
        if self.data.len() < other.data.len() {
            let extra = other.data.len() - self.data.len();
            self.data.extend(repeat(0).take(extra));
        }

        let carry = __add2(&mut self.data[..], &other.data[..]);
        if carry != 0 {
            self.data.push(carry);
        }

        self
    }
}

forward_all_binop_to_val_ref!(impl Sub for BigUint, sub);

fn sub2(a: &mut [BigDigit], b: &[BigDigit]) {
    let mut b_iter = b.iter();
    let mut borrow = 0;

    for ai in a.iter_mut() {
        if let Some(bi) = b_iter.next() {
            *ai = sbb(*ai, *bi, &mut borrow);
        } else if borrow != 0 {
            *ai = sbb(*ai, 0, &mut borrow);
        } else {
            break;
        }
    }

    /* note: we're _required_ to fail on underflow */
    assert!(borrow == 0 && b_iter.all(|x| *x == 0),
            "Cannot subtract b from a because b is larger than a.");
}

impl<'a> Sub<&'a BigUint> for BigUint {
    type Output = BigUint;

    fn sub(mut self, other: &BigUint) -> BigUint {
        sub2(&mut self.data[..], &other.data[..]);
        self.normalize()
    }
}

fn sub_sign(a: &[BigDigit], b: &[BigDigit]) -> BigInt {
    // Normalize:
    let a = &a[..a.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1)];
    let b = &b[..b.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1)];

    match cmp_slice(a, b) {
        Greater => {
            let mut ret = BigUint::from_slice(a);
            sub2(&mut ret.data[..], b);
            BigInt::from_biguint(Plus, ret.normalize())
        },
        Less    => {
            let mut ret = BigUint::from_slice(b);
            sub2(&mut ret.data[..], a);
            BigInt::from_biguint(Minus, ret.normalize())
        },
        _       => Zero::zero(),
    }
}

forward_all_binop_to_ref_ref!(impl Mul for BigUint, mul);

/// Three argument multiply accumulate:
/// acc += b * c
fn mac_digit(acc: &mut [BigDigit], b: &[BigDigit], c: BigDigit) {
    if c == 0 { return; }

    let mut b_iter = b.iter();
    let mut carry = 0;

    for ai in acc.iter_mut() {
        if let Some(bi) = b_iter.next() {
            *ai = mac_with_carry(*ai, *bi, c, &mut carry);
        } else if carry != 0 {
            *ai = mac_with_carry(*ai, 0, c, &mut carry);
        } else {
            break;
        }
    }

    assert!(carry == 0);
}

/// Three argument multiply accumulate:
/// acc += b * c
fn mac3(acc: &mut [BigDigit], b: &[BigDigit], c: &[BigDigit]) {
    let (x, y) = if b.len() < c.len() { (b, c) } else { (c, b) };

    /*
     * Karatsuba multiplication is slower than long multiplication for small x and y:
     */
    if x.len() <= 4 {
        for (i, xi) in x.iter().enumerate() {
            mac_digit(&mut acc[i..], y, *xi);
        }
    } else {
        /*
         * Karatsuba multiplication:
         *
         * The idea is that we break x and y up into two smaller numbers that each have about half
         * as many digits, like so (note that multiplying by b is just a shift):
         *
         * x = x0 + x1 * b
         * y = y0 + y1 * b
         *
         * With some algebra, we can compute x * y with three smaller products, where the inputs to
         * each of the smaller products have only about half as many digits as x and y:
         *
         * x * y = (x0 + x1 * b) * (y0 + y1 * b)
         *
         * x * y = x0 * y0
         *       + x0 * y1 * b
         *       + x1 * y0 * b
         *       + x1 * y1 * b^2
         *
         * Let p0 = x0 * y0 and p2 = x1 * y1:
         *
         * x * y = p0
         *       + (x0 * y1 + x1 * p0) * b
         *       + p2 * b^2
         *
         * The real trick is that middle term:
         *
         *         x0 * y1 + x1 * y0
         *
         *       = x0 * y1 + x1 * y0 - p0 + p0 - p2 + p2
         *
         *       = x0 * y1 + x1 * y0 - x0 * y0 - x1 * y1 + p0 + p2
         *
         * Now we complete the square:
         *
         *       = -(x0 * y0 - x0 * y1 - x1 * y0 + x1 * y1) + p0 + p2
         *
         *       = -((x1 - x0) * (y1 - y0)) + p0 + p2
         *
         * Let p1 = (x1 - x0) * (y1 - y0), and substitute back into our original formula:
         *
         * x * y = p0
         *       + (p0 + p2 - p1) * b
         *       + p2 * b^2
         *
         * Where the three intermediate products are:
         *
         * p0 = x0 * y0
         * p1 = (x1 - x0) * (y1 - y0)
         * p2 = x1 * y1
         *
         * In doing the computation, we take great care to avoid unnecessary temporary variables
         * (since creating a BigUint requires a heap allocation): thus, we rearrange the formula a
         * bit so we can use the same temporary variable for all the intermediate products:
         *
         * x * y = p2 * b^2 + p2 * b
         *       + p0 * b + p0
         *       - p1 * b
         *
         * The other trick we use is instead of doing explicit shifts, we slice acc at the
         * appropriate offset when doing the add.
         */

        /*
         * When x is smaller than y, it's significantly faster to pick b such that x is split in
         * half, not y:
         */
        let b = x.len() / 2;
        let (x0, x1) = x.split_at(b);
        let (y0, y1) = y.split_at(b);

        /* We reuse the same BigUint for all the intermediate multiplies: */

        let len = y.len() + 1;
        let mut p = BigUint { data: vec![0; len] };

        // p2 = x1 * y1
        mac3(&mut p.data[..], x1, y1);

        // Not required, but the adds go faster if we drop any unneeded 0s from the end:
        p = p.normalize();

        add2(&mut acc[b..],        &p.data[..]);
        add2(&mut acc[b * 2..],    &p.data[..]);

        // Zero out p before the next multiply:
        p.data.truncate(0);
        p.data.extend(repeat(0).take(len));

        // p0 = x0 * y0
        mac3(&mut p.data[..], x0, y0);
        p = p.normalize();

        add2(&mut acc[..],                &p.data[..]);
        add2(&mut acc[b..],        &p.data[..]);

        // p1 = (x1 - x0) * (y1 - y0)
        // We do this one last, since it may be negative and acc can't ever be negative:
        let j0 = sub_sign(x1, x0);
        let j1 = sub_sign(y1, y0);

        match j0.sign * j1.sign {
            Plus    => {
                p.data.truncate(0);
                p.data.extend(repeat(0).take(len));

                mac3(&mut p.data[..], &j0.data.data[..], &j1.data.data[..]);
                p = p.normalize();

                sub2(&mut acc[b..], &p.data[..]);
            },
            Minus   => {
                mac3(&mut acc[b..], &j0.data.data[..], &j1.data.data[..]);
            },
            NoSign  => (),
        }
    }
}

fn mul3(x: &[BigDigit], y: &[BigDigit]) -> BigUint {
    let len = x.len() + y.len() + 1;
    let mut prod = BigUint { data: vec![0; len] };

    mac3(&mut prod.data[..], x, y);
    prod.normalize()
}

impl<'a, 'b> Mul<&'b BigUint> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn mul(self, other: &BigUint) -> BigUint {
        mul3(&self.data[..], &other.data[..])
    }
}

fn div_rem_digit(mut a: BigUint, b: BigDigit) -> (BigUint, BigDigit) {
    let mut rem = 0;

    for d in a.data.iter_mut().rev() {
        let (q, r) = div_wide(rem, *d, b);
        *d = q;
        rem = r;
    }

    (a.normalize(), rem)
}

forward_all_binop_to_ref_ref!(impl Div for BigUint, div);

impl<'a, 'b> Div<&'b BigUint> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn div(self, other: &BigUint) -> BigUint {
        let (q, _) = self.div_rem(other);
        return q;
    }
}

forward_all_binop_to_ref_ref!(impl Rem for BigUint, rem);

impl<'a, 'b> Rem<&'b BigUint> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn rem(self, other: &BigUint) -> BigUint {
        let (_, r) = self.div_rem(other);
        return r;
    }
}

impl Neg for BigUint {
    type Output = BigUint;

    #[inline]
    fn neg(self) -> BigUint { panic!() }
}

impl<'a> Neg for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn neg(self) -> BigUint { panic!() }
}

impl CheckedAdd for BigUint {
    #[inline]
    fn checked_add(&self, v: &BigUint) -> Option<BigUint> {
        return Some(self.add(v));
    }
}

impl CheckedSub for BigUint {
    #[inline]
    fn checked_sub(&self, v: &BigUint) -> Option<BigUint> {
        match self.cmp(v) {
            Less => None,
            Equal => Some(Zero::zero()),
            Greater => Some(self.sub(v)),
        }
    }
}

impl CheckedMul for BigUint {
    #[inline]
    fn checked_mul(&self, v: &BigUint) -> Option<BigUint> {
        return Some(self.mul(v));
    }
}

impl CheckedDiv for BigUint {
    #[inline]
    fn checked_div(&self, v: &BigUint) -> Option<BigUint> {
        if v.is_zero() {
            return None;
        }
        return Some(self.div(v));
    }
}

impl Integer for BigUint {
    #[inline]
    fn div_rem(&self, other: &BigUint) -> (BigUint, BigUint) {
        self.div_mod_floor(other)
    }

    #[inline]
    fn div_floor(&self, other: &BigUint) -> BigUint {
        let (d, _) = self.div_mod_floor(other);
        return d;
    }

    #[inline]
    fn mod_floor(&self, other: &BigUint) -> BigUint {
        let (_, m) = self.div_mod_floor(other);
        return m;
    }

    fn div_mod_floor(&self, other: &BigUint) -> (BigUint, BigUint) {
        if other.is_zero() { panic!() }
        if self.is_zero() { return (Zero::zero(), Zero::zero()); }
        if *other == One::one() { return (self.clone(), Zero::zero()); }

        /* Required or the q_len calculation below can underflow: */
        match self.cmp(other) {
            Less    => return (Zero::zero(), self.clone()),
            Equal   => return (One::one(), Zero::zero()),
            Greater => {} // Do nothing
        }

        /*
         * This algorithm is from Knuth, TAOCP vol 2 section 4.3, algorithm D:
         *
         * First, normalize the arguments so the highest bit in the highest digit of the divisor is
         * set: the main loop uses the highest digit of the divisor for generating guesses, so we
         * want it to be the largest number we can efficiently divide by.
         */
        let shift = other.data.last().unwrap().leading_zeros() as usize;
        let mut a = self << shift;
        let b     = other << shift;

        /*
         * The algorithm works by incrementally calculating "guesses", q0, for part of the
         * remainder. Once we have any number q0 such that q0 * b <= a, we can set
         *
         *     q += q0
         *     a -= q0 * b
         *
         * and then iterate until a < b. Then, (q, a) will be our desired quotient and remainder.
         *
         * q0, our guess, is calculated by dividing the last few digits of a by the last digit of b
         * - this should give us a guess that is "close" to the actual quotient, but is possibly
         * greater than the actual quotient. If q0 * b > a, we simply use iterated subtraction
         * until we have a guess such that q0 & b <= a.
         */

        let bn = *b.data.last().unwrap();
        let q_len = a.data.len() - b.data.len() + 1;
        let mut q = BigUint { data: vec![0; q_len] };

        /*
         * We reuse the same temporary to avoid hitting the allocator in our inner loop - this is
         * sized to hold a0 (in the common case; if a particular digit of the quotient is zero a0
         * can be bigger).
         */
        let mut tmp = BigUint { data: Vec::with_capacity(2) };

        for j in (0..q_len).rev() {
            /*
             * When calculating our next guess q0, we don't need to consider the digits below j
             * + b.data.len() - 1: we're guessing digit j of the quotient (i.e. q0 << j) from
             * digit bn of the divisor (i.e. bn << (b.data.len() - 1) - so the product of those
             * two numbers will be zero in all digits up to (j + b.data.len() - 1).
             */
            let offset = j + b.data.len() - 1;
            if offset >= a.data.len() {
                continue;
            }

            /* just avoiding a heap allocation: */
            let mut a0 = tmp;
            a0.data.truncate(0);
            a0.data.extend(a.data[offset..].iter().cloned());

            /*
             * q0 << j * big_digit::BITS is our actual quotient estimate - we do the shifts
             * implicitly at the end, when adding and subtracting to a and q. Not only do we
             * save the cost of the shifts, the rest of the arithmetic gets to work with
             * smaller numbers.
             */
            let (mut q0, _) = div_rem_digit(a0, bn);
            let mut prod = &b * &q0;

            while cmp_slice(&prod.data[..], &a.data[j..]) == Greater {
                let one: BigUint = One::one();
                q0 = q0 - one;
                prod = prod - &b;
            }

            add2(&mut q.data[j..], &q0.data[..]);
            sub2(&mut a.data[j..], &prod.data[..]);
            a = a.normalize();

            tmp = q0;
        }

        debug_assert!(a < b);

        (q.normalize(), a >> shift)
    }

    /// Calculates the Greatest Common Divisor (GCD) of the number and `other`.
    ///
    /// The result is always positive.
    #[inline]
    fn gcd(&self, other: &BigUint) -> BigUint {
        // Use Euclid's algorithm
        let mut m = (*self).clone();
        let mut n = (*other).clone();
        while !m.is_zero() {
            let temp = m;
            m = n % &temp;
            n = temp;
        }
        return n;
    }

    /// Calculates the Lowest Common Multiple (LCM) of the number and `other`.
    #[inline]
    fn lcm(&self, other: &BigUint) -> BigUint { ((self * other) / self.gcd(other)) }

    /// Deprecated, use `is_multiple_of` instead.
    #[inline]
    fn divides(&self, other: &BigUint) -> bool { self.is_multiple_of(other) }

    /// Returns `true` if the number is a multiple of `other`.
    #[inline]
    fn is_multiple_of(&self, other: &BigUint) -> bool { (self % other).is_zero() }

    /// Returns `true` if the number is divisible by `2`.
    #[inline]
    fn is_even(&self) -> bool {
        // Considering only the last digit.
        match self.data.first() {
            Some(x) => x.is_even(),
            None => true
        }
    }

    /// Returns `true` if the number is not divisible by `2`.
    #[inline]
    fn is_odd(&self) -> bool { !self.is_even() }
}

impl ToPrimitive for BigUint {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.to_u64().and_then(|n| {
            // If top bit of u64 is set, it's too large to convert to i64.
            if n >> 63 == 0 {
                Some(n as i64)
            } else {
                None
            }
        })
    }

    // `DoubleBigDigit` size dependent
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        match self.data.len() {
            0 => Some(0),
            1 => Some(self.data[0] as u64),
            2 => Some(big_digit::to_doublebigdigit(self.data[1], self.data[0])
                      as u64),
            _ => None
        }
    }

    // `DoubleBigDigit` size dependent
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        match self.data.len() {
            0 => Some(f32::zero()),
            1 => Some(self.data[0] as f32),
            len => {
                // prevent overflow of exponant
                if len > (f32::MAX_EXP as usize) / big_digit::BITS {
                    None
                } else {
                    let exponant = (len - 2) * big_digit::BITS;
                    // we need 25 significant digits, 24 to be stored and 1 for rounding
                    // this gives at least 33 significant digits
                    let mantissa = big_digit::to_doublebigdigit(self.data[len - 1], self.data[len - 2]);
                    // this cast handles rounding
                    let ret = (mantissa as f32) * 2.0.powi(exponant as i32);
                    if ret.is_infinite() {
                        None
                    } else {
                        Some(ret)
                    }
                }
            }
        }
    }

    // `DoubleBigDigit` size dependent
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        match self.data.len() {
            0 => Some(f64::zero()),
            1 => Some(self.data[0] as f64),
            2 => Some(big_digit::to_doublebigdigit(self.data[1], self.data[0]) as f64),
            len => {
                // this will prevent any overflow of exponant
                if len > (f64::MAX_EXP as usize) / big_digit::BITS {
                    None
                } else {
                    let mut exponant = (len - 2) * big_digit::BITS;
                    let mut mantissa = big_digit::to_doublebigdigit(self.data[len - 1], self.data[len - 2]);
                    // we need at least 54 significant bit digits, 53 to be stored and 1 for rounding
                    // so we take enough from the next BigDigit to make it up if needed
                    let needed = (f64::MANTISSA_DIGITS as usize) + 1;
                    let bits = (2 * big_digit::BITS) - (mantissa.leading_zeros() as usize);
                    if needed > bits {
                        let diff = needed - bits;
                        mantissa <<= diff;
                        exponant -= diff;
                        let mut x = self.data[len - 3];
                        x >>= big_digit::BITS - diff;
                        mantissa |= x as u64;
                    }
                    // this cast handles rounding
                    let ret = (mantissa as f64) * 2.0.powi(exponant as i32);
                    if ret.is_infinite() {
                        None
                    } else {
                        Some(ret)
                    }
                }
            }
        }
    }
}

impl FromPrimitive for BigUint {
    #[inline]
    fn from_i64(n: i64) -> Option<BigUint> {
        if n >= 0 {
            Some(BigUint::from(n as u64))
        } else {
            None
        }
    }

    #[inline]
    fn from_u64(n: u64) -> Option<BigUint> {
        Some(BigUint::from(n))
    }

    #[inline]
    fn from_f32(n: f32) -> Option<BigUint> {
        BigUint::from_f64(n as f64)
    }

    #[inline]
    fn from_f64(mut n: f64) -> Option<BigUint> {
        // handle NAN, INFINITY, NEG_INFINITY
        if !n.is_finite() {
            return None;
        }

        // match the rounding of casting from float to int
        n = n.trunc();

        // handle 0.x, -0.x
        if n.is_zero() {
            return Some(BigUint::zero());
        }

        let (mantissa, exponent, sign) = Float::integer_decode(n);

        if sign == -1 {
            return None;
        }

        let mut ret = BigUint::from(mantissa);
        if exponent > 0 {
            ret = ret << exponent as usize;
        } else if exponent < 0 {
            ret = ret >> (-exponent) as usize;
        }
        Some(ret)
    }
}

impl From<u64> for BigUint {
    // `DoubleBigDigit` size dependent
    #[inline]
    fn from(n: u64) -> Self {
        match big_digit::from_doublebigdigit(n) {
            (0, 0) => BigUint::zero(),
            (0, n0) => BigUint { data: vec![n0] },
            (n1, n0) => BigUint { data: vec![n0, n1] },
        }
    }
}

macro_rules! impl_biguint_from_uint {
    ($T:ty) => {
        impl From<$T> for BigUint {
            #[inline]
            fn from(n: $T) -> Self {
                BigUint::from(n as u64)
            }
        }
    }
}

impl_biguint_from_uint!(u8);
impl_biguint_from_uint!(u16);
impl_biguint_from_uint!(u32);
impl_biguint_from_uint!(usize);

/// A generic trait for converting a value to a `BigUint`.
pub trait ToBigUint {
    /// Converts the value of `self` to a `BigUint`.
    fn to_biguint(&self) -> Option<BigUint>;
}

impl ToBigUint for BigInt {
    #[inline]
    fn to_biguint(&self) -> Option<BigUint> {
        if self.sign == Plus {
            Some(self.data.clone())
        } else if self.sign == NoSign {
            Some(Zero::zero())
        } else {
            None
        }
    }
}

impl ToBigUint for BigUint {
    #[inline]
    fn to_biguint(&self) -> Option<BigUint> {
        Some(self.clone())
    }
}

macro_rules! impl_to_biguint {
    ($T:ty, $from_ty:path) => {
        impl ToBigUint for $T {
            #[inline]
            fn to_biguint(&self) -> Option<BigUint> {
                $from_ty(*self)
            }
        }
    }
}

impl_to_biguint!(isize,  FromPrimitive::from_isize);
impl_to_biguint!(i8,   FromPrimitive::from_i8);
impl_to_biguint!(i16,  FromPrimitive::from_i16);
impl_to_biguint!(i32,  FromPrimitive::from_i32);
impl_to_biguint!(i64,  FromPrimitive::from_i64);
impl_to_biguint!(usize, FromPrimitive::from_usize);
impl_to_biguint!(u8,   FromPrimitive::from_u8);
impl_to_biguint!(u16,  FromPrimitive::from_u16);
impl_to_biguint!(u32,  FromPrimitive::from_u32);
impl_to_biguint!(u64,  FromPrimitive::from_u64);
impl_to_biguint!(f32,  FromPrimitive::from_f32);
impl_to_biguint!(f64,  FromPrimitive::from_f64);

// Extract bitwise digits that evenly divide BigDigit
fn to_bitwise_digits_le(u: &BigUint, bits: usize) -> Vec<u8> {
    debug_assert!(!u.is_zero() && bits <= 8 && big_digit::BITS % bits == 0);

    let last_i = u.data.len() - 1;
    let mask: BigDigit = (1 << bits) - 1;
    let digits_per_big_digit = big_digit::BITS / bits;
    let digits = (u.bits() + bits - 1) / bits;
    let mut res = Vec::with_capacity(digits);

    for mut r in u.data[..last_i].iter().cloned() {
        for _ in 0..digits_per_big_digit {
            res.push((r & mask) as u8);
            r >>= bits;
        }
    }

    let mut r = u.data[last_i];
    while r != 0 {
        res.push((r & mask) as u8);
        r >>= bits;
    }

    res
}

// Extract bitwise digits that don't evenly divide BigDigit
fn to_inexact_bitwise_digits_le(u: &BigUint, bits: usize) -> Vec<u8> {
    debug_assert!(!u.is_zero() && bits <= 8 && big_digit::BITS % bits != 0);

    let last_i = u.data.len() - 1;
    let mask: DoubleBigDigit = (1 << bits) - 1;
    let digits = (u.bits() + bits - 1) / bits;
    let mut res = Vec::with_capacity(digits);

    let mut r = 0;
    let mut rbits = 0;
    for hi in u.data[..last_i].iter().cloned() {
        r |= (hi as DoubleBigDigit) << rbits;
        rbits += big_digit::BITS;

        while rbits >= bits {
            res.push((r & mask) as u8);
            r >>= bits;
            rbits -= bits;
        }
    }

    r |= (u.data[last_i] as DoubleBigDigit) << rbits;
    while r != 0 {
        res.push((r & mask) as u8);
        r >>= bits;
    }

    res
}

// Extract little-endian radix digits
#[inline(always)] // forced inline to get const-prop for radix=10
fn to_radix_digits_le(u: &BigUint, radix: u32) -> Vec<u8> {
    debug_assert!(!u.is_zero() && !radix.is_power_of_two());

    // Estimate how big the result will be, so we can pre-allocate it.
    let radix_digits = ((u.bits() as f64) / (radix as f64).log2()).ceil();
    let mut res = Vec::with_capacity(radix_digits as usize);
    let mut digits = u.clone();

    let (base, power) = get_radix_base(radix);
    debug_assert!(base < (1 << 32));
    let base = base as BigDigit;

    while digits.data.len() > 1 {
        let (q, mut r) = div_rem_digit(digits, base);
        for _ in 0..power {
            res.push((r % radix) as u8);
            r /= radix;
        }
        digits = q;
    }

    let mut r = digits.data[0];
    while r != 0 {
        res.push((r % radix) as u8);
        r /= radix;
    }

    res
}

fn to_str_radix_reversed(u: &BigUint, radix: u32) -> Vec<u8> {
    assert!(2 <= radix && radix <= 36, "The radix must be within 2...36");

    if u.is_zero() {
        return vec![b'0']
    }

    let mut res = if radix.is_power_of_two() {
        // Powers of two can use bitwise masks and shifting instead of division
        let bits = radix.trailing_zeros() as usize;
        if big_digit::BITS % bits == 0 {
            to_bitwise_digits_le(u, bits)
        } else {
            to_inexact_bitwise_digits_le(u, bits)
        }
    } else if radix == 10 {
        // 10 is so common that it's worth separating out for const-propagation.
        // Optimizers can often turn constant division into a faster multiplication.
        to_radix_digits_le(u, 10)
    } else {
        to_radix_digits_le(u, radix)
    };

    // Now convert everything to ASCII digits.
    for r in &mut res {
        const DIGITS: &'static [u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        *r = DIGITS[*r as usize];
    }
    res
}

impl BigUint {
    /// Creates and initializes a `BigUint`.
    ///
    /// The digits are in little-endian base 2^32.
    #[inline]
    pub fn new(digits: Vec<BigDigit>) -> BigUint {
        BigUint { data: digits }.normalize()
    }

    /// Creates and initializes a `BigUint`.
    ///
    /// The digits are in little-endian base 2^32.
    #[inline]
    pub fn from_slice(slice: &[BigDigit]) -> BigUint {
        BigUint::new(slice.to_vec())
    }

    /// Creates and initializes a `BigUint`.
    ///
    /// The bytes are in big-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::BigUint;
    ///
    /// assert_eq!(BigUint::from_bytes_be(b"A"),
    ///            BigUint::parse_bytes(b"65", 10).unwrap());
    /// assert_eq!(BigUint::from_bytes_be(b"AA"),
    ///            BigUint::parse_bytes(b"16705", 10).unwrap());
    /// assert_eq!(BigUint::from_bytes_be(b"AB"),
    ///            BigUint::parse_bytes(b"16706", 10).unwrap());
    /// assert_eq!(BigUint::from_bytes_be(b"Hello world!"),
    ///            BigUint::parse_bytes(b"22405534230753963835153736737", 10).unwrap());
    /// ```
    #[inline]
    pub fn from_bytes_be(bytes: &[u8]) -> BigUint {
        if bytes.is_empty() {
            Zero::zero()
        } else {
            let mut v = bytes.to_vec();
            v.reverse();
            BigUint::from_bytes_le(&*v)
        }
    }

    /// Creates and initializes a `BigUint`.
    ///
    /// The bytes are in little-endian byte order.
    #[inline]
    pub fn from_bytes_le(bytes: &[u8]) -> BigUint {
        if bytes.is_empty() {
            Zero::zero()
        } else {
            from_bitwise_digits_le(bytes, 8)
        }
    }

    /// Returns the byte representation of the `BigUint` in little-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::BigUint;
    ///
    /// let i = BigUint::parse_bytes(b"1125", 10).unwrap();
    /// assert_eq!(i.to_bytes_le(), vec![101, 4]);
    /// ```
    #[inline]
    pub fn to_bytes_le(&self) -> Vec<u8> {
        if self.is_zero() {
            vec![0]
        } else {
            to_bitwise_digits_le(self, 8)
        }
    }

    /// Returns the byte representation of the `BigUint` in big-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::BigUint;
    ///
    /// let i = BigUint::parse_bytes(b"1125", 10).unwrap();
    /// assert_eq!(i.to_bytes_be(), vec![4, 101]);
    /// ```
    #[inline]
    pub fn to_bytes_be(&self) -> Vec<u8> {
        let mut v = self.to_bytes_le();
        v.reverse();
        v
    }

    /// Returns the integer formatted as a string in the given radix.
    /// `radix` must be in the range `[2, 36]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::BigUint;
    ///
    /// let i = BigUint::parse_bytes(b"ff", 16).unwrap();
    /// assert_eq!(i.to_str_radix(16), "ff");
    /// ```
    #[inline]
    pub fn to_str_radix(&self, radix: u32) -> String {
        let mut v = to_str_radix_reversed(self, radix);
        v.reverse();
        unsafe { String::from_utf8_unchecked(v) }
    }

    /// Creates and initializes a `BigUint`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::{BigUint, ToBigUint};
    ///
    /// assert_eq!(BigUint::parse_bytes(b"1234", 10), ToBigUint::to_biguint(&1234));
    /// assert_eq!(BigUint::parse_bytes(b"ABCD", 16), ToBigUint::to_biguint(&0xABCD));
    /// assert_eq!(BigUint::parse_bytes(b"G", 16), None);
    /// ```
    #[inline]
    pub fn parse_bytes(buf: &[u8], radix: u32) -> Option<BigUint> {
        str::from_utf8(buf).ok().and_then(|s| BigUint::from_str_radix(s, radix).ok())
    }

    #[inline]
    fn shl_unit(&self, n_unit: usize) -> BigUint {
        if n_unit == 0 || self.is_zero() { return self.clone(); }

        let mut v = vec![0; n_unit];
        v.extend(self.data.iter().cloned());
        BigUint::new(v)
    }

    #[inline]
    fn shl_bits(self, n_bits: usize) -> BigUint {
        if n_bits == 0 || self.is_zero() { return self; }

        assert!(n_bits < big_digit::BITS);

        let mut carry = 0;
        let mut shifted = self.data;
        for elem in shifted.iter_mut() {
            let new_carry = *elem >> (big_digit::BITS - n_bits);
            *elem = (*elem << n_bits) | carry;
            carry = new_carry;
        }
        if carry != 0 {
            shifted.push(carry);
        }
        BigUint::new(shifted)
    }

    #[inline]
    fn shr_unit(&self, n_unit: usize) -> BigUint {
        if n_unit == 0 { return self.clone(); }
        if self.data.len() < n_unit { return Zero::zero(); }
        BigUint::from_slice(&self.data[n_unit ..])
    }

    #[inline]
    fn shr_bits(self, n_bits: usize) -> BigUint {
        if n_bits == 0 || self.data.is_empty() { return self; }

        assert!(n_bits < big_digit::BITS);

        let mut borrow = 0;
        let mut shifted = self.data;
        for elem in shifted.iter_mut().rev() {
            let new_borrow = *elem << (big_digit::BITS - n_bits);
            *elem = (*elem >> n_bits) | borrow;
            borrow = new_borrow;
        }
        BigUint::new(shifted)
    }

    /// Determines the fewest bits necessary to express the `BigUint`.
    pub fn bits(&self) -> usize {
        if self.is_zero() { return 0; }
        let zeros = self.data.last().unwrap().leading_zeros();
        return self.data.len()*big_digit::BITS - zeros as usize;
    }

    /// Strips off trailing zero bigdigits - comparisons require the last element in the vector to
    /// be nonzero.
    #[inline]
    fn normalize(mut self) -> BigUint {
        while let Some(&0) = self.data.last() {
            self.data.pop();
        }
        self
    }
}

// `DoubleBigDigit` size dependent
/// Returns the greatest power of the radix <= big_digit::BASE
#[inline]
fn get_radix_base(radix: u32) -> (DoubleBigDigit, usize) {
    // To generate this table:
    //    let target = std::u32::max as u64 + 1;
    //    for radix in 2u64..37 {
    //        let power = (target as f64).log(radix as f64) as u32;
    //        let base = radix.pow(power);
    //        println!("({:10}, {:2}), // {:2}", base, power, radix);
    //    }
    const BASES: [(DoubleBigDigit, usize); 37] = [
        (0, 0), (0, 0),
        (4294967296, 32), //  2
        (3486784401, 20), //  3
        (4294967296, 16), //  4
        (1220703125, 13), //  5
        (2176782336, 12), //  6
        (1977326743, 11), //  7
        (1073741824, 10), //  8
        (3486784401, 10), //  9
        (1000000000,  9), // 10
        (2357947691,  9), // 11
        ( 429981696,  8), // 12
        ( 815730721,  8), // 13
        (1475789056,  8), // 14
        (2562890625,  8), // 15
        (4294967296,  8), // 16
        ( 410338673,  7), // 17
        ( 612220032,  7), // 18
        ( 893871739,  7), // 19
        (1280000000,  7), // 20
        (1801088541,  7), // 21
        (2494357888,  7), // 22
        (3404825447,  7), // 23
        ( 191102976,  6), // 24
        ( 244140625,  6), // 25
        ( 308915776,  6), // 26
        ( 387420489,  6), // 27
        ( 481890304,  6), // 28
        ( 594823321,  6), // 29
        ( 729000000,  6), // 30
        ( 887503681,  6), // 31
        (1073741824,  6), // 32
        (1291467969,  6), // 33
        (1544804416,  6), // 34
        (1838265625,  6), // 35
        (2176782336,  6), // 36
    ];

    assert!(2 <= radix && radix <= 36, "The radix must be within 2...36");
    BASES[radix as usize]
}

/// A Sign is a `BigInt`'s composing element.
#[derive(PartialEq, PartialOrd, Eq, Ord, Copy, Clone, Debug, RustcEncodable, RustcDecodable, Hash)]
pub enum Sign { Minus, NoSign, Plus }

impl Neg for Sign {
    type Output = Sign;

    /// Negate Sign value.
    #[inline]
    fn neg(self) -> Sign {
        match self {
          Minus  => Plus,
          NoSign => NoSign,
          Plus   => Minus
        }
    }
}

impl Mul<Sign> for Sign {
    type Output = Sign;

    #[inline]
    fn mul(self, other: Sign) -> Sign {
        match (self, other) {
            (NoSign, _) | (_, NoSign)  => NoSign,
            (Plus, Plus) | (Minus, Minus) => Plus,
            (Plus, Minus) | (Minus, Plus) => Minus,
        }
    }
}

/// A big signed integer type.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Hash)]
pub struct BigInt {
    sign: Sign,
    data: BigUint
}

impl PartialEq for BigInt {
    #[inline]
    fn eq(&self, other: &BigInt) -> bool {
        self.cmp(other) == Equal
    }
}

impl Eq for BigInt {}

impl PartialOrd for BigInt {
    #[inline]
    fn partial_cmp(&self, other: &BigInt) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    #[inline]
    fn cmp(&self, other: &BigInt) -> Ordering {
        let scmp = self.sign.cmp(&other.sign);
        if scmp != Equal { return scmp; }

        match self.sign {
            NoSign  => Equal,
            Plus  => self.data.cmp(&other.data),
            Minus => other.data.cmp(&self.data),
        }
    }
}

impl Default for BigInt {
    #[inline]
    fn default() -> BigInt { Zero::zero() }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_str_radix(10))
    }
}

impl FromStr for BigInt {
    type Err = ParseBigIntError;

    #[inline]
    fn from_str(s: &str) -> Result<BigInt, ParseBigIntError> {
        BigInt::from_str_radix(s, 10)
    }
}

impl Num for BigInt {
    type FromStrRadixErr = ParseBigIntError;

    /// Creates and initializes a BigInt.
    #[inline]
    fn from_str_radix(mut s: &str, radix: u32) -> Result<BigInt, ParseBigIntError> {
        let sign = if s.starts_with('-') { s = &s[1..]; Minus } else { Plus };
        let bu = try!(BigUint::from_str_radix(s, radix));
        Ok(BigInt::from_biguint(sign, bu))
    }
}

impl Shl<usize> for BigInt {
    type Output = BigInt;

    #[inline]
    fn shl(self, rhs: usize) -> BigInt { (&self) << rhs }
}

impl<'a> Shl<usize> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn shl(self, rhs: usize) -> BigInt {
        BigInt::from_biguint(self.sign, &self.data << rhs)
    }
}

impl Shr<usize> for BigInt {
    type Output = BigInt;

    #[inline]
    fn shr(self, rhs: usize) -> BigInt { (&self) >> rhs }
}

impl<'a> Shr<usize> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn shr(self, rhs: usize) -> BigInt {
        BigInt::from_biguint(self.sign, &self.data >> rhs)
    }
}

impl Zero for BigInt {
    #[inline]
    fn zero() -> BigInt {
        BigInt::from_biguint(NoSign, Zero::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool { self.sign == NoSign }
}

impl One for BigInt {
    #[inline]
    fn one() -> BigInt {
        BigInt::from_biguint(Plus, One::one())
    }
}

impl Signed for BigInt {
    #[inline]
    fn abs(&self) -> BigInt {
        match self.sign {
            Plus | NoSign => self.clone(),
            Minus => BigInt::from_biguint(Plus, self.data.clone())
        }
    }

    #[inline]
    fn abs_sub(&self, other: &BigInt) -> BigInt {
        if *self <= *other { Zero::zero() } else { self - other }
    }

    #[inline]
    fn signum(&self) -> BigInt {
        match self.sign {
            Plus  => BigInt::from_biguint(Plus, One::one()),
            Minus => BigInt::from_biguint(Minus, One::one()),
            NoSign  => Zero::zero(),
        }
    }

    #[inline]
    fn is_positive(&self) -> bool { self.sign == Plus }

    #[inline]
    fn is_negative(&self) -> bool { self.sign == Minus }
}

// We want to forward to BigUint::add, but it's not clear how that will go until
// we compare both sign and magnitude.  So we duplicate this body for every
// val/ref combination, deferring that decision to BigUint's own forwarding.
macro_rules! bigint_add {
    ($a:expr, $a_owned:expr, $a_data:expr, $b:expr, $b_owned:expr, $b_data:expr) => {
        match ($a.sign, $b.sign) {
            (_, NoSign) => $a_owned,
            (NoSign, _) => $b_owned,
            // same sign => keep the sign with the sum of magnitudes
            (Plus, Plus) | (Minus, Minus) =>
                BigInt::from_biguint($a.sign, $a_data + $b_data),
            // opposite signs => keep the sign of the larger with the difference of magnitudes
            (Plus, Minus) | (Minus, Plus) =>
                match $a.data.cmp(&$b.data) {
                    Less => BigInt::from_biguint($b.sign, $b_data - $a_data),
                    Greater => BigInt::from_biguint($a.sign, $a_data - $b_data),
                    Equal => Zero::zero(),
                },
        }
    };
}

impl<'a, 'b> Add<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn add(self, other: &BigInt) -> BigInt {
        bigint_add!(self, self.clone(), &self.data, other, other.clone(), &other.data)
    }
}

impl<'a> Add<BigInt> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn add(self, other: BigInt) -> BigInt {
        bigint_add!(self, self.clone(), &self.data, other, other, other.data)
    }
}

impl<'a> Add<&'a BigInt> for BigInt {
    type Output = BigInt;

    #[inline]
    fn add(self, other: &BigInt) -> BigInt {
        bigint_add!(self, self, self.data, other, other.clone(), &other.data)
    }
}

impl Add<BigInt> for BigInt {
    type Output = BigInt;

    #[inline]
    fn add(self, other: BigInt) -> BigInt {
        bigint_add!(self, self, self.data, other, other, other.data)
    }
}

// We want to forward to BigUint::sub, but it's not clear how that will go until
// we compare both sign and magnitude.  So we duplicate this body for every
// val/ref combination, deferring that decision to BigUint's own forwarding.
macro_rules! bigint_sub {
    ($a:expr, $a_owned:expr, $a_data:expr, $b:expr, $b_owned:expr, $b_data:expr) => {
        match ($a.sign, $b.sign) {
            (_, NoSign) => $a_owned,
            (NoSign, _) => -$b_owned,
            // opposite signs => keep the sign of the left with the sum of magnitudes
            (Plus, Minus) | (Minus, Plus) =>
                BigInt::from_biguint($a.sign, $a_data + $b_data),
            // same sign => keep or toggle the sign of the left with the difference of magnitudes
            (Plus, Plus) | (Minus, Minus) =>
                match $a.data.cmp(&$b.data) {
                    Less => BigInt::from_biguint(-$a.sign, $b_data - $a_data),
                    Greater => BigInt::from_biguint($a.sign, $a_data - $b_data),
                    Equal => Zero::zero(),
                },
        }
    };
}

impl<'a, 'b> Sub<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn sub(self, other: &BigInt) -> BigInt {
        bigint_sub!(self, self.clone(), &self.data, other, other.clone(), &other.data)
    }
}

impl<'a> Sub<BigInt> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn sub(self, other: BigInt) -> BigInt {
        bigint_sub!(self, self.clone(), &self.data, other, other, other.data)
    }
}

impl<'a> Sub<&'a BigInt> for BigInt {
    type Output = BigInt;

    #[inline]
    fn sub(self, other: &BigInt) -> BigInt {
        bigint_sub!(self, self, self.data, other, other.clone(), &other.data)
    }
}

impl Sub<BigInt> for BigInt {
    type Output = BigInt;

    #[inline]
    fn sub(self, other: BigInt) -> BigInt {
        bigint_sub!(self, self, self.data, other, other, other.data)
    }
}

forward_all_binop_to_ref_ref!(impl Mul for BigInt, mul);

impl<'a, 'b> Mul<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn mul(self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(self.sign * other.sign,
                             &self.data * &other.data)
    }
}

forward_all_binop_to_ref_ref!(impl Div for BigInt, div);

impl<'a, 'b> Div<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn div(self, other: &BigInt) -> BigInt {
        let (q, _) = self.div_rem(other);
        q
    }
}

forward_all_binop_to_ref_ref!(impl Rem for BigInt, rem);

impl<'a, 'b> Rem<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn rem(self, other: &BigInt) -> BigInt {
        let (_, r) = self.div_rem(other);
        r
    }
}

impl Neg for BigInt {
    type Output = BigInt;

    #[inline]
    fn neg(mut self) -> BigInt {
        self.sign = -self.sign;
        self
    }
}

impl<'a> Neg for &'a BigInt {
    type Output = BigInt;

    #[inline]
    fn neg(self) -> BigInt {
        -self.clone()
    }
}

impl CheckedAdd for BigInt {
    #[inline]
    fn checked_add(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.add(v));
    }
}

impl CheckedSub for BigInt {
    #[inline]
    fn checked_sub(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.sub(v));
    }
}

impl CheckedMul for BigInt {
    #[inline]
    fn checked_mul(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.mul(v));
    }
}

impl CheckedDiv for BigInt {
    #[inline]
    fn checked_div(&self, v: &BigInt) -> Option<BigInt> {
        if v.is_zero() {
            return None;
        }
        return Some(self.div(v));
    }
}

impl Integer for BigInt {
    #[inline]
    fn div_rem(&self, other: &BigInt) -> (BigInt, BigInt) {
        // r.sign == self.sign
        let (d_ui, r_ui) = self.data.div_mod_floor(&other.data);
        let d = BigInt::from_biguint(self.sign, d_ui);
        let r = BigInt::from_biguint(self.sign, r_ui);
        if other.is_negative() { (-d, r) } else { (d, r) }
    }

    #[inline]
    fn div_floor(&self, other: &BigInt) -> BigInt {
        let (d, _) = self.div_mod_floor(other);
        d
    }

    #[inline]
    fn mod_floor(&self, other: &BigInt) -> BigInt {
        let (_, m) = self.div_mod_floor(other);
        m
    }

    fn div_mod_floor(&self, other: &BigInt) -> (BigInt, BigInt) {
        // m.sign == other.sign
        let (d_ui, m_ui) = self.data.div_rem(&other.data);
        let d = BigInt::from_biguint(Plus, d_ui);
        let m = BigInt::from_biguint(Plus, m_ui);
        let one: BigInt = One::one();
        match (self.sign, other.sign) {
            (_,    NoSign)   => panic!(),
            (Plus, Plus)  | (NoSign, Plus)  => (d, m),
            (Plus, Minus) | (NoSign, Minus) => {
                if m.is_zero() {
                    (-d, Zero::zero())
                } else {
                    (-d - one, m + other)
                }
            },
            (Minus, Plus) => {
                if m.is_zero() {
                    (-d, Zero::zero())
                } else {
                    (-d - one, other - m)
                }
            }
            (Minus, Minus) => (d, -m)
        }
    }

    /// Calculates the Greatest Common Divisor (GCD) of the number and `other`.
    ///
    /// The result is always positive.
    #[inline]
    fn gcd(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.gcd(&other.data))
    }

    /// Calculates the Lowest Common Multiple (LCM) of the number and `other`.
    #[inline]
    fn lcm(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.lcm(&other.data))
    }

    /// Deprecated, use `is_multiple_of` instead.
    #[inline]
    fn divides(&self, other: &BigInt) -> bool { return self.is_multiple_of(other); }

    /// Returns `true` if the number is a multiple of `other`.
    #[inline]
    fn is_multiple_of(&self, other: &BigInt) -> bool { self.data.is_multiple_of(&other.data) }

    /// Returns `true` if the number is divisible by `2`.
    #[inline]
    fn is_even(&self) -> bool { self.data.is_even() }

    /// Returns `true` if the number is not divisible by `2`.
    #[inline]
    fn is_odd(&self) -> bool { self.data.is_odd() }
}

impl ToPrimitive for BigInt {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        match self.sign {
            Plus  => self.data.to_i64(),
            NoSign  => Some(0),
            Minus => {
                self.data.to_u64().and_then(|n| {
                    let m: u64 = 1 << 63;
                    if n < m {
                        Some(-(n as i64))
                    } else if n == m {
                        Some(i64::MIN)
                    } else {
                        None
                    }
                })
            }
        }
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        match self.sign {
            Plus => self.data.to_u64(),
            NoSign => Some(0),
            Minus => None
        }
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.data.to_f32().map(|n| if self.sign == Minus { -n } else { n })
    }

    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.data.to_f64().map(|n| if self.sign == Minus { -n } else { n })
    }
}

impl FromPrimitive for BigInt {
    #[inline]
    fn from_i64(n: i64) -> Option<BigInt> {
        Some(BigInt::from(n))
    }

    #[inline]
    fn from_u64(n: u64) -> Option<BigInt> {
        Some(BigInt::from(n))
    }

    #[inline]
    fn from_f32(n: f32) -> Option<BigInt> {
        if n >= 0.0 {
            BigUint::from_f32(n).map(|x| BigInt::from_biguint(Plus, x))
        } else {
            BigUint::from_f32(-n).map(|x| BigInt::from_biguint(Minus, x))
        }
    }

    #[inline]
    fn from_f64(n: f64) -> Option<BigInt> {
        if n >= 0.0 {
            BigUint::from_f64(n).map(|x| BigInt::from_biguint(Plus, x))
        } else {
            BigUint::from_f64(-n).map(|x| BigInt::from_biguint(Minus, x))
        }
    }
}

impl From<i64> for BigInt {
    #[inline]
    fn from(n: i64) -> Self {
        if n >= 0 {
            BigInt::from(n as u64)
        } else {
            let u = u64::MAX - (n as u64) + 1;
            BigInt { sign: Minus, data: BigUint::from(u) }
        }
    }
}

macro_rules! impl_bigint_from_int {
    ($T:ty) => {
        impl From<$T> for BigInt {
            #[inline]
            fn from(n: $T) -> Self {
                BigInt::from(n as i64)
            }
        }
    }
}

impl_bigint_from_int!(i8);
impl_bigint_from_int!(i16);
impl_bigint_from_int!(i32);
impl_bigint_from_int!(isize);

impl From<u64> for BigInt {
    #[inline]
    fn from(n: u64) -> Self {
        if n > 0 {
            BigInt { sign: Plus, data: BigUint::from(n) }
        } else {
            BigInt::zero()
        }
    }
}

macro_rules! impl_bigint_from_uint {
    ($T:ty) => {
        impl From<$T> for BigInt {
            #[inline]
            fn from(n: $T) -> Self {
                BigInt::from(n as u64)
            }
        }
    }
}

impl_bigint_from_uint!(u8);
impl_bigint_from_uint!(u16);
impl_bigint_from_uint!(u32);
impl_bigint_from_uint!(usize);

impl From<BigUint> for BigInt {
    #[inline]
    fn from(n: BigUint) -> Self {
        if n.is_zero() {
            BigInt::zero()
        } else {
            BigInt { sign: Plus, data: n }
        }
    }
}

/// A generic trait for converting a value to a `BigInt`.
pub trait ToBigInt {
    /// Converts the value of `self` to a `BigInt`.
    fn to_bigint(&self) -> Option<BigInt>;
}

impl ToBigInt for BigInt {
    #[inline]
    fn to_bigint(&self) -> Option<BigInt> {
        Some(self.clone())
    }
}

impl ToBigInt for BigUint {
    #[inline]
    fn to_bigint(&self) -> Option<BigInt> {
        if self.is_zero() {
            Some(Zero::zero())
        } else {
            Some(BigInt { sign: Plus, data: self.clone() })
        }
    }
}

macro_rules! impl_to_bigint {
    ($T:ty, $from_ty:path) => {
        impl ToBigInt for $T {
            #[inline]
            fn to_bigint(&self) -> Option<BigInt> {
                $from_ty(*self)
            }
        }
    }
}

impl_to_bigint!(isize,  FromPrimitive::from_isize);
impl_to_bigint!(i8,   FromPrimitive::from_i8);
impl_to_bigint!(i16,  FromPrimitive::from_i16);
impl_to_bigint!(i32,  FromPrimitive::from_i32);
impl_to_bigint!(i64,  FromPrimitive::from_i64);
impl_to_bigint!(usize, FromPrimitive::from_usize);
impl_to_bigint!(u8,   FromPrimitive::from_u8);
impl_to_bigint!(u16,  FromPrimitive::from_u16);
impl_to_bigint!(u32,  FromPrimitive::from_u32);
impl_to_bigint!(u64,  FromPrimitive::from_u64);
impl_to_bigint!(f32,  FromPrimitive::from_f32);
impl_to_bigint!(f64,  FromPrimitive::from_f64);

pub trait RandBigInt {
    /// Generate a random `BigUint` of the given bit size.
    fn gen_biguint(&mut self, bit_size: usize) -> BigUint;

    /// Generate a random BigInt of the given bit size.
    fn gen_bigint(&mut self, bit_size: usize) -> BigInt;

    /// Generate a random `BigUint` less than the given bound. Fails
    /// when the bound is zero.
    fn gen_biguint_below(&mut self, bound: &BigUint) -> BigUint;

    /// Generate a random `BigUint` within the given range. The lower
    /// bound is inclusive; the upper bound is exclusive. Fails when
    /// the upper bound is not greater than the lower bound.
    fn gen_biguint_range(&mut self, lbound: &BigUint, ubound: &BigUint) -> BigUint;

    /// Generate a random `BigInt` within the given range. The lower
    /// bound is inclusive; the upper bound is exclusive. Fails when
    /// the upper bound is not greater than the lower bound.
    fn gen_bigint_range(&mut self, lbound: &BigInt, ubound: &BigInt) -> BigInt;
}

impl<R: Rng> RandBigInt for R {
    fn gen_biguint(&mut self, bit_size: usize) -> BigUint {
        let (digits, rem) = bit_size.div_rem(&big_digit::BITS);
        let mut data = Vec::with_capacity(digits+1);
        for _ in 0 .. digits {
            data.push(self.gen());
        }
        if rem > 0 {
            let final_digit: BigDigit = self.gen();
            data.push(final_digit >> (big_digit::BITS - rem));
        }
        BigUint::new(data)
    }

    fn gen_bigint(&mut self, bit_size: usize) -> BigInt {
        // Generate a random BigUint...
        let biguint = self.gen_biguint(bit_size);
        // ...and then randomly assign it a Sign...
        let sign = if biguint.is_zero() {
            // ...except that if the BigUint is zero, we need to try
            // again with probability 0.5. This is because otherwise,
            // the probability of generating a zero BigInt would be
            // double that of any other number.
            if self.gen() {
                return self.gen_bigint(bit_size);
            } else {
                NoSign
            }
        } else if self.gen() {
            Plus
        } else {
            Minus
        };
        BigInt::from_biguint(sign, biguint)
    }

    fn gen_biguint_below(&mut self, bound: &BigUint) -> BigUint {
        assert!(!bound.is_zero());
        let bits = bound.bits();
        loop {
            let n = self.gen_biguint(bits);
            if n < *bound { return n; }
        }
    }

    fn gen_biguint_range(&mut self,
                         lbound: &BigUint,
                         ubound: &BigUint)
                         -> BigUint {
        assert!(*lbound < *ubound);
        return lbound + self.gen_biguint_below(&(ubound - lbound));
    }

    fn gen_bigint_range(&mut self,
                        lbound: &BigInt,
                        ubound: &BigInt)
                        -> BigInt {
        assert!(*lbound < *ubound);
        let delta = (ubound - lbound).to_biguint().unwrap();
        return lbound + self.gen_biguint_below(&delta).to_bigint().unwrap();
    }
}

impl BigInt {
    /// Creates and initializes a BigInt.
    ///
    /// The digits are in little-endian base 2^32.
    #[inline]
    pub fn new(sign: Sign, digits: Vec<BigDigit>) -> BigInt {
        BigInt::from_biguint(sign, BigUint::new(digits))
    }

    /// Creates and initializes a `BigInt`.
    ///
    /// The digits are in little-endian base 2^32.
    #[inline]
    pub fn from_biguint(sign: Sign, data: BigUint) -> BigInt {
        if sign == NoSign || data.is_zero() {
            return BigInt { sign: NoSign, data: Zero::zero() };
        }
        BigInt { sign: sign, data: data }
    }

    /// Creates and initializes a `BigInt`.
    #[inline]
    pub fn from_slice(sign: Sign, slice: &[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_slice(slice))
    }

    /// Creates and initializes a `BigInt`.
    ///
    /// The bytes are in big-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::{BigInt, Sign};
    ///
    /// assert_eq!(BigInt::from_bytes_be(Sign::Plus, b"A"),
    ///            BigInt::parse_bytes(b"65", 10).unwrap());
    /// assert_eq!(BigInt::from_bytes_be(Sign::Plus, b"AA"),
    ///            BigInt::parse_bytes(b"16705", 10).unwrap());
    /// assert_eq!(BigInt::from_bytes_be(Sign::Plus, b"AB"),
    ///            BigInt::parse_bytes(b"16706", 10).unwrap());
    /// assert_eq!(BigInt::from_bytes_be(Sign::Plus, b"Hello world!"),
    ///            BigInt::parse_bytes(b"22405534230753963835153736737", 10).unwrap());
    /// ```
    #[inline]
    pub fn from_bytes_be(sign: Sign, bytes: &[u8]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_bytes_be(bytes))
    }

    /// Creates and initializes a `BigInt`.
    ///
    /// The bytes are in little-endian byte order.
    #[inline]
    pub fn from_bytes_le(sign: Sign, bytes: &[u8]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_bytes_le(bytes))
    }

    /// Returns the sign and the byte representation of the `BigInt` in little-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::{ToBigInt, Sign};
    ///
    /// let i = -1125.to_bigint().unwrap();
    /// assert_eq!(i.to_bytes_le(), (Sign::Minus, vec![101, 4]));
    /// ```
    #[inline]
    pub fn to_bytes_le(&self) -> (Sign, Vec<u8>) {
        (self.sign, self.data.to_bytes_le())
    }

    /// Returns the sign and the byte representation of the `BigInt` in big-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::{ToBigInt, Sign};
    ///
    /// let i = -1125.to_bigint().unwrap();
    /// assert_eq!(i.to_bytes_be(), (Sign::Minus, vec![4, 101]));
    /// ```
    #[inline]
    pub fn to_bytes_be(&self) -> (Sign, Vec<u8>) {
        (self.sign, self.data.to_bytes_be())
    }

    /// Returns the integer formatted as a string in the given radix.
    /// `radix` must be in the range `[2, 36]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::BigInt;
    ///
    /// let i = BigInt::parse_bytes(b"ff", 16).unwrap();
    /// assert_eq!(i.to_str_radix(16), "ff");
    /// ```
    #[inline]
    pub fn to_str_radix(&self, radix: u32) -> String {
        let mut v = to_str_radix_reversed(&self.data, radix);

        if self.is_negative() {
            v.push(b'-');
        }

        v.reverse();
        unsafe { String::from_utf8_unchecked(v) }
    }

    /// Returns the sign of the `BigInt` as a `Sign`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::{ToBigInt, Sign};
    ///
    /// assert_eq!(ToBigInt::to_bigint(&1234).unwrap().sign(), Sign::Plus);
    /// assert_eq!(ToBigInt::to_bigint(&-4321).unwrap().sign(), Sign::Minus);
    /// assert_eq!(ToBigInt::to_bigint(&0).unwrap().sign(), Sign::NoSign);
    /// ```
    #[inline]
    pub fn sign(&self) -> Sign {
        self.sign
    }

    /// Creates and initializes a `BigInt`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num::bigint::{BigInt, ToBigInt};
    ///
    /// assert_eq!(BigInt::parse_bytes(b"1234", 10), ToBigInt::to_bigint(&1234));
    /// assert_eq!(BigInt::parse_bytes(b"ABCD", 16), ToBigInt::to_bigint(&0xABCD));
    /// assert_eq!(BigInt::parse_bytes(b"G", 16), None);
    /// ```
    #[inline]
    pub fn parse_bytes(buf: &[u8], radix: u32) -> Option<BigInt> {
        str::from_utf8(buf).ok().and_then(|s| BigInt::from_str_radix(s, radix).ok())
    }


    /// Converts this `BigInt` into a `BigUint`, if it's not negative.
    #[inline]
    pub fn to_biguint(&self) -> Option<BigUint> {
        match self.sign {
            Plus => Some(self.data.clone()),
            NoSign => Some(Zero::zero()),
            Minus => None
        }
    }

    #[inline]
    pub fn checked_add(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.add(v));
    }

    #[inline]
    pub fn checked_sub(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.sub(v));
    }

    #[inline]
    pub fn checked_mul(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.mul(v));
    }

    #[inline]
    pub fn checked_div(&self, v: &BigInt) -> Option<BigInt> {
        if v.is_zero() {
            return None;
        }
        return Some(self.div(v));
    }
}

#[derive(Debug, PartialEq)]
pub enum ParseBigIntError {
    ParseInt(ParseIntError),
    Other,
}

impl fmt::Display for ParseBigIntError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &ParseBigIntError::ParseInt(ref e) => e.fmt(f),
            &ParseBigIntError::Other => "failed to parse provided string".fmt(f)
        }
    }
}

impl Error for ParseBigIntError {
    fn description(&self) -> &str { "failed to parse bigint/biguint" }
}

impl From<ParseIntError> for ParseBigIntError {
    fn from(err: ParseIntError) -> ParseBigIntError {
        ParseBigIntError::ParseInt(err)
    }
}

#[cfg(test)]
mod biguint_tests {
    use Integer;
    use super::{BigDigit, BigUint, ToBigUint, big_digit};
    use super::{BigInt, RandBigInt, ToBigInt};
    use super::Sign::Plus;

    use std::cmp::Ordering::{Less, Equal, Greater};
    use std::{f32, f64};
    use std::i64;
    use std::iter::repeat;
    use std::str::FromStr;
    use std::{u8, u16, u32, u64, usize};

    use rand::thread_rng;
    use {Num, Zero, One, CheckedAdd, CheckedSub, CheckedMul, CheckedDiv};
    use {ToPrimitive, FromPrimitive};
    use Float;

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
    fn test_from_slice() {
        fn check(slice: &[BigDigit], data: &[BigDigit]) {
            assert!(BigUint::from_slice(slice).data == data);
        }
        check(&[1], &[1]);
        check(&[0, 0, 0], &[]);
        check(&[1, 2, 0, 0], &[1, 2]);
        check(&[0, 0, 1, 2], &[0, 0, 1, 2]);
        check(&[0, 0, 1, 2, 0, 0], &[0, 0, 1, 2]);
        check(&[-1i32 as BigDigit], &[-1i32 as BigDigit]);
    }

    #[test]
    fn test_from_bytes_be() {
        fn check(s: &str, result: &str) {
            assert_eq!(BigUint::from_bytes_be(s.as_bytes()),
                       BigUint::parse_bytes(result.as_bytes(), 10).unwrap());
        }
        check("A", "65");
        check("AA", "16705");
        check("AB", "16706");
        check("Hello world!", "22405534230753963835153736737");
        assert_eq!(BigUint::from_bytes_be(&[]), Zero::zero());
    }

    #[test]
    fn test_to_bytes_be() {
        fn check(s: &str, result: &str) {
            let b = BigUint::parse_bytes(result.as_bytes(), 10).unwrap();
            assert_eq!(b.to_bytes_be(), s.as_bytes());
        }
        check("A", "65");
        check("AA", "16705");
        check("AB", "16706");
        check("Hello world!", "22405534230753963835153736737");
        let b: BigUint = Zero::zero();
        assert_eq!(b.to_bytes_be(), [0]);

        // Test with leading/trailing zero bytes and a full BigDigit of value 0
        let b = BigUint::from_str_radix("00010000000000000200", 16).unwrap();
        assert_eq!(b.to_bytes_be(), [1, 0, 0, 0, 0, 0, 0, 2, 0]);
    }

    #[test]
    fn test_from_bytes_le() {
        fn check(s: &str, result: &str) {
            assert_eq!(BigUint::from_bytes_le(s.as_bytes()),
                       BigUint::parse_bytes(result.as_bytes(), 10).unwrap());
        }
        check("A", "65");
        check("AA", "16705");
        check("BA", "16706");
        check("!dlrow olleH", "22405534230753963835153736737");
        assert_eq!(BigUint::from_bytes_le(&[]), Zero::zero());
    }

    #[test]
    fn test_to_bytes_le() {
        fn check(s: &str, result: &str) {
            let b = BigUint::parse_bytes(result.as_bytes(), 10).unwrap();
            assert_eq!(b.to_bytes_le(), s.as_bytes());
        }
        check("A", "65");
        check("AA", "16705");
        check("BA", "16706");
        check("!dlrow olleH", "22405534230753963835153736737");
        let b: BigUint = Zero::zero();
        assert_eq!(b.to_bytes_le(), [0]);

        // Test with leading/trailing zero bytes and a full BigDigit of value 0
        let b = BigUint::from_str_radix("00010000000000000200", 16).unwrap();
        assert_eq!(b.to_bytes_le(), [0, 2, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_cmp() {
        let data: [&[_]; 7] = [ &[], &[1], &[2], &[!0], &[0, 1], &[2, 1], &[1, 1, 1]  ];
        let data: Vec<BigUint> = data.iter().map(|v| BigUint::from_slice(*v)).collect();
        for (i, ni) in data.iter().enumerate() {
            for (j0, nj) in data[i..].iter().enumerate() {
                let j = j0 + i;
                if i == j {
                    assert_eq!(ni.cmp(nj), Equal);
                    assert_eq!(nj.cmp(ni), Equal);
                    assert_eq!(ni, nj);
                    assert!(!(ni != nj));
                    assert!(ni <= nj);
                    assert!(ni >= nj);
                    assert!(!(ni < nj));
                    assert!(!(ni > nj));
                } else {
                    assert_eq!(ni.cmp(nj), Less);
                    assert_eq!(nj.cmp(ni), Greater);

                    assert!(!(ni == nj));
                    assert!(ni != nj);

                    assert!(ni <= nj);
                    assert!(!(ni >= nj));
                    assert!(ni < nj);
                    assert!(!(ni > nj));

                    assert!(!(nj <= ni));
                    assert!(nj >= ni);
                    assert!(!(nj < ni));
                    assert!(nj > ni);
                }
            }
        }
    }

    #[test]
    fn test_hash() {
        let a = BigUint::new(vec!());
        let b = BigUint::new(vec!(0));
        let c = BigUint::new(vec!(1));
        let d = BigUint::new(vec!(1,0,0,0,0,0));
        let e = BigUint::new(vec!(0,0,0,0,0,1));
        assert!(::hash(&a) == ::hash(&b));
        assert!(::hash(&b) != ::hash(&c));
        assert!(::hash(&c) == ::hash(&d));
        assert!(::hash(&d) != ::hash(&e));
    }

    const BIT_TESTS: &'static [(&'static [BigDigit],
                                &'static [BigDigit],
                                &'static [BigDigit],
                                &'static [BigDigit],
                                &'static [BigDigit])] = &[
        // LEFT              RIGHT        AND          OR                XOR
        ( &[],              &[],         &[],         &[],              &[]             ),
        ( &[268, 482, 17],  &[964, 54],  &[260, 34],  &[972, 502, 17],  &[712, 468, 17] ),
    ];

    #[test]
    fn test_bitand() {
        for elm in BIT_TESTS {
            let (a_vec, b_vec, c_vec, _, _) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert_op!(a & b == c);
            assert_op!(b & a == c);
        }
    }

    #[test]
    fn test_bitor() {
        for elm in BIT_TESTS {
            let (a_vec, b_vec, _, c_vec, _) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert_op!(a | b == c);
            assert_op!(b | a == c);
        }
    }

    #[test]
    fn test_bitxor() {
        for elm in BIT_TESTS {
            let (a_vec, b_vec, _, _, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert_op!(a ^ b == c);
            assert_op!(b ^ a == c);
            assert_op!(a ^ c == b);
            assert_op!(c ^ a == b);
            assert_op!(b ^ c == a);
            assert_op!(c ^ b == a);
        }
    }

    #[test]
    fn test_shl() {
        fn check(s: &str, shift: usize, ans: &str) {
            let opt_biguint = BigUint::from_str_radix(s, 16).ok();
            let bu = (opt_biguint.unwrap() << shift).to_str_radix(16);
            assert_eq!(bu, ans);
        }

        check("0", 3, "0");
        check("1", 3, "8");

        check("1\
               0000\
               0000\
               0000\
               0001\
               0000\
               0000\
               0000\
               0001",
              3,
              "8\
               0000\
               0000\
               0000\
               0008\
               0000\
               0000\
               0000\
               0008");
        check("1\
               0000\
               0001\
               0000\
               0001",
              2,
              "4\
               0000\
               0004\
               0000\
               0004");
        check("1\
               0001\
               0001",
              1,
              "2\
               0002\
               0002");

        check("\
              4000\
              0000\
              0000\
              0000",
              3,
              "2\
              0000\
              0000\
              0000\
              0000");
        check("4000\
              0000",
              2,
              "1\
              0000\
              0000");
        check("4000",
              2,
              "1\
              0000");

        check("4000\
              0000\
              0000\
              0000",
              67,
              "2\
              0000\
              0000\
              0000\
              0000\
              0000\
              0000\
              0000\
              0000");
        check("4000\
              0000",
              35,
              "2\
              0000\
              0000\
              0000\
              0000");
        check("4000",
              19,
              "2\
              0000\
              0000");

        check("fedc\
              ba98\
              7654\
              3210\
              fedc\
              ba98\
              7654\
              3210",
              4,
              "f\
              edcb\
              a987\
              6543\
              210f\
              edcb\
              a987\
              6543\
              2100");
        check("88887777666655554444333322221111", 16,
              "888877776666555544443333222211110000");
    }

    #[test]
    fn test_shr() {
        fn check(s: &str, shift: usize, ans: &str) {
            let opt_biguint = BigUint::from_str_radix(s, 16).ok();
            let bu = (opt_biguint.unwrap() >> shift).to_str_radix(16);
            assert_eq!(bu, ans);
        }

        check("0", 3, "0");
        check("f", 3, "1");

        check("1\
              0000\
              0000\
              0000\
              0001\
              0000\
              0000\
              0000\
              0001",
              3,
              "2000\
              0000\
              0000\
              0000\
              2000\
              0000\
              0000\
              0000");
        check("1\
              0000\
              0001\
              0000\
              0001",
              2,
              "4000\
              0000\
              4000\
              0000");
        check("1\
              0001\
              0001",
              1,
              "8000\
              8000");

        check("2\
              0000\
              0000\
              0000\
              0001\
              0000\
              0000\
              0000\
              0001",
              67,
              "4000\
              0000\
              0000\
              0000");
        check("2\
              0000\
              0001\
              0000\
              0001",
              35,
              "4000\
              0000");
        check("2\
              0001\
              0001",
              19,
              "4000");

        check("1\
              0000\
              0000\
              0000\
              0000",
              1,
              "8000\
              0000\
              0000\
              0000");
        check("1\
              0000\
              0000",
              1,
              "8000\
              0000");
        check("1\
              0000",
              1,
              "8000");
        check("f\
              edcb\
              a987\
              6543\
              210f\
              edcb\
              a987\
              6543\
              2100",
              4,
              "fedc\
              ba98\
              7654\
              3210\
              fedc\
              ba98\
              7654\
              3210");

        check("888877776666555544443333222211110000", 16,
              "88887777666655554444333322221111");
    }

    const N1: BigDigit = -1i32 as BigDigit;
    const N2: BigDigit = -2i32 as BigDigit;

    // `DoubleBigDigit` size dependent
    #[test]
    fn test_convert_i64() {
        fn check(b1: BigUint, i: i64) {
            let b2: BigUint = FromPrimitive::from_i64(i).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_i64().unwrap() == i);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(i64::MAX.to_biguint().unwrap(), i64::MAX);

        check(BigUint::new(vec!(           )), 0);
        check(BigUint::new(vec!( 1         )), (1 << (0*big_digit::BITS)));
        check(BigUint::new(vec!(N1         )), (1 << (1*big_digit::BITS)) - 1);
        check(BigUint::new(vec!( 0,  1     )), (1 << (1*big_digit::BITS)));
        check(BigUint::new(vec!(N1, N1 >> 1)), i64::MAX);

        assert_eq!(i64::MIN.to_biguint(), None);
        assert_eq!(BigUint::new(vec!(N1, N1    )).to_i64(), None);
        assert_eq!(BigUint::new(vec!( 0,  0,  1)).to_i64(), None);
        assert_eq!(BigUint::new(vec!(N1, N1, N1)).to_i64(), None);
    }

    // `DoubleBigDigit` size dependent
    #[test]
    fn test_convert_u64() {
        fn check(b1: BigUint, u: u64) {
            let b2: BigUint = FromPrimitive::from_u64(u).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_u64().unwrap() == u);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(u64::MIN.to_biguint().unwrap(), u64::MIN);
        check(u64::MAX.to_biguint().unwrap(), u64::MAX);

        check(BigUint::new(vec!(      )), 0);
        check(BigUint::new(vec!( 1    )), (1 << (0*big_digit::BITS)));
        check(BigUint::new(vec!(N1    )), (1 << (1*big_digit::BITS)) - 1);
        check(BigUint::new(vec!( 0,  1)), (1 << (1*big_digit::BITS)));
        check(BigUint::new(vec!(N1, N1)), u64::MAX);

        assert_eq!(BigUint::new(vec!( 0,  0,  1)).to_u64(), None);
        assert_eq!(BigUint::new(vec!(N1, N1, N1)).to_u64(), None);
    }

    #[test]
    fn test_convert_f32() {
        fn check(b1: &BigUint, f: f32) {
            let b2 = BigUint::from_f32(f).unwrap();
            assert_eq!(b1, &b2);
            assert_eq!(b1.to_f32().unwrap(), f);
        }

        check(&BigUint::zero(), 0.0);
        check(&BigUint::one(), 1.0);
        check(&BigUint::from(u16::MAX), 2.0.powi(16) - 1.0);
        check(&BigUint::from(1u64 << 32), 2.0.powi(32));
        check(&BigUint::from_slice(&[0, 0, 1]), 2.0.powi(64));
        check(&((BigUint::one() << 100) + (BigUint::one() << 123)), 2.0.powi(100) + 2.0.powi(123));
        check(&(BigUint::one() << 127), 2.0.powi(127));
        check(&(BigUint::from((1u64 << 24) - 1) << (128 - 24)), f32::MAX);

        // keeping all 24 digits with the bits at different offsets to the BigDigits
        let x: u32 = 0b00000000101111011111011011011101;
        let mut f = x as f32;
        let mut b = BigUint::from(x);
        for _ in 0..64 {
            check(&b, f);
            f *= 2.0;
            b = b << 1;
        }

        // this number when rounded to f64 then f32 isn't the same as when rounded straight to f32
        let n: u64 = 0b0000000000111111111111111111111111011111111111111111111111111111;
        assert!((n as f64) as f32 != n as f32);
        assert_eq!(BigUint::from(n).to_f32(), Some(n as f32));

        // test rounding up with the bits at different offsets to the BigDigits
        let mut f = ((1u64 << 25) - 1) as f32;
        let mut b = BigUint::from(1u64 << 25);
        for _ in 0..64 {
            assert_eq!(b.to_f32(), Some(f));
            f *= 2.0;
            b = b << 1;
        }

        // rounding
        assert_eq!(BigUint::from_f32(-1.0), None);
        assert_eq!(BigUint::from_f32(-0.99999), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f32(-0.5), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f32(-0.0), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f32(f32::MIN_POSITIVE / 2.0), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f32(f32::MIN_POSITIVE), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f32(0.5), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f32(0.99999), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f32(f32::consts::E), Some(BigUint::from(2u32)));
        assert_eq!(BigUint::from_f32(f32::consts::PI), Some(BigUint::from(3u32)));

        // special float values
        assert_eq!(BigUint::from_f32(f32::NAN), None);
        assert_eq!(BigUint::from_f32(f32::INFINITY), None);
        assert_eq!(BigUint::from_f32(f32::NEG_INFINITY), None);
        assert_eq!(BigUint::from_f32(f32::MIN), None);

        // largest BigUint that will round to a finite f32 value
        let big_num = (BigUint::one() << 128) - BigUint::one() - (BigUint::one() << (128 - 25));
        assert_eq!(big_num.to_f32(), Some(f32::MAX));
        assert_eq!((big_num + BigUint::one()).to_f32(), None);

        assert_eq!(((BigUint::one() << 128) - BigUint::one()).to_f32(), None);
        assert_eq!((BigUint::one() << 128).to_f32(), None);
    }

    #[test]
    fn test_convert_f64() {
        fn check(b1: &BigUint, f: f64) {
            let b2 = BigUint::from_f64(f).unwrap();
            assert_eq!(b1, &b2);
            assert_eq!(b1.to_f64().unwrap(), f);
        }

        check(&BigUint::zero(), 0.0);
        check(&BigUint::one(), 1.0);
        check(&BigUint::from(u32::MAX), 2.0.powi(32) - 1.0);
        check(&BigUint::from(1u64 << 32), 2.0.powi(32));
        check(&BigUint::from_slice(&[0, 0, 1]), 2.0.powi(64));
        check(&((BigUint::one() << 100) + (BigUint::one() << 152)), 2.0.powi(100) + 2.0.powi(152));
        check(&(BigUint::one() << 1023), 2.0.powi(1023));
        check(&(BigUint::from((1u64 << 53) - 1) << (1024 - 53)), f64::MAX);

        // keeping all 53 digits with the bits at different offsets to the BigDigits
        let x: u64 = 0b0000000000011110111110110111111101110111101111011111011011011101;
        let mut f = x as f64;
        let mut b = BigUint::from(x);
        for _ in 0..128 {
            check(&b, f);
            f *= 2.0;
            b = b << 1;
        }

        // test rounding up with the bits at different offsets to the BigDigits
        let mut f = ((1u64 << 54) - 1) as f64;
        let mut b = BigUint::from(1u64 << 54);
        for _ in 0..128 {
            assert_eq!(b.to_f64(), Some(f));
            f *= 2.0;
            b = b << 1;
        }

        // rounding
        assert_eq!(BigUint::from_f64(-1.0), None);
        assert_eq!(BigUint::from_f64(-0.99999), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f64(-0.5), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f64(-0.0), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f64(f64::MIN_POSITIVE / 2.0), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f64(f64::MIN_POSITIVE), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f64(0.5), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f64(0.99999), Some(BigUint::zero()));
        assert_eq!(BigUint::from_f64(f64::consts::E), Some(BigUint::from(2u32)));
        assert_eq!(BigUint::from_f64(f64::consts::PI), Some(BigUint::from(3u32)));

        // special float values
        assert_eq!(BigUint::from_f64(f64::NAN), None);
        assert_eq!(BigUint::from_f64(f64::INFINITY), None);
        assert_eq!(BigUint::from_f64(f64::NEG_INFINITY), None);
        assert_eq!(BigUint::from_f64(f64::MIN), None);

        // largest BigUint that will round to a finite f64 value
        let big_num = (BigUint::one() << 1024) - BigUint::one() - (BigUint::one() << (1024 - 54));
        assert_eq!(big_num.to_f64(), Some(f64::MAX));
        assert_eq!((big_num + BigUint::one()).to_f64(), None);

        assert_eq!(((BigInt::one() << 1024) - BigInt::one()).to_f64(), None);
        assert_eq!((BigUint::one() << 1024).to_f64(), None);
    }

    #[test]
    fn test_convert_to_bigint() {
        fn check(n: BigUint, ans: BigInt) {
            assert_eq!(n.to_bigint().unwrap(), ans);
            assert_eq!(n.to_bigint().unwrap().to_biguint().unwrap(), n);
        }
        check(Zero::zero(), Zero::zero());
        check(BigUint::new(vec!(1,2,3)),
              BigInt::from_biguint(Plus, BigUint::new(vec!(1,2,3))));
    }

    #[test]
    fn test_convert_from_uint() {
        macro_rules! check {
            ($ty:ident, $max:expr) => {
                assert_eq!(BigUint::from($ty::zero()), BigUint::zero());
                assert_eq!(BigUint::from($ty::one()), BigUint::one());
                assert_eq!(BigUint::from($ty::MAX - $ty::one()), $max - BigUint::one());
                assert_eq!(BigUint::from($ty::MAX), $max);
            }
        }

        check!(u8, BigUint::from_slice(&[u8::MAX as BigDigit]));
        check!(u16, BigUint::from_slice(&[u16::MAX as BigDigit]));
        check!(u32, BigUint::from_slice(&[u32::MAX]));
        check!(u64, BigUint::from_slice(&[u32::MAX, u32::MAX]));
        check!(usize, BigUint::from(usize::MAX as u64));
    }

    const SUM_TRIPLES: &'static [(&'static [BigDigit],
                                  &'static [BigDigit],
                                  &'static [BigDigit])] = &[
        (&[],          &[],       &[]),
        (&[],          &[ 1],     &[ 1]),
        (&[ 1],        &[ 1],     &[ 2]),
        (&[ 1],        &[ 1,  1], &[ 2,  1]),
        (&[ 1],        &[N1],     &[ 0,  1]),
        (&[ 1],        &[N1, N1], &[ 0,  0, 1]),
        (&[N1, N1],    &[N1, N1], &[N2, N1, 1]),
        (&[ 1,  1, 1], &[N1, N1], &[ 0,  1, 2]),
        (&[ 2,  2, 1], &[N1, N2], &[ 1,  1, 2])
    ];

    #[test]
    fn test_add() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert_op!(a + b == c);
            assert_op!(b + a == c);
        }
    }

    #[test]
    fn test_sub() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert_op!(c - a == b);
            assert_op!(c - b == a);
        }
    }

    #[test]
    #[should_panic]
    fn test_sub_fail_on_underflow() {
        let (a, b) : (BigUint, BigUint) = (Zero::zero(), One::one());
        a - b;
    }

    const M: u32 = ::std::u32::MAX;
    const MUL_TRIPLES: &'static [(&'static [BigDigit],
                                  &'static [BigDigit],
                                  &'static [BigDigit])] = &[
        (&[],               &[],               &[]),
        (&[],               &[ 1],             &[]),
        (&[ 2],             &[],               &[]),
        (&[ 1],             &[ 1],             &[1]),
        (&[ 2],             &[ 3],             &[ 6]),
        (&[ 1],             &[ 1,  1,  1],     &[1, 1,  1]),
        (&[ 1,  2,  3],     &[ 3],             &[ 3,  6,  9]),
        (&[ 1,  1,  1],     &[N1],             &[N1, N1, N1]),
        (&[ 1,  2,  3],     &[N1],             &[N1, N2, N2, 2]),
        (&[ 1,  2,  3,  4], &[N1],             &[N1, N2, N2, N2, 3]),
        (&[N1],             &[N1],             &[ 1, N2]),
        (&[N1, N1],         &[N1],             &[ 1, N1, N2]),
        (&[N1, N1, N1],     &[N1],             &[ 1, N1, N1, N2]),
        (&[N1, N1, N1, N1], &[N1],             &[ 1, N1, N1, N1, N2]),
        (&[ M/2 + 1],       &[ 2],             &[ 0,  1]),
        (&[0,  M/2 + 1],    &[ 2],             &[ 0,  0,  1]),
        (&[ 1,  2],         &[ 1,  2,  3],     &[1, 4,  7,  6]),
        (&[N1, N1],         &[N1, N1, N1],     &[1, 0, N1, N2, N1]),
        (&[N1, N1, N1],     &[N1, N1, N1, N1], &[1, 0,  0, N1, N2, N1, N1]),
        (&[ 0,  0,  1],     &[ 1,  2,  3],     &[0, 0,  1,  2,  3]),
        (&[ 0,  0,  1],     &[ 0,  0,  0,  1], &[0, 0,  0,  0,  0,  1])
    ];

    const DIV_REM_QUADRUPLES: &'static [(&'static [BigDigit],
                                         &'static [BigDigit],
                                         &'static [BigDigit],
                                         &'static [BigDigit])]
        = &[
            (&[ 1],        &[ 2], &[],               &[1]),
            (&[ 1,  1],    &[ 2], &[ M/2+1],         &[1]),
            (&[ 1,  1, 1], &[ 2], &[ M/2+1,  M/2+1], &[1]),
            (&[ 0,  1],    &[N1], &[1],              &[1]),
            (&[N1, N1],    &[N2], &[2, 1],           &[3])
        ];

    #[test]
    fn test_mul() {
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert_op!(a * b == c);
            assert_op!(b * a == c);
        }

        for elm in DIV_REM_QUADRUPLES.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);
            let d = BigUint::from_slice(d_vec);

            assert!(a == &b * &c + &d);
            assert!(a == &c * &b + &d);
        }
    }

    #[test]
    fn test_div_rem() {
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            if !a.is_zero() {
                assert_op!(c / a == b);
                assert_op!(c % a == Zero::zero());
                assert_eq!(c.div_rem(&a), (b.clone(), Zero::zero()));
            }
            if !b.is_zero() {
                assert_op!(c / b == a);
                assert_op!(c % b == Zero::zero());
                assert_eq!(c.div_rem(&b), (a.clone(), Zero::zero()));
            }
        }

        for elm in DIV_REM_QUADRUPLES.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);
            let d = BigUint::from_slice(d_vec);

            if !b.is_zero() {
                assert_op!(a / b == c);
                assert_op!(a % b == d);
                assert!(a.div_rem(&b) == (c, d));
            }
        }
    }

    #[test]
    fn test_checked_add() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert!(a.checked_add(&b).unwrap() == c);
            assert!(b.checked_add(&a).unwrap() == c);
        }
    }

    #[test]
    fn test_checked_sub() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert!(c.checked_sub(&a).unwrap() == b);
            assert!(c.checked_sub(&b).unwrap() == a);

            if a > c {
                assert!(a.checked_sub(&c).is_none());
            }
            if b > c {
                assert!(b.checked_sub(&c).is_none());
            }
        }
    }

    #[test]
    fn test_checked_mul() {
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert!(a.checked_mul(&b).unwrap() == c);
            assert!(b.checked_mul(&a).unwrap() == c);
        }

        for elm in DIV_REM_QUADRUPLES.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);
            let d = BigUint::from_slice(d_vec);

            assert!(a == b.checked_mul(&c).unwrap() + &d);
            assert!(a == c.checked_mul(&b).unwrap() + &d);
        }
    }

    #[test]
    fn test_checked_div() {
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            if !a.is_zero() {
                assert!(c.checked_div(&a).unwrap() == b);
            }
            if !b.is_zero() {
                assert!(c.checked_div(&b).unwrap() == a);
            }

            assert!(c.checked_div(&Zero::zero()).is_none());
        }
    }

    #[test]
    fn test_gcd() {
        fn check(a: usize, b: usize, c: usize) {
            let big_a: BigUint = FromPrimitive::from_usize(a).unwrap();
            let big_b: BigUint = FromPrimitive::from_usize(b).unwrap();
            let big_c: BigUint = FromPrimitive::from_usize(c).unwrap();

            assert_eq!(big_a.gcd(&big_b), big_c);
        }

        check(10, 2, 2);
        check(10, 3, 1);
        check(0, 3, 3);
        check(3, 3, 3);
        check(56, 42, 14);
    }

    #[test]
    fn test_lcm() {
        fn check(a: usize, b: usize, c: usize) {
            let big_a: BigUint = FromPrimitive::from_usize(a).unwrap();
            let big_b: BigUint = FromPrimitive::from_usize(b).unwrap();
            let big_c: BigUint = FromPrimitive::from_usize(c).unwrap();

            assert_eq!(big_a.lcm(&big_b), big_c);
        }

        check(1, 0, 0);
        check(0, 1, 0);
        check(1, 1, 1);
        check(8, 9, 72);
        check(11, 5, 55);
        check(99, 17, 1683);
    }

    #[test]
    fn test_is_even() {
        let one: BigUint = FromStr::from_str("1").unwrap();
        let two: BigUint = FromStr::from_str("2").unwrap();
        let thousand: BigUint = FromStr::from_str("1000").unwrap();
        let big: BigUint = FromStr::from_str("1000000000000000000000").unwrap();
        let bigger: BigUint = FromStr::from_str("1000000000000000000001").unwrap();
        assert!(one.is_odd());
        assert!(two.is_even());
        assert!(thousand.is_even());
        assert!(big.is_even());
        assert!(bigger.is_odd());
        assert!((&one << 64).is_even());
        assert!(((&one << 64) + one).is_odd());
    }

    fn to_str_pairs() -> Vec<(BigUint, Vec<(u32, String)>)> {
        let bits = big_digit::BITS;
        vec!(( Zero::zero(), vec!(
            (2, "0".to_string()), (3, "0".to_string())
        )), ( BigUint::from_slice(&[ 0xff ]), vec!(
            (2,  "11111111".to_string()),
            (3,  "100110".to_string()),
            (4,  "3333".to_string()),
            (5,  "2010".to_string()),
            (6,  "1103".to_string()),
            (7,  "513".to_string()),
            (8,  "377".to_string()),
            (9,  "313".to_string()),
            (10, "255".to_string()),
            (11, "212".to_string()),
            (12, "193".to_string()),
            (13, "168".to_string()),
            (14, "143".to_string()),
            (15, "120".to_string()),
            (16, "ff".to_string())
        )), ( BigUint::from_slice(&[ 0xfff ]), vec!(
            (2,  "111111111111".to_string()),
            (4,  "333333".to_string()),
            (16, "fff".to_string())
        )), ( BigUint::from_slice(&[ 1, 2 ]), vec!(
            (2,
             format!("10{}1", repeat("0").take(bits - 1).collect::<String>())),
            (4,
             format!("2{}1", repeat("0").take(bits / 2 - 1).collect::<String>())),
            (10, match bits {
                32 => "8589934593".to_string(),
                16 => "131073".to_string(),
                _ => panic!()
            }),
            (16,
             format!("2{}1", repeat("0").take(bits / 4 - 1).collect::<String>()))
        )), ( BigUint::from_slice(&[ 1, 2, 3 ]), vec!(
            (2,
             format!("11{}10{}1",
                     repeat("0").take(bits - 2).collect::<String>(),
                     repeat("0").take(bits - 1).collect::<String>())),
            (4,
             format!("3{}2{}1",
                     repeat("0").take(bits / 2 - 1).collect::<String>(),
                     repeat("0").take(bits / 2 - 1).collect::<String>())),
            (8, match bits {
                32 => "6000000000100000000001".to_string(),
                16 => "140000400001".to_string(),
                _ => panic!()
            }),
            (10, match bits {
                32 => "55340232229718589441".to_string(),
                16 => "12885032961".to_string(),
                _ => panic!()
            }),
            (16,
             format!("3{}2{}1",
                     repeat("0").take(bits / 4 - 1).collect::<String>(),
                     repeat("0").take(bits / 4 - 1).collect::<String>()))
        )) )
    }

    #[test]
    fn test_to_str_radix() {
        let r = to_str_pairs();
        for num_pair in r.iter() {
            let &(ref n, ref rs) = num_pair;
            for str_pair in rs.iter() {
                let &(ref radix, ref str) = str_pair;
                assert_eq!(n.to_str_radix(*radix), *str);
            }
        }
    }

    #[test]
    fn test_from_str_radix() {
        let r = to_str_pairs();
        for num_pair in r.iter() {
            let &(ref n, ref rs) = num_pair;
            for str_pair in rs.iter() {
                let &(ref radix, ref str) = str_pair;
                assert_eq!(n,
                           &BigUint::from_str_radix(str, *radix).unwrap());
            }
        }

        let zed = BigUint::from_str_radix("Z", 10).ok();
        assert_eq!(zed, None);
        let blank = BigUint::from_str_radix("_", 2).ok();
        assert_eq!(blank, None);
        let minus_one = BigUint::from_str_radix("-1", 10).ok();
        assert_eq!(minus_one, None);
    }

    #[test]
    fn test_all_str_radix() {
        use std::ascii::AsciiExt;

        let n = BigUint::new((0..10).collect());
        for radix in 2..37 {
            let s = n.to_str_radix(radix);
            let x = BigUint::from_str_radix(&s, radix);
            assert_eq!(x.unwrap(), n);

            let s = s.to_ascii_uppercase();
            let x = BigUint::from_str_radix(&s, radix);
            assert_eq!(x.unwrap(), n);
        }
    }

    #[test]
    fn test_factor() {
        fn factor(n: usize) -> BigUint {
            let mut f: BigUint = One::one();
            for i in 2..n + 1 {
                // FIXME(#5992): assignment operator overloads
                // f *= FromPrimitive::from_usize(i);
                let bu: BigUint = FromPrimitive::from_usize(i).unwrap();
                f = f * bu;
            }
            return f;
        }

        fn check(n: usize, s: &str) {
            let n = factor(n);
            let ans = match BigUint::from_str_radix(s, 10) {
                Ok(x) => x, Err(_) => panic!()
            };
            assert_eq!(n, ans);
        }

        check(3, "6");
        check(10, "3628800");
        check(20, "2432902008176640000");
        check(30, "265252859812191058636308480000000");
    }

    #[test]
    fn test_bits() {
        assert_eq!(BigUint::new(vec!(0,0,0,0)).bits(), 0);
        let n: BigUint = FromPrimitive::from_usize(0).unwrap();
        assert_eq!(n.bits(), 0);
        let n: BigUint = FromPrimitive::from_usize(1).unwrap();
        assert_eq!(n.bits(), 1);
        let n: BigUint = FromPrimitive::from_usize(3).unwrap();
        assert_eq!(n.bits(), 2);
        let n: BigUint = BigUint::from_str_radix("4000000000", 16).unwrap();
        assert_eq!(n.bits(), 39);
        let one: BigUint = One::one();
        assert_eq!((one << 426).bits(), 427);
    }

    #[test]
    fn test_rand() {
        let mut rng = thread_rng();
        let _n: BigUint = rng.gen_biguint(137);
        assert!(rng.gen_biguint(0).is_zero());
    }

    #[test]
    fn test_rand_range() {
        let mut rng = thread_rng();

        for _ in 0..10 {
            assert_eq!(rng.gen_bigint_range(&FromPrimitive::from_usize(236).unwrap(),
                                            &FromPrimitive::from_usize(237).unwrap()),
                       FromPrimitive::from_usize(236).unwrap());
        }

        let l = FromPrimitive::from_usize(403469000 + 2352).unwrap();
        let u = FromPrimitive::from_usize(403469000 + 3513).unwrap();
        for _ in 0..1000 {
            let n: BigUint = rng.gen_biguint_below(&u);
            assert!(n < u);

            let n: BigUint = rng.gen_biguint_range(&l, &u);
            assert!(n >= l);
            assert!(n < u);
        }
    }

    #[test]
    #[should_panic]
    fn test_zero_rand_range() {
        thread_rng().gen_biguint_range(&FromPrimitive::from_usize(54).unwrap(),
                                     &FromPrimitive::from_usize(54).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_negative_rand_range() {
        let mut rng = thread_rng();
        let l = FromPrimitive::from_usize(2352).unwrap();
        let u = FromPrimitive::from_usize(3513).unwrap();
        // Switching u and l should fail:
        let _n: BigUint = rng.gen_biguint_range(&u, &l);
    }

    #[test]
    fn test_sub_sign() {
        use super::sub_sign;
        let a = BigInt::from_str_radix("265252859812191058636308480000000", 10).unwrap();
        let b = BigInt::from_str_radix("26525285981219105863630848000000", 10).unwrap();

        assert_eq!(sub_sign(&a.data.data[..], &b.data.data[..]), &a - &b);
        assert_eq!(sub_sign(&b.data.data[..], &a.data.data[..]), &b - &a);
    }

    fn test_mul_divide_torture_count(count: usize) {
        use rand::{SeedableRng, StdRng, Rng};

        let bits_max = 1 << 12;
        let seed: &[_] = &[1, 2, 3, 4];
        let mut rng: StdRng = SeedableRng::from_seed(seed);

        for _ in 0..count {
            /* Test with numbers of random sizes: */
            let xbits = rng.gen_range(0, bits_max);
            let ybits = rng.gen_range(0, bits_max);

            let x = rng.gen_biguint(xbits);
            let y = rng.gen_biguint(ybits);

            if x.is_zero() || y.is_zero() {
                continue;
            }

            let prod = &x * &y;
            assert_eq!(&prod / &x, y);
            assert_eq!(&prod / &y, x);
        }
    }

    #[test]
    fn test_mul_divide_torture() {
        test_mul_divide_torture_count(1000);
    }

    #[test]
    #[ignore]
    fn test_mul_divide_torture_long() {
        test_mul_divide_torture_count(1000000);
    }
}

#[cfg(test)]
mod bigint_tests {
    use Integer;
    use super::{BigDigit, BigUint, ToBigUint};
    use super::{Sign, BigInt, RandBigInt, ToBigInt, big_digit};
    use super::Sign::{Minus, NoSign, Plus};

    use std::cmp::Ordering::{Less, Equal, Greater};
    use std::{f32, f64};
    use std::{i8, i16, i32, i64, isize};
    use std::iter::repeat;
    use std::{u8, u16, u32, u64, usize};
    use std::ops::{Neg};

    use rand::thread_rng;

    use {Zero, One, Signed, ToPrimitive, FromPrimitive, Num};
    use Float;

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
    fn test_from_biguint() {
        fn check(inp_s: Sign, inp_n: usize, ans_s: Sign, ans_n: usize) {
            let inp = BigInt::from_biguint(inp_s, FromPrimitive::from_usize(inp_n).unwrap());
            let ans = BigInt { sign: ans_s, data: FromPrimitive::from_usize(ans_n).unwrap()};
            assert_eq!(inp, ans);
        }
        check(Plus, 1, Plus, 1);
        check(Plus, 0, NoSign, 0);
        check(Minus, 1, Minus, 1);
        check(NoSign, 1, NoSign, 0);
    }

    #[test]
    fn test_from_bytes_be() {
        fn check(s: &str, result: &str) {
            assert_eq!(BigInt::from_bytes_be(Plus, s.as_bytes()),
                       BigInt::parse_bytes(result.as_bytes(), 10).unwrap());
        }
        check("A", "65");
        check("AA", "16705");
        check("AB", "16706");
        check("Hello world!", "22405534230753963835153736737");
        assert_eq!(BigInt::from_bytes_be(Plus, &[]), Zero::zero());
        assert_eq!(BigInt::from_bytes_be(Minus, &[]), Zero::zero());
    }

    #[test]
    fn test_to_bytes_be() {
        fn check(s: &str, result: &str) {
            let b = BigInt::parse_bytes(result.as_bytes(), 10).unwrap();
            let (sign, v) = b.to_bytes_be();
            assert_eq!((Plus, s.as_bytes()), (sign, &*v));
        }
        check("A", "65");
        check("AA", "16705");
        check("AB", "16706");
        check("Hello world!", "22405534230753963835153736737");
        let b: BigInt = Zero::zero();
        assert_eq!(b.to_bytes_be(), (NoSign, vec![0]));

        // Test with leading/trailing zero bytes and a full BigDigit of value 0
        let b = BigInt::from_str_radix("00010000000000000200", 16).unwrap();
        assert_eq!(b.to_bytes_be(), (Plus, vec![1, 0, 0, 0, 0, 0, 0, 2, 0]));
    }

    #[test]
    fn test_from_bytes_le() {
        fn check(s: &str, result: &str) {
            assert_eq!(BigInt::from_bytes_le(Plus, s.as_bytes()),
                       BigInt::parse_bytes(result.as_bytes(), 10).unwrap());
        }
        check("A", "65");
        check("AA", "16705");
        check("BA", "16706");
        check("!dlrow olleH", "22405534230753963835153736737");
        assert_eq!(BigInt::from_bytes_le(Plus, &[]), Zero::zero());
        assert_eq!(BigInt::from_bytes_le(Minus, &[]), Zero::zero());
    }

    #[test]
    fn test_to_bytes_le() {
        fn check(s: &str, result: &str) {
            let b = BigInt::parse_bytes(result.as_bytes(), 10).unwrap();
            let (sign, v) = b.to_bytes_le();
            assert_eq!((Plus, s.as_bytes()), (sign, &*v));
        }
        check("A", "65");
        check("AA", "16705");
        check("BA", "16706");
        check("!dlrow olleH", "22405534230753963835153736737");
        let b: BigInt = Zero::zero();
        assert_eq!(b.to_bytes_le(), (NoSign, vec![0]));

        // Test with leading/trailing zero bytes and a full BigDigit of value 0
        let b = BigInt::from_str_radix("00010000000000000200", 16).unwrap();
        assert_eq!(b.to_bytes_le(), (Plus, vec![0, 2, 0, 0, 0, 0, 0, 0, 1]));
    }

    #[test]
    fn test_cmp() {
        let vs: [&[BigDigit]; 4] = [ &[2 as BigDigit], &[1, 1], &[2, 1], &[1, 1, 1] ];
        let mut nums = Vec::new();
        for s in vs.iter().rev() {
            nums.push(BigInt::from_slice(Minus, *s));
        }
        nums.push(Zero::zero());
        nums.extend(vs.iter().map(|s| BigInt::from_slice(Plus, *s)));

        for (i, ni) in nums.iter().enumerate() {
            for (j0, nj) in nums[i..].iter().enumerate() {
                let j = i + j0;
                if i == j {
                    assert_eq!(ni.cmp(nj), Equal);
                    assert_eq!(nj.cmp(ni), Equal);
                    assert_eq!(ni, nj);
                    assert!(!(ni != nj));
                    assert!(ni <= nj);
                    assert!(ni >= nj);
                    assert!(!(ni < nj));
                    assert!(!(ni > nj));
                } else {
                    assert_eq!(ni.cmp(nj), Less);
                    assert_eq!(nj.cmp(ni), Greater);

                    assert!(!(ni == nj));
                    assert!(ni != nj);

                    assert!(ni <= nj);
                    assert!(!(ni >= nj));
                    assert!(ni < nj);
                    assert!(!(ni > nj));

                    assert!(!(nj <= ni));
                    assert!(nj >= ni);
                    assert!(!(nj < ni));
                    assert!(nj > ni);
                }
            }
        }
    }

    #[test]
    fn test_hash() {
        let a = BigInt::new(NoSign, vec!());
        let b = BigInt::new(NoSign, vec!(0));
        let c = BigInt::new(Plus, vec!(1));
        let d = BigInt::new(Plus, vec!(1,0,0,0,0,0));
        let e = BigInt::new(Plus, vec!(0,0,0,0,0,1));
        let f = BigInt::new(Minus, vec!(1));
        assert!(::hash(&a) == ::hash(&b));
        assert!(::hash(&b) != ::hash(&c));
        assert!(::hash(&c) == ::hash(&d));
        assert!(::hash(&d) != ::hash(&e));
        assert!(::hash(&c) != ::hash(&f));
    }

    #[test]
    fn test_convert_i64() {
        fn check(b1: BigInt, i: i64) {
            let b2: BigInt = FromPrimitive::from_i64(i).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_i64().unwrap() == i);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(i64::MIN.to_bigint().unwrap(), i64::MIN);
        check(i64::MAX.to_bigint().unwrap(), i64::MAX);

        assert_eq!(
            (i64::MAX as u64 + 1).to_bigint().unwrap().to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Plus,  BigUint::new(vec!(1, 2, 3, 4, 5))).to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Minus, BigUint::new(vec!(1,0,0,1<<(big_digit::BITS-1)))).to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Minus, BigUint::new(vec!(1, 2, 3, 4, 5))).to_i64(),
            None);
    }

    #[test]
    fn test_convert_u64() {
        fn check(b1: BigInt, u: u64) {
            let b2: BigInt = FromPrimitive::from_u64(u).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_u64().unwrap() == u);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(u64::MIN.to_bigint().unwrap(), u64::MIN);
        check(u64::MAX.to_bigint().unwrap(), u64::MAX);

        assert_eq!(
            BigInt::from_biguint(Plus, BigUint::new(vec!(1, 2, 3, 4, 5))).to_u64(),
            None);

        let max_value: BigUint = FromPrimitive::from_u64(u64::MAX).unwrap();
        assert_eq!(BigInt::from_biguint(Minus, max_value).to_u64(), None);
        assert_eq!(BigInt::from_biguint(Minus, BigUint::new(vec!(1, 2, 3, 4, 5))).to_u64(), None);
    }

    #[test]
    fn test_convert_f32() {
        fn check(b1: &BigInt, f: f32) {
            let b2 = BigInt::from_f32(f).unwrap();
            assert_eq!(b1, &b2);
            assert_eq!(b1.to_f32().unwrap(), f);
            let neg_b1 = -b1;
            let neg_b2 = BigInt::from_f32(-f).unwrap();
            assert_eq!(neg_b1, neg_b2);
            assert_eq!(neg_b1.to_f32().unwrap(), -f);
        }

        check(&BigInt::zero(), 0.0);
        check(&BigInt::one(), 1.0);
        check(&BigInt::from(u16::MAX), 2.0.powi(16) - 1.0);
        check(&BigInt::from(1u64 << 32), 2.0.powi(32));
        check(&BigInt::from_slice(Plus, &[0, 0, 1]), 2.0.powi(64));
        check(&((BigInt::one() << 100) + (BigInt::one() << 123)), 2.0.powi(100) + 2.0.powi(123));
        check(&(BigInt::one() << 127), 2.0.powi(127));
        check(&(BigInt::from((1u64 << 24) - 1) << (128 - 24)), f32::MAX);

        // keeping all 24 digits with the bits at different offsets to the BigDigits
        let x: u32 = 0b00000000101111011111011011011101;
        let mut f = x as f32;
        let mut b = BigInt::from(x);
        for _ in 0..64 {
            check(&b, f);
            f *= 2.0;
            b = b << 1;
        }

        // this number when rounded to f64 then f32 isn't the same as when rounded straight to f32
        let mut n: i64 = 0b0000000000111111111111111111111111011111111111111111111111111111;
        assert!((n as f64) as f32 != n as f32);
        assert_eq!(BigInt::from(n).to_f32(), Some(n as f32));
        n = -n;
        assert!((n as f64) as f32 != n as f32);
        assert_eq!(BigInt::from(n).to_f32(), Some(n as f32));

        // test rounding up with the bits at different offsets to the BigDigits
        let mut f = ((1u64 << 25) - 1) as f32;
        let mut b = BigInt::from(1u64 << 25);
        for _ in 0..64 {
            assert_eq!(b.to_f32(), Some(f));
            f *= 2.0;
            b = b << 1;
        }

        // rounding
        assert_eq!(BigInt::from_f32(-f32::consts::PI), Some(BigInt::from(-3i32)));
        assert_eq!(BigInt::from_f32(-f32::consts::E), Some(BigInt::from(-2i32)));
        assert_eq!(BigInt::from_f32(-0.99999), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f32(-0.5), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f32(-0.0), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f32(f32::MIN_POSITIVE / 2.0), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f32(f32::MIN_POSITIVE), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f32(0.5), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f32(0.99999), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f32(f32::consts::E), Some(BigInt::from(2u32)));
        assert_eq!(BigInt::from_f32(f32::consts::PI), Some(BigInt::from(3u32)));

        // special float values
        assert_eq!(BigInt::from_f32(f32::NAN), None);
        assert_eq!(BigInt::from_f32(f32::INFINITY), None);
        assert_eq!(BigInt::from_f32(f32::NEG_INFINITY), None);

        // largest BigInt that will round to a finite f32 value
        let big_num = (BigInt::one() << 128) - BigInt::one() - (BigInt::one() << (128 - 25));
        assert_eq!(big_num.to_f32(), Some(f32::MAX));
        assert_eq!((&big_num + BigInt::one()).to_f32(), None);
        assert_eq!((-&big_num).to_f32(), Some(f32::MIN));
        assert_eq!(((-&big_num) - BigInt::one()).to_f32(), None);

        assert_eq!(((BigInt::one() << 128) - BigInt::one()).to_f32(), None);
        assert_eq!((BigInt::one() << 128).to_f32(), None);
        assert_eq!((-((BigInt::one() << 128) - BigInt::one())).to_f32(), None);
        assert_eq!((-(BigInt::one() << 128)).to_f32(), None);
    }

    #[test]
    fn test_convert_f64() {
        fn check(b1: &BigInt, f: f64) {
            let b2 =  BigInt::from_f64(f).unwrap();
            assert_eq!(b1, &b2);
            assert_eq!(b1.to_f64().unwrap(), f);
            let neg_b1 = -b1;
            let neg_b2 = BigInt::from_f64(-f).unwrap();
            assert_eq!(neg_b1, neg_b2);
            assert_eq!(neg_b1.to_f64().unwrap(), -f);
        }

        check(&BigInt::zero(), 0.0);
        check(&BigInt::one(), 1.0);
        check(&BigInt::from(u32::MAX), 2.0.powi(32) - 1.0);
        check(&BigInt::from(1u64 << 32), 2.0.powi(32));
        check(&BigInt::from_slice(Plus, &[0, 0, 1]), 2.0.powi(64));
        check(&((BigInt::one() << 100) + (BigInt::one() << 152)), 2.0.powi(100) + 2.0.powi(152));
        check(&(BigInt::one() << 1023), 2.0.powi(1023));
        check(&(BigInt::from((1u64 << 53) - 1) << (1024 - 53)), f64::MAX);

        // keeping all 53 digits with the bits at different offsets to the BigDigits
        let x: u64 = 0b0000000000011110111110110111111101110111101111011111011011011101;
        let mut f = x as f64;
        let mut b = BigInt::from(x);
        for _ in 0..128 {
            check(&b, f);
            f *= 2.0;
            b = b << 1;
        }

        // test rounding up with the bits at different offsets to the BigDigits
        let mut f = ((1u64 << 54) - 1) as f64;
        let mut b = BigInt::from(1u64 << 54);
        for _ in 0..128 {
            assert_eq!(b.to_f64(), Some(f));
            f *= 2.0;
            b = b << 1;
        }

        // rounding
        assert_eq!(BigInt::from_f64(-f64::consts::PI), Some(BigInt::from(-3i32)));
        assert_eq!(BigInt::from_f64(-f64::consts::E), Some(BigInt::from(-2i32)));
        assert_eq!(BigInt::from_f64(-0.99999), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f64(-0.5), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f64(-0.0), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f64(f64::MIN_POSITIVE / 2.0), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f64(f64::MIN_POSITIVE), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f64(0.5), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f64(0.99999), Some(BigInt::zero()));
        assert_eq!(BigInt::from_f64(f64::consts::E), Some(BigInt::from(2u32)));
        assert_eq!(BigInt::from_f64(f64::consts::PI), Some(BigInt::from(3u32)));

        // special float values
        assert_eq!(BigInt::from_f64(f64::NAN), None);
        assert_eq!(BigInt::from_f64(f64::INFINITY), None);
        assert_eq!(BigInt::from_f64(f64::NEG_INFINITY), None);

        // largest BigInt that will round to a finite f64 value
        let big_num = (BigInt::one() << 1024) - BigInt::one() - (BigInt::one() << (1024 - 54));
        assert_eq!(big_num.to_f64(), Some(f64::MAX));
        assert_eq!((&big_num + BigInt::one()).to_f64(), None);
        assert_eq!((-&big_num).to_f64(), Some(f64::MIN));
        assert_eq!(((-&big_num) - BigInt::one()).to_f64(), None);

        assert_eq!(((BigInt::one() << 1024) - BigInt::one()).to_f64(), None);
        assert_eq!((BigInt::one() << 1024).to_f64(), None);
        assert_eq!((-((BigInt::one() << 1024) - BigInt::one())).to_f64(), None);
        assert_eq!((-(BigInt::one() << 1024)).to_f64(), None);
    }

    #[test]
    fn test_convert_to_biguint() {
        fn check(n: BigInt, ans_1: BigUint) {
            assert_eq!(n.to_biguint().unwrap(), ans_1);
            assert_eq!(n.to_biguint().unwrap().to_bigint().unwrap(), n);
        }
        let zero: BigInt = Zero::zero();
        let unsigned_zero: BigUint = Zero::zero();
        let positive = BigInt::from_biguint(
            Plus, BigUint::new(vec!(1,2,3)));
        let negative = -&positive;

        check(zero, unsigned_zero);
        check(positive, BigUint::new(vec!(1,2,3)));

        assert_eq!(negative.to_biguint(), None);
    }

    #[test]
    fn test_convert_from_uint() {
        macro_rules! check {
            ($ty:ident, $max:expr) => {
                assert_eq!(BigInt::from($ty::zero()), BigInt::zero());
                assert_eq!(BigInt::from($ty::one()), BigInt::one());
                assert_eq!(BigInt::from($ty::MAX - $ty::one()), $max - BigInt::one());
                assert_eq!(BigInt::from($ty::MAX), $max);
            }
        }

        check!(u8, BigInt::from_slice(Plus, &[u8::MAX as BigDigit]));
        check!(u16, BigInt::from_slice(Plus, &[u16::MAX as BigDigit]));
        check!(u32, BigInt::from_slice(Plus, &[u32::MAX as BigDigit]));
        check!(u64, BigInt::from_slice(Plus, &[u32::MAX as BigDigit, u32::MAX as BigDigit]));
        check!(usize, BigInt::from(usize::MAX as u64));
    }

    #[test]
    fn test_convert_from_int() {
        macro_rules! check {
            ($ty:ident, $min:expr, $max:expr) => {
                assert_eq!(BigInt::from($ty::MIN), $min);
                assert_eq!(BigInt::from($ty::MIN + $ty::one()), $min + BigInt::one());
                assert_eq!(BigInt::from(-$ty::one()), -BigInt::one());
                assert_eq!(BigInt::from($ty::zero()), BigInt::zero());
                assert_eq!(BigInt::from($ty::one()), BigInt::one());
                assert_eq!(BigInt::from($ty::MAX - $ty::one()), $max - BigInt::one());
                assert_eq!(BigInt::from($ty::MAX), $max);
            }
        }

        check!(i8, BigInt::from_slice(Minus, &[1 << 7]),
                BigInt::from_slice(Plus, &[i8::MAX as BigDigit]));
        check!(i16, BigInt::from_slice(Minus, &[1 << 15]),
                BigInt::from_slice(Plus, &[i16::MAX as BigDigit]));
        check!(i32, BigInt::from_slice(Minus, &[1 << 31]),
                BigInt::from_slice(Plus, &[i32::MAX as BigDigit]));
        check!(i64, BigInt::from_slice(Minus, &[0, 1 << 31]),
                BigInt::from_slice(Plus, &[u32::MAX as BigDigit, i32::MAX as BigDigit]));
        check!(isize, BigInt::from(isize::MIN as i64),
                BigInt::from(isize::MAX as i64));
    }

    #[test]
    fn test_convert_from_biguint() {
        assert_eq!(BigInt::from(BigUint::zero()), BigInt::zero());
        assert_eq!(BigInt::from(BigUint::one()), BigInt::one());
        assert_eq!(BigInt::from(BigUint::from_slice(&[1, 2, 3])), BigInt::from_slice(Plus, &[1, 2, 3]));
    }

    const N1: BigDigit = -1i32 as BigDigit;
    const N2: BigDigit = -2i32 as BigDigit;

    const SUM_TRIPLES: &'static [(&'static [BigDigit],
                                  &'static [BigDigit],
                                  &'static [BigDigit])] = &[
        (&[],          &[],       &[]),
        (&[],          &[ 1],     &[ 1]),
        (&[ 1],        &[ 1],     &[ 2]),
        (&[ 1],        &[ 1,  1], &[ 2,  1]),
        (&[ 1],        &[N1],     &[ 0,  1]),
        (&[ 1],        &[N1, N1], &[ 0,  0, 1]),
        (&[N1, N1],    &[N1, N1], &[N2, N1, 1]),
        (&[ 1,  1, 1], &[N1, N1], &[ 0,  1, 2]),
        (&[ 2,  2, 1], &[N1, N2], &[ 1,  1, 2])
    ];

    #[test]
    fn test_add() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let (na, nb, nc) = (-&a, -&b, -&c);

            assert_op!(a + b == c);
            assert_op!(b + a == c);
            assert_op!(c + na == b);
            assert_op!(c + nb == a);
            assert_op!(a + nc == nb);
            assert_op!(b + nc == na);
            assert_op!(na + nb == nc);
            assert_op!(a + na == Zero::zero());
        }
    }

    #[test]
    fn test_sub() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let (na, nb, nc) = (-&a, -&b, -&c);

            assert_op!(c - a == b);
            assert_op!(c - b == a);
            assert_op!(nb - a == nc);
            assert_op!(na - b == nc);
            assert_op!(b - na == c);
            assert_op!(a - nb == c);
            assert_op!(nc - na == nb);
            assert_op!(a - a == Zero::zero());
        }
    }

    const M: u32 = ::std::u32::MAX;
    static MUL_TRIPLES: &'static [(&'static [BigDigit],
                                   &'static [BigDigit],
                                   &'static [BigDigit])] = &[
        (&[],               &[],               &[]),
        (&[],               &[ 1],             &[]),
        (&[ 2],             &[],               &[]),
        (&[ 1],             &[ 1],             &[1]),
        (&[ 2],             &[ 3],             &[ 6]),
        (&[ 1],             &[ 1,  1,  1],     &[1, 1,  1]),
        (&[ 1,  2,  3],     &[ 3],             &[ 3,  6,  9]),
        (&[ 1,  1,  1],     &[N1],             &[N1, N1, N1]),
        (&[ 1,  2,  3],     &[N1],             &[N1, N2, N2, 2]),
        (&[ 1,  2,  3,  4], &[N1],             &[N1, N2, N2, N2, 3]),
        (&[N1],             &[N1],             &[ 1, N2]),
        (&[N1, N1],         &[N1],             &[ 1, N1, N2]),
        (&[N1, N1, N1],     &[N1],             &[ 1, N1, N1, N2]),
        (&[N1, N1, N1, N1], &[N1],             &[ 1, N1, N1, N1, N2]),
        (&[ M/2 + 1],       &[ 2],             &[ 0,  1]),
        (&[0,  M/2 + 1],    &[ 2],             &[ 0,  0,  1]),
        (&[ 1,  2],         &[ 1,  2,  3],     &[1, 4,  7,  6]),
        (&[N1, N1],         &[N1, N1, N1],     &[1, 0, N1, N2, N1]),
        (&[N1, N1, N1],     &[N1, N1, N1, N1], &[1, 0,  0, N1, N2, N1, N1]),
        (&[ 0,  0,  1],     &[ 1,  2,  3],     &[0, 0,  1,  2,  3]),
        (&[ 0,  0,  1],     &[ 0,  0,  0,  1], &[0, 0,  0,  0,  0,  1])
    ];

    static DIV_REM_QUADRUPLES: &'static [(&'static [BigDigit],
                                          &'static [BigDigit],
                                          &'static [BigDigit],
                                          &'static [BigDigit])]
        = &[
            (&[ 1],        &[ 2], &[],               &[1]),
            (&[ 1,  1],    &[ 2], &[ M/2+1],         &[1]),
            (&[ 1,  1, 1], &[ 2], &[ M/2+1,  M/2+1], &[1]),
            (&[ 0,  1],    &[N1], &[1],              &[1]),
            (&[N1, N1],    &[N2], &[2, 1],           &[3])
        ];

    #[test]
    fn test_mul() {
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let (na, nb, nc) = (-&a, -&b, -&c);

            assert_op!(a * b == c);
            assert_op!(b * a == c);
            assert_op!(na * nb == c);

            assert_op!(na * b == nc);
            assert_op!(nb * a == nc);
        }

        for elm in DIV_REM_QUADRUPLES.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let d = BigInt::from_slice(Plus, d_vec);

            assert!(a == &b * &c + &d);
            assert!(a == &c * &b + &d);
        }
    }

    #[test]
    fn test_div_mod_floor() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_d: &BigInt, ans_m: &BigInt) {
            let (d, m) = a.div_mod_floor(b);
            if !m.is_zero() {
                assert_eq!(m.sign, b.sign);
            }
            assert!(m.abs() <= b.abs());
            assert!(*a == b * &d + &m);
            assert!(d == *ans_d);
            assert!(m == *ans_m);
        }

        fn check(a: &BigInt, b: &BigInt, d: &BigInt, m: &BigInt) {
            if m.is_zero() {
                check_sub(a, b, d, m);
                check_sub(a, &b.neg(), &d.neg(), m);
                check_sub(&a.neg(), b, &d.neg(), m);
                check_sub(&a.neg(), &b.neg(), d, m);
            } else {
                let one: BigInt = One::one();
                check_sub(a, b, d, m);
                check_sub(a, &b.neg(), &(d.neg() - &one), &(m - b));
                check_sub(&a.neg(), b, &(d.neg() - &one), &(b - m));
                check_sub(&a.neg(), &b.neg(), d, &m.neg());
            }
        }

        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for elm in DIV_REM_QUADRUPLES.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let d = BigInt::from_slice(Plus, d_vec);

            if !b.is_zero() {
                check(&a, &b, &c, &d);
            }
        }
    }


    #[test]
    fn test_div_rem() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_q: &BigInt, ans_r: &BigInt) {
            let (q, r) = a.div_rem(b);
            if !r.is_zero() {
                assert_eq!(r.sign, a.sign);
            }
            assert!(r.abs() <= b.abs());
            assert!(*a == b * &q + &r);
            assert!(q == *ans_q);
            assert!(r == *ans_r);

            let (a, b, ans_q, ans_r) = (a.clone(), b.clone(), ans_q.clone(), ans_r.clone());
            assert_op!(a / b == ans_q);
            assert_op!(a % b == ans_r);
        }

        fn check(a: &BigInt, b: &BigInt, q: &BigInt, r: &BigInt) {
            check_sub(a, b, q, r);
            check_sub(a, &b.neg(), &q.neg(), r);
            check_sub(&a.neg(), b, &q.neg(), &r.neg());
            check_sub(&a.neg(), &b.neg(), q, &r.neg());
        }
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for elm in DIV_REM_QUADRUPLES.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let d = BigInt::from_slice(Plus, d_vec);

            if !b.is_zero() {
                check(&a, &b, &c, &d);
            }
        }
    }

    #[test]
    fn test_checked_add() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            assert!(a.checked_add(&b).unwrap() == c);
            assert!(b.checked_add(&a).unwrap() == c);
            assert!(c.checked_add(&(-&a)).unwrap() == b);
            assert!(c.checked_add(&(-&b)).unwrap() == a);
            assert!(a.checked_add(&(-&c)).unwrap() == (-&b));
            assert!(b.checked_add(&(-&c)).unwrap() == (-&a));
            assert!((-&a).checked_add(&(-&b)).unwrap() == (-&c));
            assert!(a.checked_add(&(-&a)).unwrap() == Zero::zero());
        }
    }

    #[test]
    fn test_checked_sub() {
        for elm in SUM_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            assert!(c.checked_sub(&a).unwrap() == b);
            assert!(c.checked_sub(&b).unwrap() == a);
            assert!((-&b).checked_sub(&a).unwrap() == (-&c));
            assert!((-&a).checked_sub(&b).unwrap() == (-&c));
            assert!(b.checked_sub(&(-&a)).unwrap() == c);
            assert!(a.checked_sub(&(-&b)).unwrap() == c);
            assert!((-&c).checked_sub(&(-&a)).unwrap() == (-&b));
            assert!(a.checked_sub(&a).unwrap() == Zero::zero());
        }
    }

    #[test]
    fn test_checked_mul() {
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            assert!(a.checked_mul(&b).unwrap() == c);
            assert!(b.checked_mul(&a).unwrap() == c);

            assert!((-&a).checked_mul(&b).unwrap() == -&c);
            assert!((-&b).checked_mul(&a).unwrap() == -&c);
        }

        for elm in DIV_REM_QUADRUPLES.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let d = BigInt::from_slice(Plus, d_vec);

            assert!(a == b.checked_mul(&c).unwrap() + &d);
            assert!(a == c.checked_mul(&b).unwrap() + &d);
        }
    }
    #[test]
    fn test_checked_div() {
        for elm in MUL_TRIPLES.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            if !a.is_zero() {
                assert!(c.checked_div(&a).unwrap() == b);
                assert!((-&c).checked_div(&(-&a)).unwrap() == b);
                assert!((-&c).checked_div(&a).unwrap() == -&b);
            }
            if !b.is_zero() {
                assert!(c.checked_div(&b).unwrap() == a);
                assert!((-&c).checked_div(&(-&b)).unwrap() == a);
                assert!((-&c).checked_div(&b).unwrap() == -&a);
            }

            assert!(c.checked_div(&Zero::zero()).is_none());
            assert!((-&c).checked_div(&Zero::zero()).is_none());
        }
    }

    #[test]
    fn test_gcd() {
        fn check(a: isize, b: isize, c: isize) {
            let big_a: BigInt = FromPrimitive::from_isize(a).unwrap();
            let big_b: BigInt = FromPrimitive::from_isize(b).unwrap();
            let big_c: BigInt = FromPrimitive::from_isize(c).unwrap();

            assert_eq!(big_a.gcd(&big_b), big_c);
        }

        check(10, 2, 2);
        check(10, 3, 1);
        check(0, 3, 3);
        check(3, 3, 3);
        check(56, 42, 14);
        check(3, -3, 3);
        check(-6, 3, 3);
        check(-4, -2, 2);
    }

    #[test]
    fn test_lcm() {
        fn check(a: isize, b: isize, c: isize) {
            let big_a: BigInt = FromPrimitive::from_isize(a).unwrap();
            let big_b: BigInt = FromPrimitive::from_isize(b).unwrap();
            let big_c: BigInt = FromPrimitive::from_isize(c).unwrap();

            assert_eq!(big_a.lcm(&big_b), big_c);
        }

        check(1, 0, 0);
        check(0, 1, 0);
        check(1, 1, 1);
        check(-1, 1, 1);
        check(1, -1, 1);
        check(-1, -1, 1);
        check(8, 9, 72);
        check(11, 5, 55);
    }

    #[test]
    fn test_abs_sub() {
        let zero: BigInt = Zero::zero();
        let one: BigInt = One::one();
        assert_eq!((-&one).abs_sub(&one), zero);
        let one: BigInt = One::one();
        let zero: BigInt = Zero::zero();
        assert_eq!(one.abs_sub(&one), zero);
        let one: BigInt = One::one();
        let zero: BigInt = Zero::zero();
        assert_eq!(one.abs_sub(&zero), one);
        let one: BigInt = One::one();
        let two: BigInt = FromPrimitive::from_isize(2).unwrap();
        assert_eq!(one.abs_sub(&-&one), two);
    }

    #[test]
    fn test_from_str_radix() {
        fn check(s: &str, ans: Option<isize>) {
            let ans = ans.map(|n| {
                let x: BigInt = FromPrimitive::from_isize(n).unwrap();
                x
            });
            assert_eq!(BigInt::from_str_radix(s, 10).ok(), ans);
        }
        check("10", Some(10));
        check("1", Some(1));
        check("0", Some(0));
        check("-1", Some(-1));
        check("-10", Some(-10));
        check("Z", None);
        check("_", None);

        // issue 10522, this hit an edge case that caused it to
        // attempt to allocate a vector of size (-1u) == huge.
        let x: BigInt =
            format!("1{}", repeat("0").take(36).collect::<String>()).parse().unwrap();
        let _y = x.to_string();
    }

    #[test]
    fn test_neg() {
        assert!(-BigInt::new(Plus,  vec!(1, 1, 1)) ==
            BigInt::new(Minus, vec!(1, 1, 1)));
        assert!(-BigInt::new(Minus, vec!(1, 1, 1)) ==
            BigInt::new(Plus,  vec!(1, 1, 1)));
        let zero: BigInt = Zero::zero();
        assert_eq!(-&zero, zero);
    }

    #[test]
    fn test_rand() {
        let mut rng = thread_rng();
        let _n: BigInt = rng.gen_bigint(137);
        assert!(rng.gen_bigint(0).is_zero());
    }

    #[test]
    fn test_rand_range() {
        let mut rng = thread_rng();

        for _ in 0..10 {
            assert_eq!(rng.gen_bigint_range(&FromPrimitive::from_usize(236).unwrap(),
                                            &FromPrimitive::from_usize(237).unwrap()),
                       FromPrimitive::from_usize(236).unwrap());
        }

        fn check(l: BigInt, u: BigInt) {
            let mut rng = thread_rng();
            for _ in 0..1000 {
                let n: BigInt = rng.gen_bigint_range(&l, &u);
                assert!(n >= l);
                assert!(n < u);
            }
        }
        let l: BigInt = FromPrimitive::from_usize(403469000 + 2352).unwrap();
        let u: BigInt = FromPrimitive::from_usize(403469000 + 3513).unwrap();
        check( l.clone(),  u.clone());
        check(-l.clone(),  u.clone());
        check(-u.clone(), -l.clone());
    }

    #[test]
    #[should_panic]
    fn test_zero_rand_range() {
        thread_rng().gen_bigint_range(&FromPrimitive::from_isize(54).unwrap(),
                                    &FromPrimitive::from_isize(54).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_negative_rand_range() {
        let mut rng = thread_rng();
        let l = FromPrimitive::from_usize(2352).unwrap();
        let u = FromPrimitive::from_usize(3513).unwrap();
        // Switching u and l should fail:
        let _n: BigInt = rng.gen_bigint_range(&u, &l);
    }
}
