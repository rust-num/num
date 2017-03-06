use std::borrow::Cow;
use std::default::Default;
use std::iter::repeat;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Shl, Shr, Sub};
use std::str::{self, FromStr};
use std::fmt;
use std::cmp;
use std::cmp::Ordering::{self, Less, Greater, Equal};
use std::{f32, f64};
use std::{u8, u64};
use std::ascii::AsciiExt;

#[cfg(feature = "serde")]
use serde;

use integer::Integer;
use traits::{ToPrimitive, FromPrimitive, Float, Num, Unsigned, CheckedAdd, CheckedSub, CheckedMul,
             CheckedDiv, Zero, One};

#[path = "algorithms.rs"]
mod algorithms;
pub use self::algorithms::big_digit;
pub use self::big_digit::{BigDigit, DoubleBigDigit, ZERO_BIG_DIGIT};

use self::algorithms::{mac_with_carry, mul3, div_rem, div_rem_digit};
use self::algorithms::{__add2, add2, sub2, sub2rev};
use self::algorithms::{biguint_shl, biguint_shr};
use self::algorithms::{cmp_slice, fls, ilog2};

use ParseBigIntError;

#[cfg(test)]
#[path = "tests/biguint.rs"]
mod biguint_tests;

/// A big unsigned integer type.
///
/// A `BigUint`-typed value `BigUint { data: vec!(a, b, c) }` represents a number
/// `(a + b * big_digit::BASE + c * big_digit::BASE^2)`.
#[derive(Clone, Debug, Hash)]
#[cfg_attr(feature = "rustc-serialize", derive(RustcEncodable, RustcDecodable))]
pub struct BigUint {
    data: Vec<BigDigit>,
}

impl PartialEq for BigUint {
    #[inline]
    fn eq(&self, other: &BigUint) -> bool {
        match self.cmp(other) {
            Equal => true,
            _ => false,
        }
    }
}
impl Eq for BigUint {}

impl PartialOrd for BigUint {
    #[inline]
    fn partial_cmp(&self, other: &BigUint) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigUint {
    #[inline]
    fn cmp(&self, other: &BigUint) -> Ordering {
        cmp_slice(&self.data[..], &other.data[..])
    }
}

impl Default for BigUint {
    #[inline]
    fn default() -> BigUint {
        Zero::zero()
    }
}

impl fmt::Display for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad_integral(true, "", &self.to_str_radix(10))
    }
}

impl fmt::LowerHex for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad_integral(true, "0x", &self.to_str_radix(16))
    }
}

impl fmt::UpperHex for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad_integral(true, "0x", &self.to_str_radix(16).to_ascii_uppercase())
    }
}

impl fmt::Binary for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad_integral(true, "0b", &self.to_str_radix(2))
    }
}

impl fmt::Octal for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad_integral(true, "0o", &self.to_str_radix(8))
    }
}

impl FromStr for BigUint {
    type Err = ParseBigIntError;

    #[inline]
    fn from_str(s: &str) -> Result<BigUint, ParseBigIntError> {
        BigUint::from_str_radix(s, 10)
    }
}

// Convert from a power of two radix (bits == ilog2(radix)) where bits evenly divides
// BigDigit::BITS
fn from_bitwise_digits_le(v: &[u8], bits: usize) -> BigUint {
    debug_assert!(!v.is_empty() && bits <= 8 && big_digit::BITS % bits == 0);
    debug_assert!(v.iter().all(|&c| (c as BigDigit) < (1 << bits)));

    let digits_per_big_digit = big_digit::BITS / bits;

    let data = v.chunks(digits_per_big_digit)
                .map(|chunk| {
                    chunk.iter().rev().fold(0, |acc, &c| (acc << bits) | c as BigDigit)
                })
                .collect();

    BigUint::new(data)
}

// Convert from a power of two radix (bits == ilog2(radix)) where bits doesn't evenly divide
// BigDigit::BITS
fn from_inexact_bitwise_digits_le(v: &[u8], bits: usize) -> BigUint {
    debug_assert!(!v.is_empty() && bits <= 8 && big_digit::BITS % bits != 0);
    debug_assert!(v.iter().all(|&c| (c as BigDigit) < (1 << bits)));

    let big_digits = (v.len() * bits + big_digit::BITS - 1) / big_digit::BITS;
    let mut data = Vec::with_capacity(big_digits);

    let mut d = 0;
    let mut dbits = 0; // number of bits we currently have in d

    // walk v accumululating bits in d; whenever we accumulate big_digit::BITS in d, spit out a
    // big_digit:
    for &c in v {
        d |= (c as BigDigit) << dbits;
        dbits += bits;

        if dbits >= big_digit::BITS {
            data.push(d);
            dbits -= big_digit::BITS;
            // if dbits was > big_digit::BITS, we dropped some of the bits in c (they couldn't fit
            // in d) - grab the bits we lost here:
            d = (c as BigDigit) >> (bits - dbits);
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
    let radix = radix as BigDigit;

    let r = v.len() % power;
    let i = if r == 0 {
        power
    } else {
        r
    };
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
        let mut s = s;
        if s.starts_with('+') {
            let tail = &s[1..];
            if !tail.starts_with('+') {
                s = tail
            }
        }

        if s.is_empty() {
            // create ParseIntError::Empty
            let e = u64::from_str_radix(s, radix).unwrap_err();
            return Err(e.into());
        }

        // First normalize all characters to plain digit values
        let mut v = Vec::with_capacity(s.len());
        for b in s.bytes() {
            let d = match b {
                b'0'...b'9' => b - b'0',
                b'a'...b'z' => b - b'a' + 10,
                b'A'...b'Z' => b - b'A' + 10,
                _ => u8::MAX,
            };
            if d < radix as u8 {
                v.push(d);
            } else {
                // create ParseIntError::InvalidDigit
                // Include the previous character for context.
                let i = cmp::max(v.len(), 1) - 1;
                let e = u64::from_str_radix(&s[i..], radix).unwrap_err();
                return Err(e.into());
            }
        }

        let res = if radix.is_power_of_two() {
            // Powers of two can use bitwise masks and shifting instead of multiplication
            let bits = ilog2(radix);
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
    fn shl(self, rhs: usize) -> BigUint {
        biguint_shl(Cow::Owned(self), rhs)
    }
}

impl<'a> Shl<usize> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn shl(self, rhs: usize) -> BigUint {
        biguint_shl(Cow::Borrowed(self), rhs)
    }
}

impl Shr<usize> for BigUint {
    type Output = BigUint;

    #[inline]
    fn shr(self, rhs: usize) -> BigUint {
        biguint_shr(Cow::Owned(self), rhs)
    }
}

impl<'a> Shr<usize> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn shr(self, rhs: usize) -> BigUint {
        biguint_shr(Cow::Borrowed(self), rhs)
    }
}

impl Zero for BigUint {
    #[inline]
    fn zero() -> BigUint {
        BigUint::new(Vec::new())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

impl One for BigUint {
    #[inline]
    fn one() -> BigUint {
        BigUint::new(vec![1])
    }
}

impl Unsigned for BigUint {}

forward_all_binop_to_val_ref_commutative!(impl Add for BigUint, add);

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

forward_val_val_binop!(impl Sub for BigUint, sub);
forward_ref_ref_binop!(impl Sub for BigUint, sub);

impl<'a> Sub<&'a BigUint> for BigUint {
    type Output = BigUint;

    fn sub(mut self, other: &BigUint) -> BigUint {
        sub2(&mut self.data[..], &other.data[..]);
        self.normalize()
    }
}

impl<'a> Sub<BigUint> for &'a BigUint {
    type Output = BigUint;

    fn sub(self, mut other: BigUint) -> BigUint {
        if other.data.len() < self.data.len() {
            let extra = self.data.len() - other.data.len();
            other.data.extend(repeat(0).take(extra));
        }

        sub2rev(&self.data[..], &mut other.data[..]);
        other.normalize()
    }
}

forward_all_binop_to_ref_ref!(impl Mul for BigUint, mul);

impl<'a, 'b> Mul<&'b BigUint> for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn mul(self, other: &BigUint) -> BigUint {
        mul3(&self.data[..], &other.data[..])
    }
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
    fn neg(self) -> BigUint {
        panic!()
    }
}

impl<'a> Neg for &'a BigUint {
    type Output = BigUint;

    #[inline]
    fn neg(self) -> BigUint {
        panic!()
    }
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
        div_rem(self, other)
    }

    #[inline]
    fn div_floor(&self, other: &BigUint) -> BigUint {
        let (d, _) = div_rem(self, other);
        d
    }

    #[inline]
    fn mod_floor(&self, other: &BigUint) -> BigUint {
        let (_, m) = div_rem(self, other);
        m
    }

    #[inline]
    fn div_mod_floor(&self, other: &BigUint) -> (BigUint, BigUint) {
        div_rem(self, other)
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
    fn lcm(&self, other: &BigUint) -> BigUint {
        ((self * other) / self.gcd(other))
    }

    /// Deprecated, use `is_multiple_of` instead.
    #[inline]
    fn divides(&self, other: &BigUint) -> bool {
        self.is_multiple_of(other)
    }

    /// Returns `true` if the number is a multiple of `other`.
    #[inline]
    fn is_multiple_of(&self, other: &BigUint) -> bool {
        (self % other).is_zero()
    }

    /// Returns `true` if the number is divisible by `2`.
    #[inline]
    fn is_even(&self) -> bool {
        // Considering only the last digit.
        match self.data.first() {
            Some(x) => x.is_even(),
            None => true,
        }
    }

    /// Returns `true` if the number is not divisible by `2`.
    #[inline]
    fn is_odd(&self) -> bool {
        !self.is_even()
    }
}

fn high_bits_to_u64(v: &BigUint) -> u64 {
    match v.data.len() {
        0   => 0,
        1   => v.data[0] as u64,
        _   => {
            let mut bits = v.bits();
            let mut ret = 0u64;
            let mut ret_bits = 0;

            for d in v.data.iter().rev() {
                let digit_bits = (bits - 1) % big_digit::BITS + 1;
                let bits_want = cmp::min(64 - ret_bits, digit_bits);

                if bits_want != 64 {
                    ret <<= bits_want;
                }
                ret      |= *d as u64 >> (digit_bits - bits_want);
                ret_bits += bits_want;
                bits     -= bits_want;

                if ret_bits == 64 {
                    break;
                }
            }

            ret
        }
    }
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

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        let mut ret: u64 = 0;
        let mut bits = 0;

        for i in self.data.iter() {
            if bits >= 64 {
                return None;
            }

            ret += (*i as u64) << bits;
            bits += big_digit::BITS;
        }

        Some(ret)
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        let mantissa = high_bits_to_u64(self);
        let exponent = self.bits() - fls(mantissa);

        if exponent > f32::MAX_EXP as usize {
            None
        } else {
            let ret = (mantissa as f32) * 2.0f32.powi(exponent as i32);
            if ret.is_infinite() {
                None
            } else {
                Some(ret)
            }
        }
    }

    #[inline]
    fn to_f64(&self) -> Option<f64> {
        let mantissa = high_bits_to_u64(self);
        let exponent = self.bits() - fls(mantissa);

        if exponent > f64::MAX_EXP as usize {
            None
        } else {
            let ret = (mantissa as f64) * 2.0f64.powi(exponent as i32);
            if ret.is_infinite() {
                None
            } else {
                Some(ret)
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
    #[inline]
    fn from(mut n: u64) -> Self {
        let mut ret: BigUint = Zero::zero();

        while n != 0 {
            ret.data.push(n as BigDigit);
            // don't overflow if BITS is 64:
            n = (n >> 1) >> (big_digit::BITS - 1);
        }

        ret
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

impl_to_biguint!(isize, FromPrimitive::from_isize);
impl_to_biguint!(i8, FromPrimitive::from_i8);
impl_to_biguint!(i16, FromPrimitive::from_i16);
impl_to_biguint!(i32, FromPrimitive::from_i32);
impl_to_biguint!(i64, FromPrimitive::from_i64);
impl_to_biguint!(usize, FromPrimitive::from_usize);
impl_to_biguint!(u8, FromPrimitive::from_u8);
impl_to_biguint!(u16, FromPrimitive::from_u16);
impl_to_biguint!(u32, FromPrimitive::from_u32);
impl_to_biguint!(u64, FromPrimitive::from_u64);
impl_to_biguint!(f32, FromPrimitive::from_f32);
impl_to_biguint!(f64, FromPrimitive::from_f64);

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

    let mask: BigDigit = (1 << bits) - 1;
    let digits = (u.bits() + bits - 1) / bits;
    let mut res = Vec::with_capacity(digits);

    let mut r = 0;
    let mut rbits = 0;

    for c in &u.data {
        r |= *c << rbits;
        rbits += big_digit::BITS;

        while rbits >= bits {
            res.push((r & mask) as u8);
            r >>= bits;

            // r had more bits than it could fit - grab the bits we lost
            if rbits > big_digit::BITS {
                r = *c >> (big_digit::BITS - (rbits - bits));
            }

            rbits -= bits;
        }
    }

    if rbits != 0 {
        res.push(r as u8);
    }

    while let Some(&0) = res.last() {
        res.pop();
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
    let radix = radix as BigDigit;

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

pub fn to_str_radix_reversed(u: &BigUint, radix: u32) -> Vec<u8> {
    assert!(2 <= radix && radix <= 36, "The radix must be within 2...36");

    if u.is_zero() {
        return vec![b'0'];
    }

    let mut res = if radix.is_power_of_two() {
        // Powers of two can use bitwise masks and shifting instead of division
        let bits = ilog2(radix);
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
        debug_assert!((*r as u32) < radix);
        if *r < 10 {
            *r += b'0';
        } else {
            *r += b'a' - 10;
        }
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
    /// use num_bigint::BigUint;
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
    /// use num_bigint::BigUint;
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
    /// use num_bigint::BigUint;
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
    /// use num_bigint::BigUint;
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
    /// use num_bigint::{BigUint, ToBigUint};
    ///
    /// assert_eq!(BigUint::parse_bytes(b"1234", 10), ToBigUint::to_biguint(&1234));
    /// assert_eq!(BigUint::parse_bytes(b"ABCD", 16), ToBigUint::to_biguint(&0xABCD));
    /// assert_eq!(BigUint::parse_bytes(b"G", 16), None);
    /// ```
    #[inline]
    pub fn parse_bytes(buf: &[u8], radix: u32) -> Option<BigUint> {
        str::from_utf8(buf).ok().and_then(|s| BigUint::from_str_radix(s, radix).ok())
    }

    /// Determines the fewest bits necessary to express the `BigUint`.
    #[inline]
    pub fn bits(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        let zeros = self.data.last().unwrap().leading_zeros();
        return self.data.len() * big_digit::BITS - zeros as usize;
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

#[cfg(feature = "serde")]
impl serde::Serialize for BigUint {
    fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
        where S: serde::Serializer
    {
        self.data.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl serde::Deserialize for BigUint {
    fn deserialize<D>(deserializer: &mut D) -> Result<Self, D::Error>
        where D: serde::Deserializer
    {
        let data = try!(Vec::deserialize(deserializer));
        Ok(BigUint { data: data })
    }
}

/// Returns the greatest power of the radix <= big_digit::BASE
#[inline]
fn get_radix_base(radix: u32) -> (BigDigit, usize) {
    debug_assert!(2 <= radix && radix <= 36, "The radix must be within 2...36");
    debug_assert!(!radix.is_power_of_two());

    // To generate this table:
    //    for radix in 2u64..37 {
    //        let mut power = big_digit::BITS / fls(radix as u64);
    //        let mut base = radix.pow(power as u32);
    //
    //        while let Some(b) = base.checked_mul(radix) {
    //            if b > big_digit::MAX {
    //                break;
    //            }
    //            base = b;
    //            power += 1;
    //        }
    //
    //        println!("({:10}, {:2}), // {:2}", base, power, radix);
    //    }

    match big_digit::BITS {
        32  => {
            const BASES: [(u32, usize); 37] = [(0, 0), (0, 0),
                (0,                     0), // 2
                (3486784401,            20),// 3
                (0,                     0), // 4
                (1220703125,            13),// 5
                (2176782336,            12),// 6
                (1977326743,            11),// 7
                (0,                     0), // 8
                (3486784401,            10),// 9
                (1000000000,            9), // 10
                (2357947691,            9), // 11
                (429981696,             8), // 12
                (815730721,             8), // 13
                (1475789056,            8), // 14
                (2562890625,            8), // 15
                (0,                     0), // 16
                (410338673,             7), // 17
                (612220032,             7), // 18
                (893871739,             7), // 19
                (1280000000,            7), // 20
                (1801088541,            7), // 21
                (2494357888,            7), // 22
                (3404825447,            7), // 23
                (191102976,             6), // 24
                (244140625,             6), // 25
                (308915776,             6), // 26
                (387420489,             6), // 27
                (481890304,             6), // 28
                (594823321,             6), // 29
                (729000000,             6), // 30
                (887503681,             6), // 31
                (0,                     0), // 32
                (1291467969,            6), // 33
                (1544804416,            6), // 34
                (1838265625,            6), // 35
                (2176782336,            6)  // 36
            ];

            let (base, power) = BASES[radix as usize];
            (base as BigDigit, power)
        }
        64  => {
            const BASES: [(u64, usize); 37] = [(0, 0), (0, 0),
                (9223372036854775808,	63), //  2
                (12157665459056928801,	40), //  3
                (4611686018427387904,	31), //  4
                (7450580596923828125,	27), //  5
                (4738381338321616896,	24), //  6
                (3909821048582988049,	22), //  7
                (9223372036854775808,	21), //  8
                (12157665459056928801,	20), //  9
                (10000000000000000000,	19), // 10
                (5559917313492231481,	18), // 11
                (2218611106740436992,	17), // 12
                (8650415919381337933,	17), // 13
                (2177953337809371136,	16), // 14
                (6568408355712890625,	16), // 15
                (1152921504606846976,	15), // 16
                (2862423051509815793,	15), // 17
                (6746640616477458432,	15), // 18
                (15181127029874798299,	15), // 19
                (1638400000000000000,	14), // 20
                (3243919932521508681,	14), // 21
                (6221821273427820544,	14), // 22
                (11592836324538749809,	14), // 23
                (876488338465357824,	13), // 24
                (1490116119384765625,	13), // 25
                (2481152873203736576,	13), // 26
                (4052555153018976267,	13), // 27
                (6502111422497947648,	13), // 28
                (10260628712958602189,	13), // 29
                (15943230000000000000,	13), // 30
                (787662783788549761,	12), // 31
                (1152921504606846976,	12), // 32
                (1667889514952984961,	12), // 33
                (2386420683693101056,	12), // 34
                (3379220508056640625,	12), // 35
                (4738381338321616896,	12), // 36
            ];

            let (base, power) = BASES[radix as usize];
            (base as BigDigit, power)
        }
        _   => panic!("Invalid bigdigit size")
    }
}
