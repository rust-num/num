// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Numeric traits for generic mathematics

use std::intrinsics;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::{usize, u8, u16, u32, u64};
use std::{isize, i8, i16, i32, i64};
use std::{f32, f64};

/// The base trait for numeric types
pub trait Num: PartialEq + Zero + One
    + Neg<Output = Self> + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self> {}

macro_rules! trait_impl {
    ($name:ident for $($t:ty)*) => ($(
        impl $name for $t {}
    )*)
}

trait_impl!(Num for usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64);

/// Defines an additive identity element for `Self`.
///
/// # Deriving
///
/// This trait can be automatically be derived using `#[deriving(Zero)]`
/// attribute. If you choose to use this, make sure that the laws outlined in
/// the documentation for `Zero::zero` still hold.
pub trait Zero: Add<Self, Output = Self> {
    /// Returns the additive identity element of `Self`, `0`.
    ///
    /// # Laws
    ///
    /// ```{.text}
    /// a + 0 = a       ∀ a ∈ Self
    /// 0 + a = a       ∀ a ∈ Self
    /// ```
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    // FIXME (#5527): This should be an associated constant
    fn zero() -> Self;

    /// Returns `true` if `self` is equal to the additive identity.
    #[inline]
    fn is_zero(&self) -> bool;
}

macro_rules! zero_impl {
    ($t:ty, $v:expr) => {
        impl Zero for $t {
            #[inline]
            fn zero() -> $t { $v }
            #[inline]
            fn is_zero(&self) -> bool { *self == $v }
        }
    }
}

zero_impl!(usize, 0us);
zero_impl!(u8,   0u8);
zero_impl!(u16,  0u16);
zero_impl!(u32,  0u32);
zero_impl!(u64,  0u64);

zero_impl!(isize, 0is);
zero_impl!(i8,  0i8);
zero_impl!(i16, 0i16);
zero_impl!(i32, 0i32);
zero_impl!(i64, 0i64);

zero_impl!(f32, 0.0f32);
zero_impl!(f64, 0.0f64);

/// Defines a multiplicative identity element for `Self`.
pub trait One: Mul<Self, Output = Self> {
    /// Returns the multiplicative identity element of `Self`, `1`.
    ///
    /// # Laws
    ///
    /// ```{.text}
    /// a * 1 = a       ∀ a ∈ Self
    /// 1 * a = a       ∀ a ∈ Self
    /// ```
    ///
    /// # Purity
    ///
    /// This function should return the same result at all times regardless of
    /// external mutable state, for example values stored in TLS or in
    /// `static mut`s.
    // FIXME (#5527): This should be an associated constant
    fn one() -> Self;
}

macro_rules! one_impl {
    ($t:ty, $v:expr) => {
        impl One for $t {
            #[inline]
            fn one() -> $t { $v }
        }
    }
}

one_impl!(usize, 1us);
one_impl!(u8,  1u8);
one_impl!(u16, 1u16);
one_impl!(u32, 1u32);
one_impl!(u64, 1u64);

one_impl!(isize, 1is);
one_impl!(i8,  1i8);
one_impl!(i16, 1i16);
one_impl!(i32, 1i32);
one_impl!(i64, 1i64);

one_impl!(f32, 1.0f32);
one_impl!(f64, 1.0f64);

/// Useful functions for signed numbers (i.e. numbers that can be negative).
pub trait Signed: Num + Neg<Output = Self> {
    /// Computes the absolute value.
    ///
    /// For `f32` and `f64`, `NaN` will be returned if the number is `NaN`.
    ///
    /// For signed integers, `::MIN` will be returned if the number is `::MIN`.
    fn abs(&self) -> Self;

    /// The positive difference of two numbers.
    ///
    /// Returns `zero` if the number is less than or equal to `other`, otherwise the difference
    /// between `self` and `other` is returned.
    fn abs_sub(&self, other: &Self) -> Self;

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
    fn signum(&self) -> Self;

    /// Returns true if the number is positive and false if the number is zero or negative.
    fn is_positive(&self) -> bool;

    /// Returns true if the number is negative and false if the number is zero or positive.
    fn is_negative(&self) -> bool;
}

macro_rules! signed_impl {
    ($($t:ty)*) => ($(
        impl Signed for $t {
            #[inline]
            fn abs(&self) -> $t {
                if self.is_negative() { -*self } else { *self }
            }

            #[inline]
            fn abs_sub(&self, other: &$t) -> $t {
                if *self <= *other { 0 } else { *self - *other }
            }

            #[inline]
            fn signum(&self) -> $t {
                match *self {
                    n if n > 0 => 1,
                    0 => 0,
                    _ => -1,
                }
            }

            #[inline]
            fn is_positive(&self) -> bool { *self > 0 }

            #[inline]
            fn is_negative(&self) -> bool { *self < 0 }
        }
    )*)
}

signed_impl!(isize i8 i16 i32 i64);

macro_rules! signed_float_impl {
    ($t:ty, $nan:expr, $inf:expr, $neg_inf:expr, $fabs:path, $fcopysign:path, $fdim:ident) => {
        impl Signed for $t {
            /// Computes the absolute value. Returns `NAN` if the number is `NAN`.
            #[inline]
            fn abs(&self) -> $t {
                unsafe { $fabs(*self) }
            }

            /// The positive difference of two numbers. Returns `0.0` if the number is
            /// less than or equal to `other`, otherwise the difference between`self`
            /// and `other` is returned.
            #[inline]
            fn abs_sub(&self, other: &$t) -> $t {
                extern { fn $fdim(a: $t, b: $t) -> $t; }
                unsafe { $fdim(*self, *other) }
            }

            /// # Returns
            ///
            /// - `1.0` if the number is positive, `+0.0` or `INFINITY`
            /// - `-1.0` if the number is negative, `-0.0` or `NEG_INFINITY`
            /// - `NAN` if the number is NaN
            #[inline]
            fn signum(&self) -> $t {
                if self != self { $nan } else {
                    unsafe { $fcopysign(1.0, *self) }
                }
            }

            /// Returns `true` if the number is positive, including `+0.0` and `INFINITY`
            #[inline]
            fn is_positive(&self) -> bool { *self > 0.0 || (1.0 / *self) == $inf }

            /// Returns `true` if the number is negative, including `-0.0` and `NEG_INFINITY`
            #[inline]
            fn is_negative(&self) -> bool { *self < 0.0 || (1.0 / *self) == $neg_inf }
        }
    }
}

signed_float_impl!(f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY,
                   intrinsics::fabsf32, intrinsics::copysignf32, fdimf);
signed_float_impl!(f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY,
                   intrinsics::fabsf64, intrinsics::copysignf64, fdim);

/// A trait for values which cannot be negative
pub trait Unsigned: Num {}

trait_impl!(Unsigned for usize u8 u16 u32 u64);

/// Numbers which have upper and lower bounds
pub trait Bounded {
    // FIXME (#5527): These should be associated constants
    /// returns the smallest finite number this type can represent
    fn min_value() -> Self;
    /// returns the largest finite number this type can represent
    fn max_value() -> Self;
}

macro_rules! bounded_impl {
    ($t:ty, $min:expr, $max:expr) => {
        impl Bounded for $t {
            #[inline]
            fn min_value() -> $t { $min }

            #[inline]
            fn max_value() -> $t { $max }
        }
    }
}

bounded_impl!(usize, usize::MIN, usize::MAX);
bounded_impl!(u8, u8::MIN, u8::MAX);
bounded_impl!(u16, u16::MIN, u16::MAX);
bounded_impl!(u32, u32::MIN, u32::MAX);
bounded_impl!(u64, u64::MIN, u64::MAX);

bounded_impl!(isize, isize::MIN, isize::MAX);
bounded_impl!(i8, i8::MIN, i8::MAX);
bounded_impl!(i16, i16::MIN, i16::MAX);
bounded_impl!(i32, i32::MIN, i32::MAX);
bounded_impl!(i64, i64::MIN, i64::MAX);

bounded_impl!(f32, f32::MIN, f32::MAX);
bounded_impl!(f64, f64::MIN, f64::MAX);

/// Saturating math operations
pub trait Saturating {
    /// Saturating addition operator.
    /// Returns a+b, saturating at the numeric bounds instead of overflowing.
    fn saturating_add(self, v: Self) -> Self;

    /// Saturating subtraction operator.
    /// Returns a-b, saturating at the numeric bounds instead of overflowing.
    fn saturating_sub(self, v: Self) -> Self;
}

impl<T: CheckedAdd + CheckedSub + Zero + PartialOrd + Bounded> Saturating for T {
    #[inline]
    fn saturating_add(self, v: T) -> T {
        match self.checked_add(&v) {
            Some(x) => x,
            None => if v >= Zero::zero() {
                Bounded::max_value()
            } else {
                Bounded::min_value()
            }
        }
    }

    #[inline]
    fn saturating_sub(self, v: T) -> T {
        match self.checked_sub(&v) {
            Some(x) => x,
            None => if v >= Zero::zero() {
                Bounded::min_value()
            } else {
                Bounded::max_value()
            }
        }
    }
}

/// Performs addition that returns `None` instead of wrapping around on overflow.
pub trait CheckedAdd: Add<Self, Output = Self> {
    /// Adds two numbers, checking for overflow. If overflow happens, `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use num::CheckedAdd;
    /// assert_eq!(5u16.checked_add(&65530), Some(65535));
    /// assert_eq!(6u16.checked_add(&65530), None);
    /// ```
    fn checked_add(&self, v: &Self) -> Option<Self>;
}

macro_rules! checked_impl {
    ($trait_name:ident, $method:ident, $t:ty, $op:path) => {
        impl $trait_name for $t {
            #[inline]
            fn $method(&self, v: &$t) -> Option<$t> {
                unsafe {
                    let (x, y) = $op(*self, *v);
                    if y { None } else { Some(x) }
                }
            }
        }
    }
}
macro_rules! checked_cast_impl {
    ($trait_name:ident, $method:ident, $t:ty, $cast:ty, $op:path) => {
        impl $trait_name for $t {
            #[inline]
            fn $method(&self, v: &$t) -> Option<$t> {
                unsafe {
                    let (x, y) = $op(*self as $cast, *v as $cast);
                    if y { None } else { Some(x as $t) }
                }
            }
        }
    }
}

#[cfg(target_pointer_width = "32")]
checked_cast_impl!(CheckedAdd, checked_add, usize, u32, intrinsics::u32_add_with_overflow);
#[cfg(target_pointer_width = "64")]
checked_cast_impl!(CheckedAdd, checked_add, usize, u64, intrinsics::u64_add_with_overflow);

checked_impl!(CheckedAdd, checked_add, u8,  intrinsics::u8_add_with_overflow);
checked_impl!(CheckedAdd, checked_add, u16, intrinsics::u16_add_with_overflow);
checked_impl!(CheckedAdd, checked_add, u32, intrinsics::u32_add_with_overflow);
checked_impl!(CheckedAdd, checked_add, u64, intrinsics::u64_add_with_overflow);

#[cfg(target_pointer_width = "32")]
checked_cast_impl!(CheckedAdd, checked_add, isize, i32, intrinsics::i32_add_with_overflow);
#[cfg(target_pointer_width = "64")]
checked_cast_impl!(CheckedAdd, checked_add, isize, i64, intrinsics::i64_add_with_overflow);

checked_impl!(CheckedAdd, checked_add, i8,  intrinsics::i8_add_with_overflow);
checked_impl!(CheckedAdd, checked_add, i16, intrinsics::i16_add_with_overflow);
checked_impl!(CheckedAdd, checked_add, i32, intrinsics::i32_add_with_overflow);
checked_impl!(CheckedAdd, checked_add, i64, intrinsics::i64_add_with_overflow);

/// Performs subtraction that returns `None` instead of wrapping around on underflow.
pub trait CheckedSub: Sub<Self, Output = Self> {
    /// Subtracts two numbers, checking for underflow. If underflow happens, `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use num::CheckedSub;
    /// assert_eq!((-127i8).checked_sub(&1), Some(-128));
    /// assert_eq!((-128i8).checked_sub(&1), None);
    /// ```
    fn checked_sub(&self, v: &Self) -> Option<Self>;
}

#[cfg(target_pointer_width = "32")]
checked_cast_impl!(CheckedSub, checked_sub, usize, u32, intrinsics::u32_sub_with_overflow);
#[cfg(target_pointer_width = "64")]
checked_cast_impl!(CheckedSub, checked_sub, usize, u64, intrinsics::u64_sub_with_overflow);

checked_impl!(CheckedSub, checked_sub, u8,  intrinsics::u8_sub_with_overflow);
checked_impl!(CheckedSub, checked_sub, u16, intrinsics::u16_sub_with_overflow);
checked_impl!(CheckedSub, checked_sub, u32, intrinsics::u32_sub_with_overflow);
checked_impl!(CheckedSub, checked_sub, u64, intrinsics::u64_sub_with_overflow);

#[cfg(target_pointer_width = "32")]
checked_cast_impl!(CheckedSub, checked_sub, isize, i32, intrinsics::i32_sub_with_overflow);
#[cfg(target_pointer_width = "64")]
checked_cast_impl!(CheckedSub, checked_sub, isize, i64, intrinsics::i64_sub_with_overflow);

checked_impl!(CheckedSub, checked_sub, i8,  intrinsics::i8_sub_with_overflow);
checked_impl!(CheckedSub, checked_sub, i16, intrinsics::i16_sub_with_overflow);
checked_impl!(CheckedSub, checked_sub, i32, intrinsics::i32_sub_with_overflow);
checked_impl!(CheckedSub, checked_sub, i64, intrinsics::i64_sub_with_overflow);

/// Performs multiplication that returns `None` instead of wrapping around on underflow or
/// overflow.
pub trait CheckedMul: Mul<Self, Output = Self> {
    /// Multiplies two numbers, checking for underflow or overflow. If underflow or overflow
    /// happens, `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use num::CheckedMul;
    /// assert_eq!(5u8.checked_mul(&51), Some(255));
    /// assert_eq!(5u8.checked_mul(&52), None);
    /// ```
    fn checked_mul(&self, v: &Self) -> Option<Self>;
}

#[cfg(target_pointer_width = "32")]
checked_cast_impl!(CheckedMul, checked_mul, usize, u32, intrinsics::u32_mul_with_overflow);
#[cfg(target_pointer_width = "64")]
checked_cast_impl!(CheckedMul, checked_mul, usize, u64, intrinsics::u64_mul_with_overflow);

checked_impl!(CheckedMul, checked_mul, u8,  intrinsics::u8_mul_with_overflow);
checked_impl!(CheckedMul, checked_mul, u16, intrinsics::u16_mul_with_overflow);
checked_impl!(CheckedMul, checked_mul, u32, intrinsics::u32_mul_with_overflow);
checked_impl!(CheckedMul, checked_mul, u64, intrinsics::u64_mul_with_overflow);

#[cfg(target_pointer_width = "32")]
checked_cast_impl!(CheckedMul, checked_mul, isize, i32, intrinsics::i32_mul_with_overflow);
#[cfg(target_pointer_width = "64")]
checked_cast_impl!(CheckedMul, checked_mul, isize, i64, intrinsics::i64_mul_with_overflow);

checked_impl!(CheckedMul, checked_mul, i8,  intrinsics::i8_mul_with_overflow);
checked_impl!(CheckedMul, checked_mul, i16, intrinsics::i16_mul_with_overflow);
checked_impl!(CheckedMul, checked_mul, i32, intrinsics::i32_mul_with_overflow);
checked_impl!(CheckedMul, checked_mul, i64, intrinsics::i64_mul_with_overflow);

/// Performs division that returns `None` instead of panicking on division by zero and instead of
/// wrapping around on underflow and overflow.
pub trait CheckedDiv: Div<Self, Output = Self> {
    /// Divides two numbers, checking for underflow, overflow and division by zero. If any of that
    /// happens, `None` is returned.
    ///
    /// # Example
    ///
    /// ```rust
    /// use num::CheckedDiv;
    /// assert_eq!((-127i8).checked_div(&-1), Some(127));
    /// assert_eq!((-128i8).checked_div(&-1), None);
    /// assert_eq!((1i8).checked_div(&0), None);
    /// ```
    fn checked_div(&self, v: &Self) -> Option<Self>;
}

macro_rules! checkeddiv_int_impl {
    ($t:ty, $min:expr) => {
        impl CheckedDiv for $t {
            #[inline]
            fn checked_div(&self, v: &$t) -> Option<$t> {
                if *v == 0 || (*self == $min && *v == -1) {
                    None
                } else {
                    Some(*self / *v)
                }
            }
        }
    }
}

checkeddiv_int_impl!(isize, isize::MIN);
checkeddiv_int_impl!(i8, i8::MIN);
checkeddiv_int_impl!(i16, i16::MIN);
checkeddiv_int_impl!(i32, i32::MIN);
checkeddiv_int_impl!(i64, i64::MIN);

macro_rules! checkeddiv_uint_impl {
    ($($t:ty)*) => ($(
        impl CheckedDiv for $t {
            #[inline]
            fn checked_div(&self, v: &$t) -> Option<$t> {
                if *v == 0 {
                    None
                } else {
                    Some(*self / *v)
                }
            }
        }
    )*)
}

checkeddiv_uint_impl!(usize u8 u16 u32 u64);
