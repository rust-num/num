//! Rust's integer types are defined to overflow with two's complement by default if checking is not
//! enabled (see https://github.com/rust-lang/rfcs/blob/master/text/0560-integer-overflow.md). To
//! intentionally cause wrappng the type `Wrapping<T>` is used.
//!
//! This module contains a `Saturating<T>` which causes the contained type to saturate its value
//! instead of overflowing, according to two's complement.
//!
//! * `+`, `-` and `*` saturate to `MAX` and `MIN` value for both signed and unsigned integers.
//!
//! * `/` and `%` cannot overflow for unsigned integers, for signed integers `/` can overflow with
//!   `MIN / -1` since according to two's complement `MIN / -1 = MAX + 1`, this will saturate to
//!   `MAX`. Same goes for `MIN % -1` since according to two's complement `MIN` is always even,
//!   making `MIN % -1 = MAX + 1`.
//!
//! * `<<` and `>>` for unsigned non-zero values saturate to `MAX` and `MIN` respectively, zero
//!   cannot saturate. For signed non-zero positive values saturate to `MAX` and `0` respectively,
//!   signed non-zero negative values saturate to `MIN` and `-1` respectively (carrying the two's
//!   complement sign flag in right shift). Zero cannot saturate.
//!
//! * Bitwise operators (`!`, `^`, `|` and `&`) behave like the wrapped type.
use std::mem::size_of;
use std::ops::{Add, Sub, Mul, Div, Rem, Not, Neg, BitXor, BitOr, BitAnd, Shl, Shr};

use traits::{Bounded, Num, One, Signed, Unsigned, Zero};

/// Provides saturating arithmetic on `T` based on two's complement.
///
/// Saturating arithmetic can be provided either by methods like `saturating_add`, or through the
/// `Saturating<T>` type, which says that all standard arithmetic operations are intended to have
/// saturating semantics.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Default)]
pub struct Saturating<T>(pub T);

macro_rules! impl_saturating {
    ( $t:ty ) => {
        impl Add for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn add(self, rhs: Saturating<$t>) -> Self::Output {
                Saturating(self.0.saturating_add(rhs.0))
            }
        }

        impl Sub for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn sub(self, rhs: Saturating<$t>) -> Self::Output {
                Saturating(self.0.saturating_sub(rhs.0))
            }
        }

        impl Not for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn not(self) -> Self::Output {
                Saturating(!self.0)
            }
        }

        impl BitXor for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn bitxor(self, rhs: Saturating<$t>) -> Self::Output {
                Saturating(self.0 ^ rhs.0)
            }
        }

        impl BitOr for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn bitor(self, rhs: Saturating<$t>) -> Self::Output {
                Saturating(self.0 | rhs.0)
            }
        }

        impl BitAnd for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn bitand(self, rhs: Saturating<$t>) -> Self::Output {
                Saturating(self.0 & rhs.0)
            }
        }

        impl Zero for Saturating<$t> {
            #[inline(always)]
            fn zero() -> Self {
                Saturating(0)
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                self.0 == 0
            }
        }

        impl One for Saturating<$t> {
            #[inline(always)]
            fn one() -> Self {
                Saturating(1)
            }
        }

        impl Num for Saturating<$t> {
            type FromStrRadixErr = ::std::num::ParseIntError;

            #[inline(always)]
            fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$t>::from_str_radix(s, radix).map(Saturating)
            }
        }

        impl Bounded for Saturating<$t> {
            #[inline(always)]
            fn min_value() -> Self {
                Saturating(<$t>::min_value())
            }

            #[inline(always)]
            fn max_value() -> Self {
                Saturating(<$t>::min_value())
            }
        }
    };
}

macro_rules! impl_saturating_unsigned {
    ( $t:ty ) => {
        impl_saturating!{$t}

        impl Unsigned for Saturating<$t> {}

        impl Mul for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn mul(self, rhs: Saturating<$t>) -> Self::Output {
                Saturating(self.0.checked_mul(rhs.0).unwrap_or(<$t>::max_value()))
            }
        }

        impl Div for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn div(self, rhs: Saturating<$t>) -> Self::Output {
                // Cannot overflow in unsigned
                Saturating(self.0 / rhs.0)
            }
        }

        impl Rem for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn rem(self, rhs: Saturating<$t>) -> Self::Output {
                // Cannot overflow in unsigned
                Saturating(self.0 % rhs.0)
            }
        }
    };
    ( $($t:ty)* ) => { $(impl_saturating_unsigned!{$t})* };
}

macro_rules! impl_saturating_signed {
    ( $t:ty ) => {
        impl_saturating!{$t}

        impl Mul for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn mul(self, rhs: Saturating<$t>) -> Self::Output {
                Saturating(self.0.checked_mul(rhs.0).unwrap_or_else(||
                    if self.0 < 0 && rhs.0 < 0 || self.0 > 0 && rhs.0 > 0 {
                        <$t>::max_value()
                    } else {
                        <$t>::min_value()
                    }))
            }
        }

        impl Div for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn div(self, rhs: Saturating<$t>) -> Self::Output {
                // Can overflow since MIN / -1 = MAX + 1
                if self.0 == <$t>::min_value() && rhs.0 == -1 {
                    Saturating(<$t>::max_value())
                } else {
                    Saturating(self.0 / rhs.0)
                }
            }
        }

        impl Rem for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn rem(self, rhs: Saturating<$t>) -> Self::Output {
                // Overflow when MIN % -1 = MAX + 1 (since MIN is even in two's complement)
                if self.0 == <$t>::min_value() && rhs.0 == -1 {
                    Saturating(<$t>::max_value())
                } else {
                    Saturating(self.0 % rhs.0)
                }
            }
        }

        impl Neg for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                // -MIN = MAX + 1 according to two's complement
                if self.0 == <$t>::min_value() {
                    Saturating(<$t>::max_value())
                } else {
                    Saturating(-self.0)
                }
            }
        }

        impl Signed for Saturating<$t> {
            #[inline(always)]
            fn abs(&self) -> Self {
                // According to trait-documentation abs(::MIN) -> ::MIN
                if self.0 == <$t>::min_value() {
                    Saturating(self.0)
                }
                else if self.is_negative() {
                    -*self
                }
                else {
                    *self
                }
            }

            #[inline(always)]
            fn abs_sub(&self, other: &Self) -> Self {
                if *self <= *other { Saturating(0) } else { *self - *other }
            }

            #[inline(always)]
            fn signum(&self) -> Self {
                match self.0 {
                    n if n > 0 => Saturating(1),
                    0          => Saturating(0),
                    _          => Saturating(-1),
                }
            }

            #[inline(always)]
            fn is_positive(&self) -> bool {
                self.0 > 0
            }

            #[inline(always)]
            fn is_negative(&self) -> bool {
                self.0 < 0
            }
        }
    };
    ( $($t:ty)* ) => { $(impl_saturating_signed!{$t})* };
}

macro_rules! impl_saturating_sh_unsigned {
    ( $t:ty, $bits:expr, $f:ty ) => {
        impl Shl<$f> for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn shl(self, rhs: $f) -> Self::Output {
                // Shifting zero any numer of bits is still zero
                if self.0 == 0 {
                    self
                }
                else if rhs > self.0.leading_zeros() as $f {
                    Saturating(<$t>::max_value())
                }
                else {
                    Saturating(self.0 << rhs)
                }
            }
        }

        impl Shr<$f> for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn shr(self, rhs: $f) -> Self::Output {
                // Fine for unsigned, 0 is smallest value
                if rhs > $bits - 1 {
                    Saturating(<$t>::min_value())
                }
                else {
                    Saturating(self.0 >> rhs)
                }
            }
        }
    };
    ( $t:ty, $bits:expr, ( $($f:ty)* ) ) => {
        $(impl_saturating_shl_unsigned!{$t, $bits, $f})*
    };
}

macro_rules! impl_saturating_sh_signed {
    ( $t:ty, $bits:expr, $f:ty, $unsigned:ty ) => {
        impl Shl<$f> for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn shl(self, rhs: $f) -> Self::Output {
                /// Returns the number of leading zeros of a signed number excluding the sign-bit.
                fn signed_leading_zeros(n: $t) -> $f {
                    debug_assert!(n != 0);

                    // According to two's complement representation, we can get the positive
                    // representation by negating and adding 1 in the case of the sign bit being
                    // set. In this case we can just remove 1 from the number of leading zeros
                    // instead:
                    if n > 0 {
                        // sign bit never set, at least one leading zero
                        (n.leading_zeros() - 1) as $f
                    } else {
                        // always < 0 here, sign bit always set => leading zeros on bitwise not is
                        // always > 0:
                        ((!n).leading_zeros() - 1) as $f
                    }
                }

                // Shifting zero any numer of bits is still zero
                if self.0 == 0 {
                    self
                }
                // We can only shift left at most the same as the number of leading zeros excluding
                // the sign bit.
                else if rhs > signed_leading_zeros(self.0) {
                    // Use correct saturation
                    Saturating(if self.0 < 0 { <$t>::min_value() } else { <$t>::max_value() })
                }
                else {
                    Saturating(self.0 << rhs)
                }
            }
        }

        impl Shr<$f> for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn shr(self, rhs: $f) -> Self::Output {
                if self.0 == 0 {
                    Saturating(0)
                }
                else if rhs > $bits - 1 {
                    // Negative values always keep sign bit on right shift
                    if self.0 < 0 {
                        Saturating(-1)
                    } else {
                        // Positive values saturate to zero
                        Saturating(0)
                    }
                }
                else {
                    Saturating(self.0 >> rhs)
                }
            }
        }
    };
    ( $t:ty, $bits:expr, ( $($f:ty)* ) ) => {
        $(impl_saturating_shl_unsigned!{$t, $bits, $f})*
    };
}

impl_saturating_unsigned!{u8 u16 u32 u64 usize}
impl_saturating_signed!{i8 i16 i32 i64 isize}
impl_saturating_sh_unsigned!{u8,  8,  usize}
impl_saturating_sh_unsigned!{u16, 16, usize}
impl_saturating_sh_unsigned!{u32, 32, usize}
impl_saturating_sh_unsigned!{u64, 64, usize}
impl_saturating_sh_unsigned!{usize, size_of::<usize>() * 8, usize}
impl_saturating_sh_signed!{i8,  8,  usize, u8}
impl_saturating_sh_signed!{i16, 16, usize, u16}
impl_saturating_sh_signed!{i32, 32, usize, u32}
impl_saturating_sh_signed!{i64, 64, usize, u64}
impl_saturating_sh_signed!{isize, size_of::<isize>() * 8, usize, usize}

#[cfg(test)]
mod test {
    use std::mem::size_of;

    use traits::Bounded;

    use super::Saturating;

    #[test]
    fn signed_div_saturation() {
        macro_rules! test { ( $t:ty ) => {
            assert_eq!(Saturating(<$t>::min_value()) / Saturating(-1), Saturating(<$t>::max_value()));
        } }

        test!(i8);
        test!(i16);
        test!(i32);
        test!(i64);
        test!(isize);
    }

    #[test]
    fn signed_mul_saturation() {
        macro_rules! test { ( $t:ty ) => {
            assert_eq!(Saturating(<$t>::min_value()) * Saturating(-1), Saturating(<$t>::max_value()));
        } }

        test!(i8);
        test!(i16);
        test!(i32);
        test!(i64);
        test!(isize);
    }

    #[test]
    fn unsigned_shl_saturation() {
        macro_rules! test { ( $t:ty, $bits:expr ) => {
            // (MAX / 2) << 2 overflows
            assert_eq!(Saturating(<$t>::max_value() / 2) << 2, Saturating(<$t>::max_value()));
            // MAX << 1 overflows
            assert_eq!(Saturating(<$t>::max_value()) << 1, Saturating(<$t>::max_value()));
            // (MAX - 1) << 1 overflows
            assert_eq!(Saturating(<$t>::max_value() - 1) << 1, Saturating(<$t>::max_value()));
            // 1 << $bits - 1 does not overflow, checking for correctness
            assert_eq!(Saturating(1 as $t) << ($bits - 1) as usize, Saturating(1 << ($bits - 1) as $t));
            // 1 << $bits overflows
            assert_eq!(Saturating(1 as $t) << $bits as usize, Saturating(<$t>::max_value()));
            // 1 << $bits + 1 overflows
            assert_eq!(Saturating(1 as $t) << ($bits + 1) as usize, Saturating(<$t>::max_value()));
            // zero for correctness
            assert_eq!(Saturating(0 as $t) << ($bits - 1) as usize, Saturating(0));
            assert_eq!(Saturating(0 as $t) << $bits as usize, Saturating(0));
            assert_eq!(Saturating(0 as $t) << ($bits + 1) as usize, Saturating(0));

            // Zero shift for correctness:
            assert_eq!(Saturating(<$t>::max_value()) << 0, Saturating(<$t>::max_value()));
            assert_eq!(Saturating(<$t>::min_value()) << 0, Saturating(<$t>::min_value()));
            assert_eq!(Saturating(<$t>::max_value() / 2) << 0, Saturating(<$t>::max_value() / 2));
            assert_eq!(Saturating(<$t>::max_value() - 1) << 0, Saturating(<$t>::max_value() - 1));
        } }

        test!(u8, 8);
        test!(u16, 16);
        test!(u32, 32);
        test!(u64, 64);
        test!(usize, size_of::<usize>() * 8);
    }

    #[test]
    fn unsigned_shr_saturation() {
        macro_rules! test { ( $t:ty, $bits:expr ) => {
            assert_eq!(Saturating(0 as $t) >> $bits as usize, Saturating(0));
            assert_eq!(Saturating(0 as $t) >> ($bits + 1) as usize, Saturating(0));
            // MAX >> 1 = MAX / 2
            assert_eq!(Saturating(<$t>::max_value()) >> 1, Saturating(<$t>::max_value() / 2));
            // MAX >> bits - 2 = 3
            assert_eq!(Saturating(<$t>::max_value()) >> ($bits - 2) as usize, Saturating(3));
            // MAX >> bits - 1 = 1
            assert_eq!(Saturating(<$t>::max_value()) >> ($bits - 1) as usize, Saturating(1));
            // MAX >> bits overflows
            assert_eq!(Saturating(<$t>::max_value()) >> $bits as usize, Saturating(0));
            // MAX >> bits + 1 overflows
            assert_eq!(Saturating(<$t>::max_value()) >> ($bits + 1) as usize, Saturating(0));

            // Zero shift for correctness:
            assert_eq!(Saturating(<$t>::max_value()) >> 0, Saturating(<$t>::max_value()));
            assert_eq!(Saturating(<$t>::min_value()) >> 0, Saturating(<$t>::min_value()));
            assert_eq!(Saturating(<$t>::max_value() / 2) >> 0, Saturating(<$t>::max_value() / 2));
            assert_eq!(Saturating(<$t>::max_value() - 1) >> 0, Saturating(<$t>::max_value() - 1));
        } }

        test!(u8, 8);
        test!(u16, 16);
        test!(u32, 32);
        test!(u64, 64);
        test!(usize, size_of::<usize>() * 8);
    }

    #[test]
    fn signed_shl_saturation() {
        macro_rules! test { ( $t:ty, $bits:expr ) => {
            // (MAX / 2) << 2 overflows
            assert_eq!(Saturating(<$t>::max_value() / 2) << 2, Saturating(<$t>::max_value()));
            // MAX << 1 overflows
            assert_eq!(Saturating(<$t>::max_value()) << 1, Saturating(<$t>::max_value()));
            // (MAX - 1) << 1 overflows
            assert_eq!(Saturating(<$t>::max_value() - 1) << 1, Saturating(<$t>::max_value()));
            // 2 << $bits - 2 does not overflow, checking for correctness
            assert_eq!(Saturating(2 as $t) << ($bits - 3) as usize, Saturating(2 << ($bits - 3) as $t));
            // 2 << $bits - 2 overflows
            assert_eq!(Saturating(2 as $t) << ($bits - 2) as usize, Saturating(<$t>::max_value()));
            // 2 << $bits - 1 overflows
            assert_eq!(Saturating(2 as $t) << ($bits - 1) as usize, Saturating(<$t>::max_value()));
            // 2 << $bits overflows
            assert_eq!(Saturating(2 as $t) << $bits as usize, Saturating(<$t>::max_value()));
            // 2 << $bits + 1 overflows
            assert_eq!(Saturating(2 as $t) << ($bits + 1) as usize, Saturating(<$t>::max_value()));
            // 1 << $bits - 2 does not overflow, checking for correctness
            assert_eq!(Saturating(1 as $t) << ($bits - 2) as usize, Saturating(1 << ($bits - 2) as $t));
            // 1 << $bits - 1 overflows
            assert_eq!(Saturating(1 as $t) << ($bits - 1) as usize, Saturating(<$t>::max_value()));
            // 1 << $bits overflows
            assert_eq!(Saturating(1 as $t) << $bits as usize, Saturating(<$t>::max_value()));
            // 1 << $bits + 1 overflows
            assert_eq!(Saturating(1 as $t) << ($bits + 1) as usize, Saturating(<$t>::max_value()));
            // zero for correctness
            assert_eq!(Saturating(0 as $t) << ($bits - 2) as usize, Saturating(0));
            assert_eq!(Saturating(0 as $t) << ($bits - 1) as usize, Saturating(0));
            assert_eq!(Saturating(0 as $t) << $bits as usize, Saturating(0));
            assert_eq!(Saturating(0 as $t) << ($bits + 1) as usize, Saturating(0));
            // -1 << $bits = MIN
            assert_eq!(Saturating(-1 as $t) << $bits as usize, Saturating(<$t>::min_value()));
            // -2 << $bits - 1 = MIN
            assert_eq!(Saturating(-2 as $t) << ($bits - 1) as usize, Saturating(<$t>::min_value()));
            // -2 << $bits overflows
            assert_eq!(Saturating(-2 as $t) << $bits as usize, Saturating(<$t>::min_value()));
            // MIN << 1 overflows
            assert_eq!(Saturating(<$t>::min_value()) << 1 as usize, Saturating(<$t>::min_value()));
            // MIN / 2 << 2 overflows
            assert_eq!(Saturating(<$t>::min_value() / 2) << 2 as usize, Saturating(<$t>::min_value()));
            // (MIN + 1) << 1 overflows
            assert_eq!(Saturating(<$t>::min_value() + 1) << 1 as usize, Saturating(<$t>::min_value()));

            // Zero shift for correctness:
            assert_eq!(Saturating(<$t>::max_value())     << 0, Saturating(<$t>::max_value()));
            assert_eq!(Saturating(<$t>::min_value())     << 0, Saturating(<$t>::min_value()));
            assert_eq!(Saturating(<$t>::max_value() / 2) << 0, Saturating(<$t>::max_value() / 2));
            assert_eq!(Saturating(<$t>::min_value() / 2) << 0, Saturating(<$t>::min_value() / 2));
            assert_eq!(Saturating(<$t>::max_value() - 1) << 0, Saturating(<$t>::max_value() - 1));
            assert_eq!(Saturating(<$t>::min_value() + 1) << 0, Saturating(<$t>::min_value() + 1));
            assert_eq!(Saturating(-2 as $t)              << 0, Saturating(-2));
            assert_eq!(Saturating(-1 as $t)              << 0, Saturating(-1));
            assert_eq!(Saturating(1 as $t)               << 0, Saturating(1));
            assert_eq!(Saturating(0 as $t)               << 0, Saturating(0));
        } }

        test!(i8, 8);
        test!(i16, 16);
        test!(i32, 32);
        test!(i64, 64);
        test!(isize, size_of::<isize>() * 8);
    }

    #[test]
    fn signed_shr_saturation() {
        macro_rules! test { ( $t:ty, $bits:expr ) => {
            assert_eq!(Saturating(0 as $t) >> $bits as usize, Saturating(0));
            assert_eq!(Saturating(0 as $t) >> ($bits + 1) as usize, Saturating(0));
            // MAX >> 1 = MAX / 2
            assert_eq!(Saturating(<$t>::max_value()) >> 1, Saturating(<$t>::max_value() / 2));
            // MAX >> bits - 3 = 3
            assert_eq!(Saturating(<$t>::max_value()) >> ($bits - 3) as usize, Saturating(3));
            // MAX >> bits - 2 = 1
            assert_eq!(Saturating(<$t>::max_value()) >> ($bits - 2) as usize, Saturating(1));
            // MAX >> bits - 1 overflows
            assert_eq!(Saturating(<$t>::max_value()) >> ($bits - 1) as usize, Saturating(0));
            // MAX >> bits overflows
            assert_eq!(Saturating(<$t>::max_value()) >> $bits as usize, Saturating(0));
            // MAX >> bits + 1 overflows
            assert_eq!(Saturating(<$t>::max_value()) >> ($bits + 1) as usize, Saturating(0));

            assert_eq!(Saturating(1 as $t) >> 1, Saturating(0));
            assert_eq!(Saturating(1 as $t) >> $bits as usize, Saturating(0));
            assert_eq!(Saturating(2 as $t) >> 1, Saturating(1));
            assert_eq!(Saturating(2 as $t) >> 2, Saturating(0));

            assert_eq!(Saturating(<$t>::min_value()) >> 1, Saturating(<$t>::min_value() / 2));
            assert_eq!(Saturating(<$t>::min_value()) >> ($bits - 2) as usize, Saturating(-2));
            assert_eq!(Saturating(<$t>::min_value()) >> ($bits - 1) as usize, Saturating(-1));
            assert_eq!(Saturating(<$t>::min_value()) >> $bits as usize, Saturating(-1));
            assert_eq!(Saturating(<$t>::min_value()) >> ($bits + 1) as usize, Saturating(-1));
            assert_eq!(Saturating(<$t>::min_value() / 2) >> $bits as usize, Saturating(-1));
            assert_eq!(Saturating(-1 as $t)          >> 1, Saturating(-1));
            assert_eq!(Saturating(-1 as $t)          >> 2, Saturating(-1));
            assert_eq!(Saturating(-1 as $t)          >> $bits as usize, Saturating(-1));

            // Zero shift for correctness:
            assert_eq!(Saturating(<$t>::max_value())     >> 0, Saturating(<$t>::max_value()));
            assert_eq!(Saturating(<$t>::min_value())     >> 0, Saturating(<$t>::min_value()));
            assert_eq!(Saturating(<$t>::max_value() / 2) >> 0, Saturating(<$t>::max_value() / 2));
            assert_eq!(Saturating(<$t>::min_value() / 2) >> 0, Saturating(<$t>::min_value() / 2));
            assert_eq!(Saturating(<$t>::max_value() - 1) >> 0, Saturating(<$t>::max_value() - 1));
            assert_eq!(Saturating(<$t>::min_value() + 1) >> 0, Saturating(<$t>::min_value() + 1));
            assert_eq!(Saturating(-2 as $t)              >> 0, Saturating(-2));
            assert_eq!(Saturating(-1 as $t)              >> 0, Saturating(-1));
            assert_eq!(Saturating(1 as $t)               >> 0, Saturating(1));
            assert_eq!(Saturating(0 as $t)               >> 0, Saturating(0));
        } }

        test!(i8, 8);
        test!(i16, 16);
        test!(i32, 32);
        test!(i64, 64);
        test!(isize, size_of::<isize>() * 8);
    }
}
