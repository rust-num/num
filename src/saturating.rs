use std::mem::size_of;
use std::ops::{Add, Sub, Mul, Div, Rem, Not, Neg, BitXor, BitOr, BitAnd, Shl, Shr};

use traits::{Bounded, Num, One, Signed, Unsigned, Zero};
use traits::Saturating as SaturatingOps;

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
                if self.0 == <$t>::min_value() && rhs.0 == -1 {
                    Saturating(self.0)
                } else {
                    Saturating(self.0 % rhs.0)
                }
            }
        }

        impl Neg for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn neg(self) -> Self::Output {
                // Negating minimum causes overflow
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
                if self.0 == 0 {
                    Saturating(0)
                }
                else if rhs > $bits - 1 {
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
    ( $t:ty, $bits:expr, $f:ty ) => {
        impl Shl<$f> for Saturating<$t> {
            type Output = Saturating<$t>;

            #[inline(always)]
            fn shl(self, rhs: $f) -> Self::Output {
                if self.0 == 0 {
                    Saturating(0)
                }
                // sign bit is kept when shifting left at most $bits - 1 on negative number
                else if self.0 < 0 && rhs > $bits - 1 {
                    Saturating(<$t>::min_value())
                }
                // shifting $bits - 1 results in negative number, only allow $bits - 2 shifts on
                // nonnegative numbers
                else if rhs > $bits - 2 {
                    Saturating(<$t>::max_value())
                } else {
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
                // Negative values always keep sign bit on right shift
                else if self.0 < 0 && rhs > $bits - 1 {
                    Saturating(-1)
                }
                // Positive values saturate to zero
                else if rhs > $bits - 1 {
                    Saturating(0)
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
impl_saturating_sh_unsigned!{u8, 8, usize}
impl_saturating_sh_unsigned!{u16, 16, usize}
impl_saturating_sh_unsigned!{u32, 16, usize}
impl_saturating_sh_unsigned!{u64, 16, usize}
impl_saturating_sh_unsigned!{usize, size_of::<usize>() * 8, usize}
impl_saturating_sh_signed!{i8, 8, usize}
impl_saturating_sh_signed!{i16, 16, usize}
impl_saturating_sh_signed!{i32, 16, usize}
impl_saturating_sh_signed!{i64, 16, usize}
impl_saturating_sh_signed!{isize, size_of::<isize>() * 8, usize}

#[cfg(test)]
mod test {
    use super::Saturating;

    use traits::Bounded;

    macro_rules! tests {
        ( $($t:ty)* ) => { $(test!{$t})* };
    }

    #[test]
    fn signed_div_overflow() {
        macro_rules! test { ( $t:ty ) => {
            assert_eq!(Saturating(<$t>::min_value()) / Saturating(-1), Saturating(<$t>::max_value()));
        } }

        tests!(i8 i16 i32 i64 isize);
    }
}
