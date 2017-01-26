use std::ops::{Add, Sub, Mul};

macro_rules! wrapping_impl {
    ($trait_name:ident, $method:ident, $t:ty) => {
        impl $trait_name for $t {
            #[inline]
            fn $method(&self, v: &Self) -> Self {
                <$t>::$method(*self, *v)
            }
        }
    };
    ($trait_name:ident, $method:ident, $t:ty, $rhs:ty) => {
        impl $trait_name<$rhs> for $t {
            #[inline]
            fn $method(&self, v: &$rhs) -> Self {
                <$t>::$method(*self, *v)
            }
        }
    }
}

/// Performs addition that wraps around on overflow.
pub trait WrappingAdd: Sized + Add<Self, Output=Self> {
    /// Wrapping (modular) addition. Computes `self + other`, wrapping around at the boundary of
    /// the type.
    fn wrapping_add(&self, v: &Self) -> Self;
}

wrapping_impl!(WrappingAdd, wrapping_add, u8);
wrapping_impl!(WrappingAdd, wrapping_add, u16);
wrapping_impl!(WrappingAdd, wrapping_add, u32);
wrapping_impl!(WrappingAdd, wrapping_add, u64);
wrapping_impl!(WrappingAdd, wrapping_add, usize);

wrapping_impl!(WrappingAdd, wrapping_add, i8);
wrapping_impl!(WrappingAdd, wrapping_add, i16);
wrapping_impl!(WrappingAdd, wrapping_add, i32);
wrapping_impl!(WrappingAdd, wrapping_add, i64);
wrapping_impl!(WrappingAdd, wrapping_add, isize);

/// Performs subtraction that wraps around on overflow.
pub trait WrappingSub: Sized + Sub<Self, Output=Self> {
    /// Wrapping (modular) subtraction. Computes `self - other`, wrapping around at the boundary
    /// of the type.
    fn wrapping_sub(&self, v: &Self) -> Self;
}

wrapping_impl!(WrappingSub, wrapping_sub, u8);
wrapping_impl!(WrappingSub, wrapping_sub, u16);
wrapping_impl!(WrappingSub, wrapping_sub, u32);
wrapping_impl!(WrappingSub, wrapping_sub, u64);
wrapping_impl!(WrappingSub, wrapping_sub, usize);

wrapping_impl!(WrappingSub, wrapping_sub, i8);
wrapping_impl!(WrappingSub, wrapping_sub, i16);
wrapping_impl!(WrappingSub, wrapping_sub, i32);
wrapping_impl!(WrappingSub, wrapping_sub, i64);
wrapping_impl!(WrappingSub, wrapping_sub, isize);

/// Performs multiplication that wraps around on overflow.
pub trait WrappingMul: Sized + Mul<Self, Output=Self> {
    /// Wrapping (modular) multiplication. Computes `self * other`, wrapping around at the boundary
    /// of the type.
    fn wrapping_mul(&self, v: &Self) -> Self;
}

wrapping_impl!(WrappingMul, wrapping_mul, u8);
wrapping_impl!(WrappingMul, wrapping_mul, u16);
wrapping_impl!(WrappingMul, wrapping_mul, u32);
wrapping_impl!(WrappingMul, wrapping_mul, u64);
wrapping_impl!(WrappingMul, wrapping_mul, usize);

wrapping_impl!(WrappingMul, wrapping_mul, i8);
wrapping_impl!(WrappingMul, wrapping_mul, i16);
wrapping_impl!(WrappingMul, wrapping_mul, i32);
wrapping_impl!(WrappingMul, wrapping_mul, i64);
wrapping_impl!(WrappingMul, wrapping_mul, isize);
