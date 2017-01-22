use std::ops::{Add, Sub, Mul, Div, Shl, Shr};

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

/// Performs addition that wrapps around on overflow.
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

/// Performs subtraction that wrapps around on overflow.
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

/// Performs multiplication that wrapps around on overflow.
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

/// Performs division that wrapps around on overflow.
pub trait WrappingDiv: Sized + Div<Self, Output=Self> {
    /// Wrapping (modular) division. Computes `self / other`, wrapping around at the boundary of
    /// the type.
    ///
    /// The only case where such wrapping can occur is when one divides `MIN / -1` on a signed type
    /// (where `MIN` is the negative minimal value for the type); this is equivalent to `-MIN`, a
    /// positive value that is too large to represent in the type. In such a case, this function
    /// returns `MIN` itself.
    ///
    /// # Panics
    ///
    /// This function will panic if rhs is 0.
    fn wrapping_div(&self, v: &Self) -> Self;
}

wrapping_impl!(WrappingDiv, wrapping_div, u8);
wrapping_impl!(WrappingDiv, wrapping_div, u16);
wrapping_impl!(WrappingDiv, wrapping_div, u32);
wrapping_impl!(WrappingDiv, wrapping_div, u64);
wrapping_impl!(WrappingDiv, wrapping_div, usize);

wrapping_impl!(WrappingDiv, wrapping_div, i8);
wrapping_impl!(WrappingDiv, wrapping_div, i16);
wrapping_impl!(WrappingDiv, wrapping_div, i32);
wrapping_impl!(WrappingDiv, wrapping_div, i64);
wrapping_impl!(WrappingDiv, wrapping_div, isize);

/// Performs bitwise shift left that wrapps around on overflow.
pub trait WrappingShl<RHS>: Sized + Shl<RHS, Output=Self> {
    /// Panic-free bitwise shift-left; yields `self << mask(rhs)`, where `mask` removes any
    /// high-order bits of rhs that would cause the shift to exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-left; the RHS of a wrapping shift-left is
    /// restricted to the range of the type, rather than the bits shifted out of the LHS being
    /// returned to the other end. The primitive integer types all implement a `rotate_left`
    /// function, which may be what you want instead.
    fn wrapping_shl(&self, v: &RHS) -> Self;
}

wrapping_impl!(WrappingShl, wrapping_shl, u8, u32);
wrapping_impl!(WrappingShl, wrapping_shl, u16, u32);
wrapping_impl!(WrappingShl, wrapping_shl, u32, u32);
wrapping_impl!(WrappingShl, wrapping_shl, u64, u32);
wrapping_impl!(WrappingShl, wrapping_shl, usize, u32);

wrapping_impl!(WrappingShl, wrapping_shl, i8, u32);
wrapping_impl!(WrappingShl, wrapping_shl, i16, u32);
wrapping_impl!(WrappingShl, wrapping_shl, i32, u32);
wrapping_impl!(WrappingShl, wrapping_shl, i64, u32);
wrapping_impl!(WrappingShl, wrapping_shl, isize, u32);

/// Performs bitwise shift right that wrapps around on overflow.
pub trait WrappingShr<RHS>: Sized + Shr<RHS, Output=Self> {
    /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`, where `mask` removes any
    /// high-order bits of rhs that would cause the shift to exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-right; the RHS of a wrapping shift-right is
    /// restricted to the range of the type, rather than the bits shifted out of the LHS being
    /// returned to the other end. The primitive integer types all implement a `rotate_right`
    /// function, which may be what you want instead.
    fn wrapping_shr(&self, v: &RHS) -> Self;
}

wrapping_impl!(WrappingShr, wrapping_shr, u8, u32);
wrapping_impl!(WrappingShr, wrapping_shr, u16, u32);
wrapping_impl!(WrappingShr, wrapping_shr, u32, u32);
wrapping_impl!(WrappingShr, wrapping_shr, u64, u32);
wrapping_impl!(WrappingShr, wrapping_shr, usize, u32);

wrapping_impl!(WrappingShr, wrapping_shr, i8, u32);
wrapping_impl!(WrappingShr, wrapping_shr, i16, u32);
wrapping_impl!(WrappingShr, wrapping_shr, i32, u32);
wrapping_impl!(WrappingShr, wrapping_shr, i64, u32);
wrapping_impl!(WrappingShr, wrapping_shr, isize, u32);
