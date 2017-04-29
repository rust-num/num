use std::num::Wrapping;

/// Saturating math operations
pub trait Saturating {
    /// Saturating addition operator.
    /// Returns a+b, saturating at the numeric bounds instead of overflowing.
    fn saturating_add(self, v: Self) -> Self;

    /// Saturating subtraction operator.
    /// Returns a-b, saturating at the numeric bounds instead of overflowing.
    fn saturating_sub(self, v: Self) -> Self;
}

macro_rules! saturating_impl {
    ($trait_name:ident for $($t:ty)*) => {$(
        impl $trait_name for $t {
            #[inline]
            fn saturating_add(self, v: Self) -> Self {
                Self::saturating_add(self, v)
            }

            #[inline]
            fn saturating_sub(self, v: Self) -> Self {
                Self::saturating_sub(self, v)
            }
        }
    )*}
}

saturating_impl!(Saturating for isize usize i8 u8 i16 u16 i32 u32 i64 u64);

impl<T: Saturating> Saturating for Wrapping<T> {
    fn saturating_add(self, v: Self) -> Self { Wrapping(self.0.saturating_add(v.0)) }
    fn saturating_sub(self, v: Self) -> Self { Wrapping(self.0.saturating_sub(v.0)) }
}
