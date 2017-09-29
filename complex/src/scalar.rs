
use traits::{Num, FromPrimitive, Float};
use std::ops::Neg;

use Complex;

pub trait Scalar: Num + Copy + Neg<Output = Self> {
    /// Associated Real type
    type Real;
    /// Associated Complex type
    type Complex;

    fn sqrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn abs_sqr(&self) -> Self::Real;
    fn abs(&self) -> Self::Real;

    fn powi(&self, exp: i32) -> Self;
    fn powf(&self, exp: Self::Real) -> Self;
    fn powc(&self, exp: Self::Complex) -> Self::Complex;

    fn conj(&self) -> Self;

    // trigonometric functions
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;

    // hyperbolic functions
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;

    // check normal
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_normal(self) -> bool;
}

impl<T: Clone + Float + FromPrimitive> Scalar for Complex<T> {
    type Real = T;
    type Complex = Self;

    fn sqrt(&self) -> Self {
        Complex::sqrt(self)
    }
    fn exp(&self) -> Self {
        Complex::exp(self)
    }
    fn ln(&self) -> Self {
        Complex::ln(self)
    }
    fn abs_sqr(&self) -> Self::Real {
        Complex::norm_sqr(self)
    }
    fn abs(&self) -> Self::Real {
        Complex::norm(self)
    }

    // powers
    fn powi(&self, exp: i32) -> Self {
        let exp = T::from_i32(exp).unwrap();
        Complex::powf(self, exp)
    }
    fn powf(&self, exp: Self::Real) -> Self {
        Complex::powf(self, exp)
    }
    fn powc(&self, exp: Self::Complex) -> Self::Complex {
        Complex::powc(self, exp)
    }

    fn conj(&self) -> Self {
        Complex::conj(self)
    }

    // trigonometric functions
    fn sin(&self) -> Self {
        Complex::sin(self)
    }
    fn cos(&self) -> Self {
        Complex::cos(self)
    }
    fn tan(&self) -> Self {
        Complex::tan(self)
    }
    fn asin(&self) -> Self {
        Complex::asin(self)
    }
    fn acos(&self) -> Self {
        Complex::acos(self)
    }
    fn atan(&self) -> Self {
        Complex::atan(self)
    }

    // hyperbolic functions
    fn sinh(&self) -> Self {
        Complex::sinh(self)
    }
    fn cosh(&self) -> Self {
        Complex::cosh(self)
    }
    fn tanh(&self) -> Self {
        Complex::tanh(self)
    }
    fn asinh(&self) -> Self {
        Complex::asinh(self)
    }
    fn acosh(&self) -> Self {
        Complex::acosh(self)
    }
    fn atanh(&self) -> Self {
        Complex::atanh(self)
    }

    // check normal
    fn is_nan(self) -> bool {
        Complex::is_nan(self)
    }
    fn is_infinite(self) -> bool {
        Complex::is_infinite(self)
    }
    fn is_finite(self) -> bool {
        Complex::is_finite(self)
    }
    fn is_normal(self) -> bool {
        Complex::is_normal(self)
    }
}

impl<T: Float> Scalar for T {
    /// Associated Real type
    type Real = T;
    /// Associated Complex type
    type Complex = Complex<T>;

    fn sqrt(&self) -> Self {
        Float::sqrt(*self)
    }
    fn exp(&self) -> Self {
        Float::exp(*self)
    }
    fn ln(&self) -> Self {
        Float::ln(*self)
    }
    fn abs_sqr(&self) -> Self::Real {
        Float::abs(*self).powi(2)
    }
    fn abs(&self) -> Self::Real {
        Float::abs(*self)
    }

    // powers
    fn powi(&self, exp: i32) -> Self {
        Float::powi(*self, exp)
    }
    fn powf(&self, exp: Self::Real) -> Self {
        Float::powf(*self, exp)
    }
    fn powc(&self, exp: Self::Complex) -> Self::Complex {
        let c = Complex::new(*self, T::zero());
        c.powc(exp)
    }

    fn conj(&self) -> Self {
        *self
    }

    // trigonometric functions
    fn sin(&self) -> Self {
        Float::sin(*self)
    }
    fn cos(&self) -> Self {
        Float::cos(*self)
    }
    fn tan(&self) -> Self {
        Float::tan(*self)
    }
    fn asin(&self) -> Self {
        Float::asin(*self)
    }
    fn acos(&self) -> Self {
        Float::acos(*self)
    }
    fn atan(&self) -> Self {
        Float::atan(*self)
    }

    // hyperbolic functions
    fn sinh(&self) -> Self {
        Float::sinh(*self)
    }
    fn cosh(&self) -> Self {
        Float::cosh(*self)
    }
    fn tanh(&self) -> Self {
        Float::tanh(*self)
    }
    fn asinh(&self) -> Self {
        Float::asinh(*self)
    }
    fn acosh(&self) -> Self {
        Float::acosh(*self)
    }
    fn atanh(&self) -> Self {
        Float::atanh(*self)
    }

    // check normal
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
    fn is_infinite(self) -> bool {
        Float::is_infinite(self)
    }
    fn is_finite(self) -> bool {
        Float::is_finite(self)
    }
    fn is_normal(self) -> bool {
        Float::is_normal(self)
    }
}
