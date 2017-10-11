
use traits::{Num, FromPrimitive, Float};
use std::ops::Neg;

use Complex;

pub trait Associated {
    type Real: Scalar;
    type Complex: Scalar;
}

impl<T: Scalar> Associated for T
where
    Complex<T::Repr>: Scalar,
{
    type Real = T::Repr;
    type Complex = Complex<T::Repr>;
}

pub trait Scalar: Num + Copy + Neg<Output = Self> + Associated {
    /// Associated Repr type
    type Repr: Scalar<Repr = Self::Repr>;

    /// Take the square root of a number.
    fn sqrt(&self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    fn exp(&self) -> Self;

    /// Returns the natural logarithm of the number.
    fn ln(&self) -> Self;

    /// Returns the square of the absolute value of the number
    fn abs_sqr(&self) -> Self::Repr;

    /// Returns the absolute value of the number
    fn abs(&self) -> Self::Real;

    /// Raise a number to an integer power.
    fn powi(&self, exp: i32) -> Self;

    /// Raise a number to a floating point power.
    fn powf(&self, exp: Self::Repr) -> Self;

    /// Raise a number to a complex power.
    fn powc(&self, exp: Self::Complex) -> Complex<Self::Repr>;

    /// Returns complex-conjugate number
    fn conj(&self) -> Self;

    /// Computes the sine of a number
    fn sin(&self) -> Self;

    /// Computes the cosine of a number
    fn cos(&self) -> Self;

    /// Computes the tangent of a number
    fn tan(&self) -> Self;

    /// Computes the arcsine of a number
    fn asin(&self) -> Self;

    /// Computes the arccosine of a number
    fn acos(&self) -> Self;

    /// Computes the arctangent of a number
    fn atan(&self) -> Self;

    /// Computes the hyperbolic-sine of a number
    fn sinh(&self) -> Self;

    /// Computes the hyperbolic-cosine of a number
    fn cosh(&self) -> Self;

    /// Computes the hyperbolic-tangent of a number
    fn tanh(&self) -> Self;

    /// Computes the hyperbolic-arcsine of a number
    fn asinh(&self) -> Self;

    /// Computes the hyperbolic-arccosine of a number
    fn acosh(&self) -> Self;

    /// Computes the hyperbolic-arctangent of a number
    fn atanh(&self) -> Self;

    /// Checks if the given (real or imaginary part of complex) number is NaN
    fn is_nan(self) -> bool;

    /// Checks if the given (real or imaginary part of complex) number is infinite
    fn is_infinite(self) -> bool;

    /// Checks if the given number is finite
    fn is_finite(self) -> bool;

    /// Checks if the given number is normal
    fn is_normal(self) -> bool;
}

impl<T: Clone + Float + FromPrimitive> Scalar for Complex<T> {
    type Repr = T;

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

impl<T: Float> Scalar for T
where
    Complex<T>: Scalar,
{
    type Repr = T;

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
        *self * *self
    }

    fn abs(&self) -> Self::Real {
        Float::abs(*self)
    }

    fn powi(&self, exp: i32) -> Self {
        Float::powi(*self, exp)
    }

    fn powf(&self, exp: Self::Real) -> Self {
        Float::powf(*self, exp)
    }

    fn powc(&self, exp: Self::Complex) -> Self::Complex {
        exp.expf(*self)
    }

    fn conj(&self) -> Self {
        *self
    }

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
