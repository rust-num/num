use core::mem;
use core::ops::Neg;
use core::num::FpCategory;

// Used for default implementation of `epsilon`
use core::f32;

use {ToPrimitive, Num, NumCast};

// FIXME: these doctests aren't actually helpful, because they're using and
// testing the inherent methods directly, not going through `Float`.

/// Floating point operations.
///
/// Please note that some methods are disabled for `no_std`. If you implement it
/// only for `no_std`, the build will fail if anyone else in the dependency
/// graph enables `num-traits/std`.
pub trait Float
    : Num
    + Copy
    + NumCast
    + PartialOrd
    + Neg<Output = Self>
{
    /// Returns the `NaN` value.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let nan: f32 = Float::nan();
    ///
    /// assert!(nan.is_nan());
    /// ```
    fn nan() -> Self;

    /// Returns the infinite value.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f32;
    ///
    /// let infinity: f32 = Float::infinity();
    ///
    /// assert!(infinity.is_infinite());
    /// assert!(!infinity.is_finite());
    /// assert!(infinity > f32::MAX);
    /// ```
    fn infinity() -> Self;

    /// Returns the negative infinite value.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f32;
    ///
    /// let neg_infinity: f32 = Float::neg_infinity();
    ///
    /// assert!(neg_infinity.is_infinite());
    /// assert!(!neg_infinity.is_finite());
    /// assert!(neg_infinity < f32::MIN);
    /// ```
    fn neg_infinity() -> Self;

    /// Returns `-0.0`.
    ///
    /// ```
    /// use num_traits::{Zero, Float};
    ///
    /// let inf: f32 = Float::infinity();
    /// let zero: f32 = Zero::zero();
    /// let neg_zero: f32 = Float::neg_zero();
    ///
    /// assert_eq!(zero, neg_zero);
    /// assert_eq!(7.0f32/inf, zero);
    /// assert_eq!(zero * 10.0, zero);
    /// ```
    #[inline]
    fn neg_zero() -> Self {
        -Self::zero()
    }

    /// Returns the smallest finite value that this type can represent.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::min_value();
    ///
    /// assert_eq!(x, f64::MIN);
    /// ```
    fn min_value() -> Self;

    /// Returns the smallest positive, normalized value that this type can represent.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::min_positive_value();
    ///
    /// assert_eq!(x, f64::MIN_POSITIVE);
    /// ```
    fn min_positive_value() -> Self;

    /// Returns epsilon, a small positive value.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::epsilon();
    ///
    /// assert_eq!(x, f64::EPSILON);
    /// ```
    ///
    /// # Panics
    ///
    /// The default implementation will panic if `f32::EPSILON` cannot
    /// be cast to `Self`.
    fn epsilon() -> Self {
        Self::from(f32::EPSILON).expect("Unable to cast from f32::EPSILON")
    }

    /// Returns the largest finite value that this type can represent.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x: f64 = Float::max_value();
    /// assert_eq!(x, f64::MAX);
    /// ```
    fn max_value() -> Self;

    /// Returns `true` if this value is `NaN` and false otherwise.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let nan = f64::NAN;
    /// let f = 7.0;
    ///
    /// assert!(nan.is_nan());
    /// assert!(!f.is_nan());
    /// ```
    #[inline]
    fn is_nan(self) -> bool {
        self != self
    }

    /// Returns `true` if this value is positive infinity or negative infinity and
    /// false otherwise.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf: f32 = Float::infinity();
    /// let neg_inf: f32 = Float::neg_infinity();
    /// let nan: f32 = f32::NAN;
    ///
    /// assert!(!f.is_infinite());
    /// assert!(!nan.is_infinite());
    ///
    /// assert!(inf.is_infinite());
    /// assert!(neg_inf.is_infinite());
    /// ```
    #[inline]
    fn is_infinite(self) -> bool {
        self == Self::infinity() || self == Self::neg_infinity()
    }

    /// Returns `true` if this number is neither infinite nor `NaN`.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f32;
    ///
    /// let f = 7.0f32;
    /// let inf: f32 = Float::infinity();
    /// let neg_inf: f32 = Float::neg_infinity();
    /// let nan: f32 = f32::NAN;
    ///
    /// assert!(f.is_finite());
    ///
    /// assert!(!nan.is_finite());
    /// assert!(!inf.is_finite());
    /// assert!(!neg_inf.is_finite());
    /// ```
    #[inline]
    fn is_finite(self) -> bool {
        !(self.is_nan() || self.is_infinite())
    }

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal][subnormal], or `NaN`.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f32;
    ///
    /// let min = f32::MIN_POSITIVE; // 1.17549435e-38f32
    /// let max = f32::MAX;
    /// let lower_than_min = 1.0e-40_f32;
    /// let zero = 0.0f32;
    ///
    /// assert!(min.is_normal());
    /// assert!(max.is_normal());
    ///
    /// assert!(!zero.is_normal());
    /// assert!(!f32::NAN.is_normal());
    /// assert!(!f32::INFINITY.is_normal());
    /// // Values between `0` and `min` are Subnormal.
    /// assert!(!lower_than_min.is_normal());
    /// ```
    /// [subnormal]: http://en.wikipedia.org/wiki/Denormal_number
    #[inline]
    fn is_normal(self) -> bool {
        self.classify() == FpCategory::Normal
    }

    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::num::FpCategory;
    /// use std::f32;
    ///
    /// let num = 12.4f32;
    /// let inf = f32::INFINITY;
    ///
    /// assert_eq!(num.classify(), FpCategory::Normal);
    /// assert_eq!(inf.classify(), FpCategory::Infinite);
    /// ```
    fn classify(self) -> FpCategory;

    /// Returns the largest integer less than or equal to a number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let f = 3.99;
    /// let g = 3.0;
    ///
    /// assert_eq!(f.floor(), 3.0);
    /// assert_eq!(g.floor(), 3.0);
    /// ```
    #[cfg(feature = "std")]
    fn floor(self) -> Self;

    /// Returns the smallest integer greater than or equal to a number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let f = 3.01;
    /// let g = 4.0;
    ///
    /// assert_eq!(f.ceil(), 4.0);
    /// assert_eq!(g.ceil(), 4.0);
    /// ```
    #[cfg(feature = "std")]
    fn ceil(self) -> Self;

    /// Returns the nearest integer to a number. Round half-way cases away from
    /// `0.0`.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let f = 3.3;
    /// let g = -3.3;
    ///
    /// assert_eq!(f.round(), 3.0);
    /// assert_eq!(g.round(), -3.0);
    /// ```
    #[cfg(feature = "std")]
    fn round(self) -> Self;

    /// Return the integer part of a number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let f = 3.3;
    /// let g = -3.7;
    ///
    /// assert_eq!(f.trunc(), 3.0);
    /// assert_eq!(g.trunc(), -3.0);
    /// ```
    #[cfg(feature = "std")]
    fn trunc(self) -> Self;

    /// Returns the fractional part of a number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 3.5;
    /// let y = -3.5;
    /// let abs_difference_x = (x.fract() - 0.5).abs();
    /// let abs_difference_y = (y.fract() - (-0.5)).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn fract(self) -> Self;

    /// Computes the absolute value of `self`. Returns `Float::nan()` if the
    /// number is `Float::nan()`.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x = 3.5;
    /// let y = -3.5;
    ///
    /// let abs_difference_x = (x.abs() - x).abs();
    /// let abs_difference_y = (y.abs() - (-y)).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    ///
    /// assert!(f64::NAN.abs().is_nan());
    /// ```
    #[inline]
    fn abs(self) -> Self {
        if self.is_sign_positive() {
            return self;
        }
        if self.is_sign_negative() {
            return -self;
        }
        Self::nan()
    }

    /// Returns a number that represents the sign of `self`.
    ///
    /// - `1.0` if the number is positive, `+0.0` or `Float::infinity()`
    /// - `-1.0` if the number is negative, `-0.0` or `Float::neg_infinity()`
    /// - `Float::nan()` if the number is `Float::nan()`
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let f = 3.5;
    ///
    /// assert_eq!(f.signum(), 1.0);
    /// assert_eq!(f64::NEG_INFINITY.signum(), -1.0);
    ///
    /// assert!(f64::NAN.signum().is_nan());
    /// ```
    #[inline]
    fn signum(self) -> Self {
        if self.is_nan() {
            Self::nan()
        } else if self.is_sign_negative() {
            -Self::one()
        } else {
            Self::one()
        }
    }

    /// Returns `true` if `self` is positive, including `+0.0`,
    /// `Float::infinity()`, and with newer versions of Rust `f64::NAN`.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let neg_nan: f64 = -f64::NAN;
    ///
    /// let f = 7.0;
    /// let g = -7.0;
    ///
    /// assert!(f.is_sign_positive());
    /// assert!(!g.is_sign_positive());
    /// assert!(!neg_nan.is_sign_positive());
    /// ```
    #[inline]
    fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` if `self` is negative, including `-0.0`,
    /// `Float::neg_infinity()`, and with newer versions of Rust `-f64::NAN`.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let nan: f64 = f64::NAN;
    ///
    /// let f = 7.0;
    /// let g = -7.0;
    ///
    /// assert!(!f.is_sign_negative());
    /// assert!(g.is_sign_negative());
    /// assert!(!nan.is_sign_negative());
    /// ```
    #[inline]
    fn is_sign_negative(self) -> bool {
        let (_, _, sign) = self.integer_decode();
        sign < 0
    }

    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error. This produces a more accurate result with better performance than
    /// a separate multiplication operation followed by an add.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let m = 10.0;
    /// let x = 4.0;
    /// let b = 60.0;
    ///
    /// // 100.0
    /// let abs_difference = (m.mul_add(x, b) - (m*x + b)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Take the reciprocal (inverse) of a number, `1/x`.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.recip() - (1.0/x)).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[inline]
    fn recip(self) -> Self {
        Self::one() / self
    }

    /// Raise a number to an integer power.
    ///
    /// Using this function is generally faster than using `powf`
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.powi(2) - x*x).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[inline]
    fn powi(mut self, mut exp: i32) -> Self {
        if exp < 0 {
            self = self.recip();
            exp = -exp;
        }
        // It should always be possible to convert a positive `i32` to a `usize`.
        super::pow(self, exp.to_usize().unwrap())
    }

    /// Raise a number to a floating point power.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 2.0;
    /// let abs_difference = (x.powf(2.0) - x*x).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self;

    /// Take the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let positive = 4.0;
    /// let negative = -4.0;
    ///
    /// let abs_difference = (positive.sqrt() - 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// assert!(negative.sqrt().is_nan());
    /// ```
    #[cfg(feature = "std")]
    fn sqrt(self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let one = 1.0;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn exp(self) -> Self;

    /// Returns `2^(self)`.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let f = 2.0;
    ///
    /// // 2^2 - 4 == 0
    /// let abs_difference = (f.exp2() - 4.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn exp2(self) -> Self;

    /// Returns the natural logarithm of the number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let one = 1.0;
    /// // e^1
    /// let e = one.exp();
    ///
    /// // ln(e) - 1 == 0
    /// let abs_difference = (e.ln() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn ln(self) -> Self;

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let ten = 10.0;
    /// let two = 2.0;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference_10 = (ten.log(10.0) - 1.0).abs();
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference_2 = (two.log(2.0) - 1.0).abs();
    ///
    /// assert!(abs_difference_10 < 1e-10);
    /// assert!(abs_difference_2 < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self;

    /// Returns the base 2 logarithm of the number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let two = 2.0;
    ///
    /// // log2(2) - 1 == 0
    /// let abs_difference = (two.log2() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn log2(self) -> Self;

    /// Returns the base 10 logarithm of the number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let ten = 10.0;
    ///
    /// // log10(10) - 1 == 0
    /// let abs_difference = (ten.log10() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn log10(self) -> Self;

    /// Converts radians to degrees.
    ///
    /// ```
    /// use std::f64::consts;
    ///
    /// let angle = consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    #[inline]
    fn to_degrees(self) -> Self {
        let halfpi = Self::zero().acos();
        let ninety = Self::from(90u8).unwrap();
        self * ninety / halfpi
    }

    /// Converts radians to degrees.
    ///
    /// ```
    /// use std::f64::consts;
    ///
    /// let angle = consts::PI;
    ///
    /// let abs_difference = (angle.to_degrees() - 180.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(not(feature = "std"))]
    fn to_degrees(self) -> Self;

    /// Converts degrees to radians.
    ///
    /// ```
    /// use std::f64::consts;
    ///
    /// let angle = 180.0_f64;
    ///
    /// let abs_difference = (angle.to_radians() - consts::PI).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    #[inline]
    fn to_radians(self) -> Self {
        let halfpi = Self::zero().acos();
        let ninety = Self::from(90u8).unwrap();
        self * halfpi / ninety
    }

    /// Converts degrees to radians.
    ///
    /// ```
    /// use std::f64::consts;
    ///
    /// let angle = 180.0_f64;
    ///
    /// let abs_difference = (angle.to_radians() - consts::PI).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(not(feature = "std"))]
    fn to_radians(self) -> Self;

    /// Returns the maximum of the two numbers.
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 1.0;
    /// let y = 2.0;
    ///
    /// assert_eq!(x.max(y), y);
    /// ```
    #[inline]
    fn max(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        if self > other { self } else { other }
    }

    /// Returns the minimum of the two numbers.
    ///
    /// If one of the arguments is NaN, then the other argument is returned.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 1.0;
    /// let y = 2.0;
    ///
    /// assert_eq!(x.min(y), x);
    /// ```
    #[inline]
    fn min(self, other: Self) -> Self {
        if self.is_nan() {
            return other;
        }
        if other.is_nan() {
            return self;
        }
        if self < other { self } else { other }
    }

    /// The positive difference of two numbers.
    ///
    /// * If `self <= other`: `0:0`
    /// * Else: `self - other`
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 3.0;
    /// let y = -3.0;
    ///
    /// let abs_difference_x = (x.abs_sub(1.0) - 2.0).abs();
    /// let abs_difference_y = (y.abs_sub(1.0) - 0.0).abs();
    ///
    /// assert!(abs_difference_x < 1e-10);
    /// assert!(abs_difference_y < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn abs_sub(self, other: Self) -> Self;

    /// Take the cubic root of a number.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 8.0;
    ///
    /// // x^(1/3) - 2 == 0
    /// let abs_difference = (x.cbrt() - 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn cbrt(self) -> Self;

    /// Calculate the length of the hypotenuse of a right-angle triangle given
    /// legs of length `x` and `y`.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 2.0;
    /// let y = 3.0;
    ///
    /// // sqrt(x^2 + y^2)
    /// let abs_difference = (x.hypot(y) - (x.powi(2) + y.powi(2)).sqrt()).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self;

    /// Computes the sine of a number (in radians).
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/2.0;
    ///
    /// let abs_difference = (x.sin() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn sin(self) -> Self;

    /// Computes the cosine of a number (in radians).
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x = 2.0*f64::consts::PI;
    ///
    /// let abs_difference = (x.cos() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn cos(self) -> Self;

    /// Computes the tangent of a number (in radians).
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/4.0;
    /// let abs_difference = (x.tan() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-14);
    /// ```
    #[cfg(feature = "std")]
    fn tan(self) -> Self;

    /// Computes the arcsine of a number. Return value is in radians in
    /// the range [-pi/2, pi/2] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let f = f64::consts::PI / 2.0;
    ///
    /// // asin(sin(pi/2))
    /// let abs_difference = (f.sin().asin() - f64::consts::PI / 2.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn asin(self) -> Self;

    /// Computes the arccosine of a number. Return value is in radians in
    /// the range [0, pi] or NaN if the number is outside the range
    /// [-1, 1].
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let f = f64::consts::PI / 4.0;
    ///
    /// // acos(cos(pi/4))
    /// let abs_difference = (f.cos().acos() - f64::consts::PI / 4.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn acos(self) -> Self;

    /// Computes the arctangent of a number. Return value is in radians in the
    /// range [-pi/2, pi/2];
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let f = 1.0;
    ///
    /// // atan(tan(1))
    /// let abs_difference = (f.tan().atan() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn atan(self) -> Self;

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`).
    ///
    /// * `x = 0`, `y = 0`: `0`
    /// * `x >= 0`: `arctan(y/x)` -> `[-pi/2, pi/2]`
    /// * `y >= 0`: `arctan(y/x) + pi` -> `(pi/2, pi]`
    /// * `y < 0`: `arctan(y/x) - pi` -> `(-pi, -pi/2)`
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let pi = f64::consts::PI;
    /// // All angles from horizontal right (+x)
    /// // 45 deg counter-clockwise
    /// let x1 = 3.0;
    /// let y1 = -3.0;
    ///
    /// // 135 deg clockwise
    /// let x2 = -3.0;
    /// let y2 = 3.0;
    ///
    /// let abs_difference_1 = (y1.atan2(x1) - (-pi/4.0)).abs();
    /// let abs_difference_2 = (y2.atan2(x2) - 3.0*pi/4.0).abs();
    ///
    /// assert!(abs_difference_1 < 1e-10);
    /// assert!(abs_difference_2 < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self;

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns
    /// `(sin(x), cos(x))`.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::PI/4.0;
    /// let f = x.sin_cos();
    ///
    /// let abs_difference_0 = (f.0 - x.sin()).abs();
    /// let abs_difference_1 = (f.1 - x.cos()).abs();
    ///
    /// assert!(abs_difference_0 < 1e-10);
    /// assert!(abs_difference_0 < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self);

    /// Returns `e^(self) - 1` in a way that is accurate even if the
    /// number is close to zero.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 7.0;
    ///
    /// // e^(ln(7)) - 1
    /// let abs_difference = (x.ln().exp_m1() - 6.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self;

    /// Returns `ln(1+n)` (natural logarithm) more accurately than if
    /// the operations were performed separately.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let x = f64::consts::E - 1.0;
    ///
    /// // ln(1 + (e - 1)) == ln(e) == 1
    /// let abs_difference = (x.ln_1p() - 1.0).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self;

    /// Hyperbolic sine function.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    ///
    /// let f = x.sinh();
    /// // Solving sinh() at 1 gives `(e^2-1)/(2e)`
    /// let g = (e*e - 1.0)/(2.0*e);
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    #[cfg(feature = "std")]
    fn sinh(self) -> Self;

    /// Hyperbolic cosine function.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    /// let f = x.cosh();
    /// // Solving cosh() at 1 gives this result
    /// let g = (e*e + 1.0)/(2.0*e);
    /// let abs_difference = (f - g).abs();
    ///
    /// // Same result
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[cfg(feature = "std")]
    fn cosh(self) -> Self;

    /// Hyperbolic tangent function.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let x = 1.0;
    ///
    /// let f = x.tanh();
    /// // Solving tanh() at 1 gives `(1 - e^(-2))/(1 + e^(-2))`
    /// let g = (1.0 - e.powi(-2))/(1.0 + e.powi(-2));
    /// let abs_difference = (f - g).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[cfg(feature = "std")]
    fn tanh(self) -> Self;

    /// Inverse hyperbolic sine function.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 1.0;
    /// let f = x.sinh().asinh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[cfg(feature = "std")]
    fn asinh(self) -> Self;

    /// Inverse hyperbolic cosine function.
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let x = 1.0;
    /// let f = x.cosh().acosh();
    ///
    /// let abs_difference = (f - x).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[cfg(feature = "std")]
    fn acosh(self) -> Self;

    /// Inverse hyperbolic tangent function.
    ///
    /// ```
    /// use num_traits::Float;
    /// use std::f64;
    ///
    /// let e = f64::consts::E;
    /// let f = e.tanh().atanh();
    ///
    /// let abs_difference = (f - e).abs();
    ///
    /// assert!(abs_difference < 1.0e-10);
    /// ```
    #[cfg(feature = "std")]
    fn atanh(self) -> Self;


    /// Returns the mantissa, base 2 exponent, and sign as integers, respectively.
    /// The original number can be recovered by `sign * mantissa * 2 ^ exponent`.
    /// The floating point encoding is documented in the [Reference][floating-point].
    ///
    /// ```
    /// use num_traits::Float;
    ///
    /// let num = 2.0f32;
    ///
    /// // (8388608, -22, 1)
    /// let (mantissa, exponent, sign) = Float::integer_decode(num);
    /// let sign_f = sign as f32;
    /// let mantissa_f = mantissa as f32;
    /// let exponent_f = num.powf(exponent as f32);
    ///
    /// // 1 * 8388608 * 2^(-22) == 2
    /// let abs_difference = (sign_f * mantissa_f * exponent_f - num).abs();
    ///
    /// assert!(abs_difference < 1e-10);
    /// ```
    /// [floating-point]: ../../../../../reference.html#machine-types
    fn integer_decode(self) -> (u64, i16, i8);
}

macro_rules! float_impl {
    ($T:ident $decode:ident $classify:ident) => (
        impl Float for $T {
            #[inline]
            fn nan() -> Self {
                ::core::$T::NAN
            }

            #[inline]
            fn infinity() -> Self {
                ::core::$T::INFINITY
            }

            #[inline]
            fn neg_infinity() -> Self {
                ::core::$T::NEG_INFINITY
            }

            #[inline]
            fn neg_zero() -> Self {
                -0.0
            }

            #[inline]
            fn min_value() -> Self {
                ::core::$T::MIN
            }

            #[inline]
            fn min_positive_value() -> Self {
                ::core::$T::MIN_POSITIVE
            }

            #[inline]
            fn epsilon() -> Self {
                ::core::$T::EPSILON
            }

            #[inline]
            fn max_value() -> Self {
                ::core::$T::MAX
            }

            #[cfg(feature = "std")]
            #[inline]
            fn is_nan(self) -> bool {
                <$T>::is_nan(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn is_infinite(self) -> bool {
                <$T>::is_infinite(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn is_finite(self) -> bool {
                <$T>::is_finite(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn is_normal(self) -> bool {
                <$T>::is_normal(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn classify(self) -> FpCategory {
                <$T>::classify(self)
            }

            #[cfg(not(feature = "std"))]
            #[inline]
            fn classify(self) -> FpCategory {
                $classify(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn floor(self) -> Self {
                <$T>::floor(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn ceil(self) -> Self {
                <$T>::ceil(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn round(self) -> Self {
                <$T>::round(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn trunc(self) -> Self {
                <$T>::trunc(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn fract(self) -> Self {
                <$T>::fract(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn abs(self) -> Self {
                <$T>::abs(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn signum(self) -> Self {
                <$T>::signum(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn is_sign_positive(self) -> bool {
                <$T>::is_sign_positive(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn is_sign_negative(self) -> bool {
                <$T>::is_sign_negative(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$T>::mul_add(self, a, b)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn recip(self) -> Self {
                <$T>::recip(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn powi(self, n: i32) -> Self {
                <$T>::powi(self, n)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn powf(self, n: Self) -> Self {
                <$T>::powf(self, n)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn sqrt(self) -> Self {
                <$T>::sqrt(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn exp(self) -> Self {
                <$T>::exp(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn exp2(self) -> Self {
                <$T>::exp2(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn ln(self) -> Self {
                <$T>::ln(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn log(self, base: Self) -> Self {
                <$T>::log(self, base)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn log2(self) -> Self {
                <$T>::log2(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn log10(self) -> Self {
                <$T>::log10(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn to_degrees(self) -> Self {
                <$T>::to_degrees(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn to_radians(self) -> Self {
                <$T>::to_radians(self)
            }

            #[cfg(not(feature = "std"))]
            #[inline]
            fn to_degrees(self) -> Self {
               self * (180. / ::core::$T::consts::PI)
            }

            #[cfg(not(feature = "std"))]
            #[inline]
            fn to_radians(self) -> Self {
               self * (::core::$T::consts::PI / 180.)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn max(self, other: Self) -> Self {
                <$T>::max(self, other)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn min(self, other: Self) -> Self {
                <$T>::min(self, other)
            }

            #[cfg(feature = "std")]
            #[inline]
            #[allow(deprecated)]
            fn abs_sub(self, other: Self) -> Self {
                <$T>::abs_sub(self, other)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn cbrt(self) -> Self {
                <$T>::cbrt(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn hypot(self, other: Self) -> Self {
                <$T>::hypot(self, other)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn sin(self) -> Self {
                <$T>::sin(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn cos(self) -> Self {
                <$T>::cos(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn tan(self) -> Self {
                <$T>::tan(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn asin(self) -> Self {
                <$T>::asin(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn acos(self) -> Self {
                <$T>::acos(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn atan(self) -> Self {
                <$T>::atan(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn atan2(self, other: Self) -> Self {
                <$T>::atan2(self, other)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                <$T>::sin_cos(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn exp_m1(self) -> Self {
                <$T>::exp_m1(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn ln_1p(self) -> Self {
                <$T>::ln_1p(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn sinh(self) -> Self {
                <$T>::sinh(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn cosh(self) -> Self {
                <$T>::cosh(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn tanh(self) -> Self {
                <$T>::tanh(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn asinh(self) -> Self {
                <$T>::asinh(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn acosh(self) -> Self {
                <$T>::acosh(self)
            }

            #[cfg(feature = "std")]
            #[inline]
            fn atanh(self) -> Self {
                <$T>::atanh(self)
            }

            #[inline]
            fn integer_decode(self) -> (u64, i16, i8) {
                $decode(self)
            }
        }
    )
}

fn integer_decode_f32(f: f32) -> (u64, i16, i8) {
    let bits: u32 = unsafe { mem::transmute(f) };
    let sign: i8 = if bits >> 31 == 0 {
        1
    } else {
        -1
    };
    let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0x7fffff) << 1
    } else {
        (bits & 0x7fffff) | 0x800000
    };
    // Exponent bias + mantissa shift
    exponent -= 127 + 23;
    (mantissa as u64, exponent, sign)
}

#[cfg(not(feature = "std"))]
fn classify_f32(f: f32) -> FpCategory {
    const EXP_MASK: u32 = 0x7f800000;
    const MAN_MASK: u32 = 0x007fffff;

    let bits: u32 = unsafe { mem::transmute(f) };
    match (bits & MAN_MASK, bits & EXP_MASK) {
        (0, 0) => FpCategory::Zero,
        (_, 0) => FpCategory::Subnormal,
        (0, EXP_MASK) => FpCategory::Infinite,
        (_, EXP_MASK) => FpCategory::Nan,
        _ => FpCategory::Normal,
    }
}

fn integer_decode_f64(f: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(f) };
    let sign: i8 = if bits >> 63 == 0 {
        1
    } else {
        -1
    };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };
    // Exponent bias + mantissa shift
    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

#[cfg(not(feature = "std"))]
fn classify_f64(f: f64) -> FpCategory {
    const EXP_MASK: u64 = 0x7ff0000000000000;
    const MAN_MASK: u64 = 0x000fffffffffffff;

    let bits: u64 = unsafe { mem::transmute(f) };
    match (bits & MAN_MASK, bits & EXP_MASK) {
        (0, 0) => FpCategory::Zero,
        (_, 0) => FpCategory::Subnormal,
        (0, EXP_MASK) => FpCategory::Infinite,
        (_, EXP_MASK) => FpCategory::Nan,
        _ => FpCategory::Normal,
    }
}

float_impl!(f32 integer_decode_f32 classify_f32);
float_impl!(f64 integer_decode_f64 classify_f64);

macro_rules! float_const_impl {
    ($(#[$doc:meta] $constant:ident,)+) => (
        #[allow(non_snake_case)]
        pub trait FloatConst {
            $(#[$doc] fn $constant() -> Self;)+
        }
        float_const_impl! { @float f32, $($constant,)+ }
        float_const_impl! { @float f64, $($constant,)+ }
    );
    (@float $T:ident, $($constant:ident,)+) => (
        impl FloatConst for $T {
            $(
                #[inline]
                fn $constant() -> Self {
                    ::core::$T::consts::$constant
                }
            )+
        }
    );
}

float_const_impl! {
    #[doc = "Return Euler’s number."]
    E,
    #[doc = "Return `1.0 / π`."]
    FRAC_1_PI,
    #[doc = "Return `1.0 / sqrt(2.0)`."]
    FRAC_1_SQRT_2,
    #[doc = "Return `2.0 / π`."]
    FRAC_2_PI,
    #[doc = "Return `2.0 / sqrt(π)`."]
    FRAC_2_SQRT_PI,
    #[doc = "Return `π / 2.0`."]
    FRAC_PI_2,
    #[doc = "Return `π / 3.0`."]
    FRAC_PI_3,
    #[doc = "Return `π / 4.0`."]
    FRAC_PI_4,
    #[doc = "Return `π / 6.0`."]
    FRAC_PI_6,
    #[doc = "Return `π / 8.0`."]
    FRAC_PI_8,
    #[doc = "Return `ln(10.0)`."]
    LN_10,
    #[doc = "Return `ln(2.0)`."]
    LN_2,
    #[doc = "Return `log10(e)`."]
    LOG10_E,
    #[doc = "Return `log2(e)`."]
    LOG2_E,
    #[doc = "Return Archimedes’ constant."]
    PI,
    #[doc = "Return `sqrt(2.0)`."]
    SQRT_2,
}

#[cfg(test)]
mod tests {
    use Float;
    use core::f64::consts;

    const DEG_RAD_PAIRS: [(f64, f64); 7] = [
        (0.0, 0.),
        (22.5, consts::FRAC_PI_8),
        (30.0, consts::FRAC_PI_6),
        (45.0, consts::FRAC_PI_4),
        (60.0, consts::FRAC_PI_3),
        (90.0, consts::FRAC_PI_2),
        (180.0, consts::PI),
    ];

    #[test]
    fn convert_deg_rad_core() {
        for &(deg, rad) in &DEG_RAD_PAIRS {
            assert!((Float::to_degrees(rad) - deg).abs() < 1e-6);
            assert!((Float::to_radians(deg) - rad).abs() < 1e-6);

            let (deg, rad) = (deg as f32, rad as f32);
            assert!((Float::to_degrees(rad) - deg).abs() < 1e-6);
            assert!((Float::to_radians(deg) - rad).abs() < 1e-6);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn convert_deg_rad_std() {
        use Float;

        for &(deg, rad) in &DEG_RAD_PAIRS {
            assert!((Float::to_degrees(rad) - deg).abs() < 1e-6);
            assert!((Float::to_radians(deg) - rad).abs() < 1e-6);

            let (deg, rad) = (deg as f32, rad as f32);
            assert!((Float::to_degrees(rad) - deg).abs() < 1e-6);
            assert!((Float::to_radians(deg) - rad).abs() < 1e-6);
        }
    }
}
