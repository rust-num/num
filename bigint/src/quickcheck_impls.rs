use bigint::{BigInt, Sign};
use biguint::BigUint;
use quickcheck::{Arbitrary, empty_shrinker, Gen};
use std::iter::once;
use traits::{FromPrimitive, Signed, Zero};

impl Arbitrary for BigInt {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let sign = if g.gen() {
            Sign::Plus 
        } else {
            Sign::Minus
        };
        Self::new(sign, Arbitrary::arbitrary(g))
    }

    fn shrink(&self) -> Box<Iterator<Item = Self>> {
        /// Based on the SignedShrinker for primitive types in quickcheck
        /// itself.
        struct Iter(BigInt, BigInt);
        impl Iterator for Iter {
            type Item = BigInt;

            fn next(&mut self) -> Option<BigInt> {
                if (self.0.clone() - &self.1).abs() < self.0.abs() {
                    let result = Some(self.0.clone() - &self.1);
                    // TODO This would benefit from in-place `/=`.
                    self.1 = self.1.clone() / BigInt::from_usize(2).unwrap();
                    result
                } else {
                    None
                }
            }
        }

        if self.is_zero() {
            empty_shrinker()
        } else {
            let two = BigInt::from_usize(2).unwrap();
            let shrinker = Iter(self.clone(), self.clone() / two);
            let mut items = vec![Self::zero()];
            if shrinker.1.is_negative() {
                items.push(shrinker.0.abs());
            }
            Box::new(items.into_iter().chain(shrinker))
        }
    }
}

impl Arbitrary for BigUint {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Self::new(Arbitrary::arbitrary(g))
    }

    fn shrink(&self) -> Box<Iterator<Item = Self>> {
        /// Based on the UnsignedShrinker for primitive types in quickcheck
        /// itself.
        struct Iter(BigUint, BigUint);
        impl Iterator for Iter {
            type Item = BigUint;

            fn next(&mut self) -> Option<BigUint> {
                if (self.0.clone() - &self.1) < self.0 {
                    let result = Some(self.0.clone() - &self.1);
                    // TODO This would benefit from in-place `/=`.
                    self.1 = self.1.clone() / BigUint::from_usize(2).unwrap();
                    result
                } else {
                    None
                }
            }
        }

        if self.is_zero() {
            empty_shrinker()
        } else {
            let two = BigUint::from_usize(2).unwrap();
            let shrinker = Iter(self.clone(), self.clone() / two);
            Box::new(once(Self::zero()).chain(shrinker))
        }
    }
}
