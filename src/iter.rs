// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! External iterators for generic mathematics

use {Integer, Zero, One, CheckedAdd};
use std::ops::{Add, Sub};
use std::num::{ToPrimitive, Int};

/// An iterator over the range [start, stop)
#[derive(Clone)]
pub struct Range<A> {
    state: A,
    stop: A,
    one: A
}

/// Returns an iterator over the given range [start, stop) (that is, starting
/// at start (inclusive), and ending at stop (exclusive)).
///
/// # Example
///
/// ```rust
/// let array = [0, 1, 2, 3, 4];
///
/// for i in range(0, 5u) {
///     println!("{}", i);
///     assert_eq!(i,  array[i]);
/// }
/// ```
#[inline]
pub fn range<A: Add<A, A> + PartialOrd + Clone + One>(start: A, stop: A) -> Range<A> {
    Range{state: start, stop: stop, one: One::one()}
}

// FIXME: rust-lang/rust#10414: Unfortunate type bound
impl<A: Add<A, A> + PartialOrd + Clone + ToPrimitive> Iterator<A> for Range<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.state < self.stop {
            let result = self.state.clone();
            self.state = self.state.clone() + self.one.clone();
            Some(result)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        // This first checks if the elements are representable as i64. If they aren't, try u64 (to
        // handle cases like range(huge, huger)). We don't use uint/int because the difference of
        // the i64/u64 might lie within their range.
        let bound = match self.state.to_i64() {
            Some(a) => {
                let sz = self.stop.to_i64().map(|b| b.checked_sub(a));
                match sz {
                    Some(Some(bound)) => bound.to_uint(),
                    _ => None,
                }
            },
            None => match self.state.to_u64() {
                Some(a) => {
                    let sz = self.stop.to_u64().map(|b| b.checked_sub(a));
                    match sz {
                        Some(Some(bound)) => bound.to_uint(),
                        _ => None
                    }
                },
                None => None
            }
        };

        match bound {
            Some(b) => (b, Some(b)),
            // Standard fallback for unbounded/unrepresentable bounds
            None => (0, None)
        }
    }
}

/// `Integer` is required to ensure the range will be the same regardless of
/// the direction it is consumed.
impl<A: Integer + PartialOrd + Clone + ToPrimitive> DoubleEndedIterator<A> for Range<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.stop > self.state {
            self.stop = self.stop.clone() - self.one.clone();
            Some(self.stop.clone())
        } else {
            None
        }
    }
}

/// An iterator over the range [start, stop]
#[derive(Clone)]
pub struct RangeInclusive<A> {
    range: Range<A>,
    done: bool,
}

/// Return an iterator over the range [start, stop]
#[inline]
pub fn range_inclusive<A: Add<A, A> + PartialOrd + Clone + One>(start: A, stop: A)
    -> RangeInclusive<A> {
    RangeInclusive{range: range(start, stop), done: false}
}

impl<A: Add<A, A> + PartialOrd + Clone + ToPrimitive> Iterator<A> for RangeInclusive<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        match self.range.next() {
            Some(x) => Some(x),
            None => {
                if !self.done && self.range.state == self.range.stop {
                    self.done = true;
                    Some(self.range.stop.clone())
                } else {
                    None
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let (lo, hi) = self.range.size_hint();
        if self.done {
            (lo, hi)
        } else {
            let lo = lo.saturating_add(1);
            let hi = match hi {
                Some(x) => x.checked_add(1),
                None => None
            };
            (lo, hi)
        }
    }
}

impl<A: Sub<A, A> + Integer + PartialOrd + Clone + ToPrimitive> DoubleEndedIterator<A>
    for RangeInclusive<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.range.stop > self.range.state {
            let result = self.range.stop.clone();
            self.range.stop = self.range.stop.clone() - self.range.one.clone();
            Some(result)
        } else if !self.done && self.range.state == self.range.stop {
            self.done = true;
            Some(self.range.stop.clone())
        } else {
            None
        }
    }
}

/// An iterator over the range [start, stop) by `step`. It handles overflow by stopping.
#[derive(Clone)]
pub struct RangeStep<A> {
    state: A,
    stop: A,
    step: A,
    rev: bool,
}

/// Return an iterator over the range [start, stop) by `step`. It handles overflow by stopping.
#[inline]
pub fn range_step<A: CheckedAdd + PartialOrd +
                  Clone + Zero>(start: A, stop: A, step: A) -> RangeStep<A> {
    let rev = step < Zero::zero();
    RangeStep{state: start, stop: stop, step: step, rev: rev}
}

impl<A: CheckedAdd + PartialOrd + Clone> Iterator<A> for RangeStep<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if (self.rev && self.state > self.stop) || (!self.rev && self.state < self.stop) {
            let result = self.state.clone();
            match self.state.checked_add(&self.step) {
                Some(x) => self.state = x,
                None => self.state = self.stop.clone()
            }
            Some(result)
        } else {
            None
        }
    }
}

/// An iterator over the range [start, stop] by `step`. It handles overflow by stopping.
#[derive(Clone)]
pub struct RangeStepInclusive<A> {
    state: A,
    stop: A,
    step: A,
    rev: bool,
    done: bool,
}

/// Return an iterator over the range [start, stop] by `step`. It handles overflow by stopping.
#[inline]
pub fn range_step_inclusive<A: CheckedAdd + PartialOrd + Clone + Zero>(start: A, stop: A,
                                                                step: A) -> RangeStepInclusive<A> {
    let rev = step < Zero::zero();
    RangeStepInclusive{state: start, stop: stop, step: step, rev: rev, done: false}
}

impl<A: CheckedAdd + PartialOrd + Clone + PartialEq> Iterator<A> for RangeStepInclusive<A> {
    #[inline]
    fn next(&mut self) -> Option<A> {
        if !self.done && ((self.rev && self.state >= self.stop) ||
                          (!self.rev && self.state <= self.stop)) {
            let result = self.state.clone();
            match self.state.checked_add(&self.step) {
                Some(x) => self.state = x,
                None => self.done = true
            }
            Some(result)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::uint;
    use std::num::ToPrimitive;
    use std::ops::{Add, Mul};
    use std::cmp::Ordering;
    use One;

    #[test]
    fn test_range() {
        /// A mock type to check Range when ToPrimitive returns None
        struct Foo;

        impl ToPrimitive for Foo {
            fn to_i64(&self) -> Option<i64> { None }
            fn to_u64(&self) -> Option<u64> { None }
        }

        impl Add<Foo, Foo> for Foo {
            fn add(self, _: Foo) -> Foo {
                Foo
            }
        }

        impl PartialEq for Foo {
            fn eq(&self, _: &Foo) -> bool {
                true
            }
        }

        impl PartialOrd for Foo {
            fn partial_cmp(&self, _: &Foo) -> Option<Ordering> {
                None
            }
        }

        impl Clone for Foo {
            fn clone(&self) -> Foo {
                Foo
            }
        }

        impl Mul<Foo, Foo> for Foo {
            fn mul(self, _: Foo) -> Foo {
                Foo
            }
        }

        impl One for Foo {
            fn one() -> Foo {
                Foo
            }
        }

        assert!(super::range(0i, 5).collect::<Vec<int>>() == vec![0i, 1, 2, 3, 4]);
        assert!(super::range(-10i, -1).collect::<Vec<int>>() ==
                vec![-10, -9, -8, -7, -6, -5, -4, -3, -2]);
        assert!(super::range(0i, 5).rev().collect::<Vec<int>>() == vec![4, 3, 2, 1, 0]);
        assert_eq!(super::range(200i, -5).count(), 0);
        assert_eq!(super::range(200i, -5).rev().count(), 0);
        assert_eq!(super::range(200i, 200).count(), 0);
        assert_eq!(super::range(200i, 200).rev().count(), 0);

        assert_eq!(super::range(0i, 100).size_hint(), (100, Some(100)));
        // this test is only meaningful when sizeof uint < sizeof u64
        assert_eq!(super::range(uint::MAX - 1, uint::MAX).size_hint(), (1, Some(1)));
        assert_eq!(super::range(-10i, -1).size_hint(), (9, Some(9)));
        assert_eq!(super::range(Foo, Foo).size_hint(), (0, None));
    }

    #[test]
    fn test_range_inclusive() {
        assert!(super::range_inclusive(0i, 5).collect::<Vec<int>>() ==
                vec![0i, 1, 2, 3, 4, 5]);
        assert!(super::range_inclusive(0i, 5).rev().collect::<Vec<int>>() ==
                vec![5i, 4, 3, 2, 1, 0]);
        assert_eq!(super::range_inclusive(200i, -5).count(), 0);
        assert_eq!(super::range_inclusive(200i, -5).rev().count(), 0);
        assert!(super::range_inclusive(200i, 200).collect::<Vec<int>>() == vec![200]);
        assert!(super::range_inclusive(200i, 200).rev().collect::<Vec<int>>() == vec![200]);
    }

    #[test]
    fn test_range_step() {
        assert!(super::range_step(0i, 20, 5).collect::<Vec<int>>() ==
                vec![0, 5, 10, 15]);
        assert!(super::range_step(20i, 0, -5).collect::<Vec<int>>() ==
                vec![20, 15, 10, 5]);
        assert!(super::range_step(20i, 0, -6).collect::<Vec<int>>() ==
                vec![20, 14, 8, 2]);
        assert!(super::range_step(200u8, 255, 50).collect::<Vec<u8>>() ==
                vec![200u8, 250]);
        assert!(super::range_step(200i, -5, 1).collect::<Vec<int>>() == vec![]);
        assert!(super::range_step(200i, 200, 1).collect::<Vec<int>>() == vec![]);
    }

    #[test]
    fn test_range_step_inclusive() {
        assert!(super::range_step_inclusive(0i, 20, 5).collect::<Vec<int>>() ==
                vec![0, 5, 10, 15, 20]);
        assert!(super::range_step_inclusive(20i, 0, -5).collect::<Vec<int>>() ==
                vec![20, 15, 10, 5, 0]);
        assert!(super::range_step_inclusive(20i, 0, -6).collect::<Vec<int>>() ==
                vec![20, 14, 8, 2]);
        assert!(super::range_step_inclusive(200u8, 255, 50).collect::<Vec<u8>>() ==
                vec![200u8, 250]);
        assert!(super::range_step_inclusive(200i, -5, 1).collect::<Vec<int>>() ==
                vec![]);
        assert!(super::range_step_inclusive(200i, 200, 1).collect::<Vec<int>>() ==
                vec![200]);
    }
}
