#![feature(test)]

extern crate test;
extern crate num;
extern crate rand;

use std::mem::replace;
use test::Bencher;
use num::{BigUint, Zero, One, FromPrimitive};
use num::bigint::RandBigInt;
use rand::{SeedableRng, StdRng};

fn multiply_bench(b: &mut Bencher, xbits: usize, ybits: usize) {
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let x = rng.gen_bigint(xbits);
    let y = rng.gen_bigint(ybits);

    b.iter(|| &x * &y);
}

fn divide_bench(b: &mut Bencher, xbits: usize, ybits: usize) {
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let x = rng.gen_bigint(xbits);
    let y = rng.gen_bigint(ybits);

    b.iter(|| &x / &y);
}

fn factorial(n: usize) -> BigUint {
    let mut f: BigUint = One::one();
    for i in 1..(n+1) {
        let bu: BigUint = FromPrimitive::from_usize(i).unwrap();
        f = f * bu;
    }
    f
}

fn fib(n: usize) -> BigUint {
    let mut f0: BigUint = Zero::zero();
    let mut f1: BigUint = One::one();
    for _ in 0..n {
        let f2 = f0 + &f1;
        f0 = replace(&mut f1, f2);
    }
    f0
}

#[bench]
fn multiply_0(b: &mut Bencher) {
    multiply_bench(b, 1 << 8, 1 << 8);
}

#[bench]
fn multiply_1(b: &mut Bencher) {
    multiply_bench(b, 1 << 8, 1 << 16);
}

#[bench]
fn multiply_2(b: &mut Bencher) {
    multiply_bench(b, 1 << 16, 1 << 16);
}

#[bench]
fn divide_0(b: &mut Bencher) {
    divide_bench(b, 1 << 8, 1 << 6);
}

#[bench]
fn divide_1(b: &mut Bencher) {
    divide_bench(b, 1 << 12, 1 << 8);
}

#[bench]
fn divide_2(b: &mut Bencher) {
    divide_bench(b, 1 << 16, 1 << 12);
}

#[bench]
fn factorial_100(b: &mut Bencher) {
    b.iter(|| {
        factorial(100);
    });
}

#[bench]
fn fib_100(b: &mut Bencher) {
    b.iter(|| {
        fib(100);
    });
}

#[bench]
fn to_string(b: &mut Bencher) {
    let fac = factorial(100);
    let fib = fib(100);
    b.iter(|| {
        fac.to_string();
    });
    b.iter(|| {
        fib.to_string();
    });
}

#[bench]
fn shr(b: &mut Bencher) {
    let n = { let one : BigUint = One::one(); one << 1000 };
    b.iter(|| {
        let mut m = n.clone();
        for _ in 0..10 {
            m = m >> 1;
        }
    })
}
