#![feature(test)]

extern crate test;
extern crate num;
extern crate rand;

use std::mem::replace;
use test::Bencher;
use num::{BigInt, BigUint, Zero, One, FromPrimitive};
use num::bigint::RandBigInt;
use rand::{SeedableRng, StdRng};

fn get_rng() -> StdRng {
    let seed: &[_] = &[1, 2, 3, 4];
    SeedableRng::from_seed(seed)
}

fn multiply_bench(b: &mut Bencher, xbits: usize, ybits: usize) {
    let mut rng = get_rng();
    let x = rng.gen_bigint(xbits);
    let y = rng.gen_bigint(ybits);

    b.iter(|| &x * &y);
}

fn divide_bench(b: &mut Bencher, xbits: usize, ybits: usize) {
    let mut rng = get_rng();
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
    b.iter(|| factorial(100));
}

#[bench]
fn fib_100(b: &mut Bencher) {
    b.iter(|| fib(100));
}

#[bench]
fn fac_to_string(b: &mut Bencher) {
    let fac = factorial(100);
    b.iter(|| fac.to_string());
}

#[bench]
fn fib_to_string(b: &mut Bencher) {
    let fib = fib(100);
    b.iter(|| fib.to_string());
}

fn to_str_radix_bench(b: &mut Bencher, radix: u32) {
    let mut rng = get_rng();
    let x = rng.gen_bigint(1009);
    b.iter(|| x.to_str_radix(radix));
}

#[bench]
fn to_str_radix_02(b: &mut Bencher) {
    to_str_radix_bench(b, 2);
}

#[bench]
fn to_str_radix_08(b: &mut Bencher) {
    to_str_radix_bench(b, 8);
}

#[bench]
fn to_str_radix_10(b: &mut Bencher) {
    to_str_radix_bench(b, 10);
}

#[bench]
fn to_str_radix_16(b: &mut Bencher) {
    to_str_radix_bench(b, 16);
}

#[bench]
fn to_str_radix_36(b: &mut Bencher) {
    to_str_radix_bench(b, 36);
}

fn from_str_radix_bench(b: &mut Bencher, radix: u32) {
    use num::Num;
    let mut rng = get_rng();
    let x = rng.gen_bigint(1009);
    let s = x.to_str_radix(radix);
    assert_eq!(x, BigInt::from_str_radix(&s, radix).unwrap());
    b.iter(|| BigInt::from_str_radix(&s, radix));
}

#[bench]
fn from_str_radix_02(b: &mut Bencher) {
    from_str_radix_bench(b, 2);
}

#[bench]
fn from_str_radix_08(b: &mut Bencher) {
    from_str_radix_bench(b, 8);
}

#[bench]
fn from_str_radix_10(b: &mut Bencher) {
    from_str_radix_bench(b, 10);
}

#[bench]
fn from_str_radix_16(b: &mut Bencher) {
    from_str_radix_bench(b, 16);
}

#[bench]
fn from_str_radix_36(b: &mut Bencher) {
    from_str_radix_bench(b, 36);
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

#[bench]
fn hash(b: &mut Bencher) {
    use std::collections::HashSet;
    let mut rng = get_rng();
    let v: Vec<BigInt> = (1000..2000).map(|bits| rng.gen_bigint(bits)).collect();
    b.iter(|| {
        let h: HashSet<&BigInt> = v.iter().collect();
        assert_eq!(h.len(), v.len());
    });
}
