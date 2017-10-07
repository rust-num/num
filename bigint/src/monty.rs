use std::iter::repeat;
use integer::Integer;
use traits::{Zero, One};

use biguint::BigUint;

pub struct MontyReducer<'a> {
    p: &'a BigUint,
    n: Vec<u32>,
    n0inv: u64
}

// Calculate the modular inverse of `num`, using Extended GCD.
//
// Reference:
// Brent & Zimmermann, Modern Computer Arithmetic, v0.5.9, Algorithm 1.20
fn inv_mod_u32(num: u32) -> u64 {
    // num needs to be relatively prime to u32::max_value()
    assert!(num % 2 != 0);

    let mut a: i64 = num as i64;
    let mut b: i64 = (u32::max_value() as i64) + 1;
    let mu = b;

    // ExtendedGcd
    // Input: positive integers a and b
    // Output: integers (g, u, v) such that g = gcd(a, b) = ua + vb
    // As we don't need v for modular inverse, we don't calculate it.

    // 1: (u, w) <- (1, 0)
    let mut u = 1;
    let mut w = 0;
    // 3: while b != 0
    while b != 0 {
        // 4: (q, r) <- DivRem(a, b)
        let q = a / b;
        let r = a % b;
        // 5: (a, b) <- (b, r)
        a = b; b = r;
        // 6: (u, w) <- (w, u - qw)
        let m = u - w*q;
        u = w; w = m;
    }

    assert!(a == 1);
    // Ensure returned value is in-range
    if u < 0 {
        (u + mu) as u64
    } else {
        u as u64
    }
}

impl<'a> MontyReducer<'a> {
    pub fn new(p: &'a BigUint) -> Self {
        let n : Vec<u32> = p.data.clone();
        let n0inv = inv_mod_u32(n[0]);
        MontyReducer { p: p, n: n, n0inv: n0inv }
    }
}

// Montgomery Reduction
//
// Reference:
// Brent & Zimmermann, Modern Computer Arithmetic, v0.5.9, Algorithm 2.6
pub fn monty_redc(a: BigUint, mr: &MontyReducer) -> BigUint {
    let mut c = a.data;
    let n = &mr.n;
    let n_size = n.len();
    let old_size = c.len();

    // Allocate sufficient work space
    c.reserve(2*n_size+2-old_size);
    c.extend(repeat(0).take(2*n_size+2-old_size));

    // β is the size of a word, in this case 32 bits. So "a mod β" is
    // equivalent to masking a to 32 bits.
    let beta_mask = u32::max_value() as u64;
    // mu <- -N^(-1) mod β
    let mu = (beta_mask-mr.n0inv)+1;

    // 1: for i = 0 to (n-1)
    for i in 0..n_size {
        // Carry storage
        let mut carry = 0;

        // 2: q_i <- mu*c_i mod β
        let q_i = ((c[i] as u64) * mu) & beta_mask;

        // 3: C <- C + q_i * N * β^i
        // When iterating over each word, this becomes:
        for j in 0..n_size {
            // c_(i+j) <- c_(i+j) + q_i * n_j
            let x = (c[i+j] as u64) + q_i * (n[j] as u64) + carry;
            c[i+j] = (x & beta_mask) as u32;
            carry = x >> 32;
        }

        // Apply the remaining carry to the rest of the work space
        for j in n_size..2*n_size-i+2 {
            let x = (c[i+j] as u64) + carry;
            c[i+j] = (x & beta_mask) as u32;
            carry = x >> 32;
        }
    }

    // 4: R <- C * β^(-n)
    // This is an n-word bitshift, equivalent to skipping n words.
    let r : Vec<u32> = c.iter().skip(n_size).cloned().collect();
    let ret = BigUint::new(r);

    // 5: if R >= β^n then return R-N else return R.
    if &ret < mr.p {
        ret
    } else {
        &ret-mr.p
    }
}

// Montgomery Multiplication
fn monty_mult(a: BigUint, b: &BigUint, mr: &MontyReducer) -> BigUint {
    monty_redc(a * b, mr)
}

// Montgomery Squaring
fn monty_sqr(a: BigUint, mr: &MontyReducer) -> BigUint {
    // TODO: Replace with an optimised squaring function
    monty_redc(&a * &a, mr)
}

pub fn monty_modpow(a: &BigUint, exp: &BigUint, mr: &MontyReducer) -> BigUint{
    // Calculate the Montgomery parameter
    let mut r : BigUint = One::one();
    while &r < mr.p {
        r = r << 32;
    }

    // Map the base to the Montgomery domain
    let mut apri = a * &r % mr.p;

    // Binary exponentiation
    let mut ans = &r % mr.p;
    let mut e = exp.clone();
    let zero = Zero::zero();
    while e > zero {
        if e.is_odd() {
            ans = monty_mult(ans, &apri, mr);
        }
        apri = monty_sqr(apri, mr);
        e = e >> 1;
    }

    // Map the result back to the residues domain
    monty_redc(ans, mr)
}
