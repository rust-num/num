use integer::Integer;
use {BigDigit, BigUint, ToBigUint, big_digit};
use {BigInt, RandBigInt, ToBigInt};
use Sign::Plus;

use std::cmp::Ordering::{Less, Equal, Greater};
use std::{f32, f64};
use std::i64;
use std::iter::repeat;
use std::str::FromStr;
use std::{u8, u16, u32, u64, usize};

use rand::thread_rng;
use traits::{Num, Zero, One, CheckedAdd, CheckedSub, CheckedMul, CheckedDiv, ToPrimitive,
             FromPrimitive, Float};


/// Assert that an op works for all val/ref combinations
macro_rules! assert_op {
    ($left:ident $op:tt $right:ident == $expected:expr) => {
        assert_eq!((&$left) $op (&$right), $expected);
        assert_eq!((&$left) $op $right.clone(), $expected);
        assert_eq!($left.clone() $op (&$right), $expected);
        assert_eq!($left.clone() $op $right.clone(), $expected);
    };
}

#[test]
fn test_from_slice() {
    fn check(slice: &[BigDigit], data: &[BigDigit]) {
        assert!(BigUint::from_slice(slice).data == data);
    }
    check(&[1], &[1]);
    check(&[0, 0, 0], &[]);
    check(&[1, 2, 0, 0], &[1, 2]);
    check(&[0, 0, 1, 2], &[0, 0, 1, 2]);
    check(&[0, 0, 1, 2, 0, 0], &[0, 0, 1, 2]);
    check(&[-1i32 as BigDigit], &[-1i32 as BigDigit]);
}

#[test]
fn test_from_bytes_be() {
    fn check(s: &str, result: &str) {
        assert_eq!(BigUint::from_bytes_be(s.as_bytes()),
                   BigUint::parse_bytes(result.as_bytes(), 10).unwrap());
    }
    check("A", "65");
    check("AA", "16705");
    check("AB", "16706");
    check("Hello world!", "22405534230753963835153736737");
    assert_eq!(BigUint::from_bytes_be(&[]), Zero::zero());
}

#[test]
fn test_to_bytes_be() {
    fn check(s: &str, result: &str) {
        let b = BigUint::parse_bytes(result.as_bytes(), 10).unwrap();
        assert_eq!(b.to_bytes_be(), s.as_bytes());
    }
    check("A", "65");
    check("AA", "16705");
    check("AB", "16706");
    check("Hello world!", "22405534230753963835153736737");
    let b: BigUint = Zero::zero();
    assert_eq!(b.to_bytes_be(), [0]);

    // Test with leading/trailing zero bytes and a full BigDigit of value 0
    let b = BigUint::from_str_radix("00010000000000000200", 16).unwrap();
    assert_eq!(b.to_bytes_be(), [1, 0, 0, 0, 0, 0, 0, 2, 0]);
}

#[test]
fn test_from_bytes_le() {
    fn check(s: &str, result: &str) {
        assert_eq!(BigUint::from_bytes_le(s.as_bytes()),
                   BigUint::parse_bytes(result.as_bytes(), 10).unwrap());
    }
    check("A", "65");
    check("AA", "16705");
    check("BA", "16706");
    check("!dlrow olleH", "22405534230753963835153736737");
    assert_eq!(BigUint::from_bytes_le(&[]), Zero::zero());
}

#[test]
fn test_to_bytes_le() {
    fn check(s: &str, result: &str) {
        let b = BigUint::parse_bytes(result.as_bytes(), 10).unwrap();
        assert_eq!(b.to_bytes_le(), s.as_bytes());
    }
    check("A", "65");
    check("AA", "16705");
    check("BA", "16706");
    check("!dlrow olleH", "22405534230753963835153736737");
    let b: BigUint = Zero::zero();
    assert_eq!(b.to_bytes_le(), [0]);

    // Test with leading/trailing zero bytes and a full BigDigit of value 0
    let b = BigUint::from_str_radix("00010000000000000200", 16).unwrap();
    assert_eq!(b.to_bytes_le(), [0, 2, 0, 0, 0, 0, 0, 0, 1]);
}

#[test]
fn test_cmp() {
    let data: [&[_]; 7] = [&[], &[1], &[2], &[!0], &[0, 1], &[2, 1], &[1, 1, 1]];
    let data: Vec<BigUint> = data.iter().map(|v| BigUint::from_slice(*v)).collect();
    for (i, ni) in data.iter().enumerate() {
        for (j0, nj) in data[i..].iter().enumerate() {
            let j = j0 + i;
            if i == j {
                assert_eq!(ni.cmp(nj), Equal);
                assert_eq!(nj.cmp(ni), Equal);
                assert_eq!(ni, nj);
                assert!(!(ni != nj));
                assert!(ni <= nj);
                assert!(ni >= nj);
                assert!(!(ni < nj));
                assert!(!(ni > nj));
            } else {
                assert_eq!(ni.cmp(nj), Less);
                assert_eq!(nj.cmp(ni), Greater);

                assert!(!(ni == nj));
                assert!(ni != nj);

                assert!(ni <= nj);
                assert!(!(ni >= nj));
                assert!(ni < nj);
                assert!(!(ni > nj));

                assert!(!(nj <= ni));
                assert!(nj >= ni);
                assert!(!(nj < ni));
                assert!(nj > ni);
            }
        }
    }
}

#[test]
fn test_hash() {
    use hash;

    let a = BigUint::new(vec![]);
    let b = BigUint::new(vec![0]);
    let c = BigUint::new(vec![1]);
    let d = BigUint::new(vec![1, 0, 0, 0, 0, 0]);
    let e = BigUint::new(vec![0, 0, 0, 0, 0, 1]);
    assert!(hash(&a) == hash(&b));
    assert!(hash(&b) != hash(&c));
    assert!(hash(&c) == hash(&d));
    assert!(hash(&d) != hash(&e));
}

const BIT_TESTS: &'static [(&'static [BigDigit],
           &'static [BigDigit],
           &'static [BigDigit],
           &'static [BigDigit],
           &'static [BigDigit])] = &[// LEFT              RIGHT        AND          OR                XOR
                                     (&[], &[], &[], &[], &[]),
                                     (&[268, 482, 17],
                                      &[964, 54],
                                      &[260, 34],
                                      &[972, 502, 17],
                                      &[712, 468, 17])];

#[test]
fn test_bitand() {
    for elm in BIT_TESTS {
        let (a_vec, b_vec, c_vec, _, _) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert_op!(a & b == c);
        assert_op!(b & a == c);
    }
}

#[test]
fn test_bitor() {
    for elm in BIT_TESTS {
        let (a_vec, b_vec, _, c_vec, _) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert_op!(a | b == c);
        assert_op!(b | a == c);
    }
}

#[test]
fn test_bitxor() {
    for elm in BIT_TESTS {
        let (a_vec, b_vec, _, _, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert_op!(a ^ b == c);
        assert_op!(b ^ a == c);
        assert_op!(a ^ c == b);
        assert_op!(c ^ a == b);
        assert_op!(b ^ c == a);
        assert_op!(c ^ b == a);
    }
}

#[test]
fn test_shl() {
    fn check(s: &str, shift: usize, ans: &str) {
        let opt_biguint = BigUint::from_str_radix(s, 16).ok();
        let bu = (opt_biguint.unwrap() << shift).to_str_radix(16);
        assert_eq!(bu, ans);
    }

    check("0", 3, "0");
    check("1", 3, "8");

    check("1\
           0000\
           0000\
           0000\
           0001\
           0000\
           0000\
           0000\
           0001",
          3,
          "8\
           0000\
           0000\
           0000\
           0008\
           0000\
           0000\
           0000\
           0008");
    check("1\
           0000\
           0001\
           0000\
           0001",
          2,
          "4\
           0000\
           0004\
           0000\
           0004");
    check("1\
           0001\
           0001",
          1,
          "2\
           0002\
           0002");

    check("\
          4000\
          0000\
          0000\
          0000",
          3,
          "2\
          0000\
          0000\
          0000\
          0000");
    check("4000\
          0000",
          2,
          "1\
          0000\
          0000");
    check("4000",
          2,
          "1\
          0000");

    check("4000\
          0000\
          0000\
          0000",
          67,
          "2\
          0000\
          0000\
          0000\
          0000\
          0000\
          0000\
          0000\
          0000");
    check("4000\
          0000",
          35,
          "2\
          0000\
          0000\
          0000\
          0000");
    check("4000",
          19,
          "2\
          0000\
          0000");

    check("fedc\
          ba98\
          7654\
          3210\
          fedc\
          ba98\
          7654\
          3210",
          4,
          "f\
          edcb\
          a987\
          6543\
          210f\
          edcb\
          a987\
          6543\
          2100");
    check("88887777666655554444333322221111",
          16,
          "888877776666555544443333222211110000");
}

#[test]
fn test_shr() {
    fn check(s: &str, shift: usize, ans: &str) {
        let opt_biguint = BigUint::from_str_radix(s, 16).ok();
        let bu = (opt_biguint.unwrap() >> shift).to_str_radix(16);
        assert_eq!(bu, ans);
    }

    check("0", 3, "0");
    check("f", 3, "1");

    check("1\
          0000\
          0000\
          0000\
          0001\
          0000\
          0000\
          0000\
          0001",
          3,
          "2000\
          0000\
          0000\
          0000\
          2000\
          0000\
          0000\
          0000");
    check("1\
          0000\
          0001\
          0000\
          0001",
          2,
          "4000\
          0000\
          4000\
          0000");
    check("1\
          0001\
          0001",
          1,
          "8000\
          8000");

    check("2\
          0000\
          0000\
          0000\
          0001\
          0000\
          0000\
          0000\
          0001",
          67,
          "4000\
          0000\
          0000\
          0000");
    check("2\
          0000\
          0001\
          0000\
          0001",
          35,
          "4000\
          0000");
    check("2\
          0001\
          0001",
          19,
          "4000");

    check("1\
          0000\
          0000\
          0000\
          0000",
          1,
          "8000\
          0000\
          0000\
          0000");
    check("1\
          0000\
          0000",
          1,
          "8000\
          0000");
    check("1\
          0000",
          1,
          "8000");
    check("f\
          edcb\
          a987\
          6543\
          210f\
          edcb\
          a987\
          6543\
          2100",
          4,
          "fedc\
          ba98\
          7654\
          3210\
          fedc\
          ba98\
          7654\
          3210");

    check("888877776666555544443333222211110000",
          16,
          "88887777666655554444333322221111");
}

const N1: BigDigit = -1i32 as BigDigit;
const N2: BigDigit = -2i32 as BigDigit;

// `DoubleBigDigit` size dependent
#[test]
fn test_convert_i64() {
    fn check(b1: BigUint, i: i64) {
        let b2: BigUint = FromPrimitive::from_i64(i).unwrap();
        assert_eq!(b1, b2);
        assert_eq!(b1.to_i64().unwrap(), i);
    }

    check(Zero::zero(), 0);
    check(One::one(), 1);
    check(i64::MAX.to_biguint().unwrap(), i64::MAX);

    check(BigUint::new(vec![]), 0);
    check(BigUint::new(vec![1]), (1 << (0 * big_digit::BITS)));
    check(BigUint::new(vec![N1]), (1 << (1 * big_digit::BITS)) - 1);
    check(BigUint::new(vec![0, 1]), (1 << (1 * big_digit::BITS)));
    check(BigUint::new(vec![N1, N1 >> 1]), i64::MAX);

    assert_eq!(i64::MIN.to_biguint(), None);
    assert_eq!(BigUint::new(vec![N1, N1]).to_i64(), None);
    assert_eq!(BigUint::new(vec![0, 0, 1]).to_i64(), None);
    assert_eq!(BigUint::new(vec![N1, N1, N1]).to_i64(), None);
}

// `DoubleBigDigit` size dependent
#[test]
fn test_convert_u64() {
    fn check(b1: BigUint, u: u64) {
        let b2: BigUint = FromPrimitive::from_u64(u).unwrap();
        assert_eq!(b1, b2);
        assert_eq!(b1.to_u64().unwrap(), u);
    }

    check(Zero::zero(), 0);
    check(One::one(), 1);
    check(u64::MIN.to_biguint().unwrap(), u64::MIN);
    check(u64::MAX.to_biguint().unwrap(), u64::MAX);

    check(BigUint::new(vec![]), 0);
    check(BigUint::new(vec![1]), (1 << (0 * big_digit::BITS)));
    check(BigUint::new(vec![N1]), (1 << (1 * big_digit::BITS)) - 1);
    check(BigUint::new(vec![0, 1]), (1 << (1 * big_digit::BITS)));
    check(BigUint::new(vec![N1, N1]), u64::MAX);

    assert_eq!(BigUint::new(vec![0, 0, 1]).to_u64(), None);
    assert_eq!(BigUint::new(vec![N1, N1, N1]).to_u64(), None);
}

#[test]
fn test_convert_f32() {
    fn check(b1: &BigUint, f: f32) {
        let b2 = BigUint::from_f32(f).unwrap();
        assert_eq!(b1, &b2);
        assert_eq!(b1.to_f32().unwrap(), f);
    }

    check(&BigUint::zero(), 0.0);
    check(&BigUint::one(), 1.0);
    check(&BigUint::from(u16::MAX), 2.0.powi(16) - 1.0);
    check(&BigUint::from(1u64 << 32), 2.0.powi(32));
    check(&BigUint::from_slice(&[0, 0, 1]), 2.0.powi(64));
    check(&((BigUint::one() << 100) + (BigUint::one() << 123)),
          2.0.powi(100) + 2.0.powi(123));
    check(&(BigUint::one() << 127), 2.0.powi(127));
    check(&(BigUint::from((1u64 << 24) - 1) << (128 - 24)), f32::MAX);

    // keeping all 24 digits with the bits at different offsets to the BigDigits
    let x: u32 = 0b00000000101111011111011011011101;
    let mut f = x as f32;
    let mut b = BigUint::from(x);
    for _ in 0..64 {
        check(&b, f);
        f *= 2.0;
        b = b << 1;
    }

    // this number when rounded to f64 then f32 isn't the same as when rounded straight to f32
    let n: u64 = 0b0000000000111111111111111111111111011111111111111111111111111111;
    assert!((n as f64) as f32 != n as f32);
    assert_eq!(BigUint::from(n).to_f32(), Some(n as f32));

    // test rounding up with the bits at different offsets to the BigDigits
    let mut f = ((1u64 << 25) - 1) as f32;
    let mut b = BigUint::from(1u64 << 25);
    for _ in 0..64 {
        assert_eq!(b.to_f32(), Some(f));
        f *= 2.0;
        b = b << 1;
    }

    // rounding
    assert_eq!(BigUint::from_f32(-1.0), None);
    assert_eq!(BigUint::from_f32(-0.99999), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f32(-0.5), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f32(-0.0), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f32(f32::MIN_POSITIVE / 2.0),
               Some(BigUint::zero()));
    assert_eq!(BigUint::from_f32(f32::MIN_POSITIVE), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f32(0.5), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f32(0.99999), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f32(f32::consts::E), Some(BigUint::from(2u32)));
    assert_eq!(BigUint::from_f32(f32::consts::PI),
               Some(BigUint::from(3u32)));

    // special float values
    assert_eq!(BigUint::from_f32(f32::NAN), None);
    assert_eq!(BigUint::from_f32(f32::INFINITY), None);
    assert_eq!(BigUint::from_f32(f32::NEG_INFINITY), None);
    assert_eq!(BigUint::from_f32(f32::MIN), None);

    // largest BigUint that will round to a finite f32 value
    let big_num = (BigUint::one() << 128) - BigUint::one() - (BigUint::one() << (128 - 25));
    assert_eq!(big_num.to_f32(), Some(f32::MAX));
    assert_eq!((big_num + BigUint::one()).to_f32(), None);

    assert_eq!(((BigUint::one() << 128) - BigUint::one()).to_f32(), None);
    assert_eq!((BigUint::one() << 128).to_f32(), None);
}

#[test]
fn test_convert_f64() {
    fn check(b1: &BigUint, f: f64) {
        let b2 = BigUint::from_f64(f).unwrap();
        assert_eq!(b1, &b2);
        assert_eq!(b1.to_f64().unwrap(), f);
    }

    check(&BigUint::zero(), 0.0);
    check(&BigUint::one(), 1.0);
    check(&BigUint::from(u32::MAX), 2.0.powi(32) - 1.0);
    check(&BigUint::from(1u64 << 32), 2.0.powi(32));
    check(&BigUint::from_slice(&[0, 0, 1]), 2.0.powi(64));
    check(&((BigUint::one() << 100) + (BigUint::one() << 152)),
          2.0.powi(100) + 2.0.powi(152));
    check(&(BigUint::one() << 1023), 2.0.powi(1023));
    check(&(BigUint::from((1u64 << 53) - 1) << (1024 - 53)), f64::MAX);

    // keeping all 53 digits with the bits at different offsets to the BigDigits
    let x: u64 = 0b0000000000011110111110110111111101110111101111011111011011011101;
    let mut f = x as f64;
    let mut b = BigUint::from(x);
    for _ in 0..128 {
        check(&b, f);
        f *= 2.0;
        b = b << 1;
    }

    // test rounding up with the bits at different offsets to the BigDigits
    let mut f = ((1u64 << 54) - 1) as f64;
    let mut b = BigUint::from(1u64 << 54);
    for _ in 0..128 {
        assert_eq!(b.to_f64(), Some(f));
        f *= 2.0;
        b = b << 1;
    }

    // rounding
    assert_eq!(BigUint::from_f64(-1.0), None);
    assert_eq!(BigUint::from_f64(-0.99999), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f64(-0.5), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f64(-0.0), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f64(f64::MIN_POSITIVE / 2.0),
               Some(BigUint::zero()));
    assert_eq!(BigUint::from_f64(f64::MIN_POSITIVE), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f64(0.5), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f64(0.99999), Some(BigUint::zero()));
    assert_eq!(BigUint::from_f64(f64::consts::E), Some(BigUint::from(2u32)));
    assert_eq!(BigUint::from_f64(f64::consts::PI),
               Some(BigUint::from(3u32)));

    // special float values
    assert_eq!(BigUint::from_f64(f64::NAN), None);
    assert_eq!(BigUint::from_f64(f64::INFINITY), None);
    assert_eq!(BigUint::from_f64(f64::NEG_INFINITY), None);
    assert_eq!(BigUint::from_f64(f64::MIN), None);

    // largest BigUint that will round to a finite f64 value
    let big_num = (BigUint::one() << 1024) - BigUint::one() - (BigUint::one() << (1024 - 54));
    assert_eq!(big_num.to_f64(), Some(f64::MAX));
    assert_eq!((big_num + BigUint::one()).to_f64(), None);

    assert_eq!(((BigInt::one() << 1024) - BigInt::one()).to_f64(), None);
    assert_eq!((BigUint::one() << 1024).to_f64(), None);
}

#[test]
fn test_convert_to_bigint() {
    fn check(n: BigUint, ans: BigInt) {
        assert_eq!(n.to_bigint().unwrap(), ans);
        assert_eq!(n.to_bigint().unwrap().to_biguint().unwrap(), n);
    }
    check(Zero::zero(), Zero::zero());
    check(BigUint::new(vec![1, 2, 3]),
          BigInt::from_biguint(Plus, BigUint::new(vec![1, 2, 3])));
}

#[test]
fn test_convert_from_uint() {
    macro_rules! check {
        ($ty:ident, $max:expr) => {
            assert_eq!(BigUint::from($ty::zero()), BigUint::zero());
            assert_eq!(BigUint::from($ty::one()), BigUint::one());
            assert_eq!(BigUint::from($ty::MAX - $ty::one()), $max - BigUint::one());
            assert_eq!(BigUint::from($ty::MAX), $max);
        }
    }

    check!(u8, BigUint::from_slice(&[u8::MAX as BigDigit]));
    check!(u16, BigUint::from_slice(&[u16::MAX as BigDigit]));
    check!(u32, BigUint::from_slice(&[u32::MAX]));
    check!(u64, BigUint::from_slice(&[u32::MAX, u32::MAX]));
    check!(usize, BigUint::from(usize::MAX as u64));
}

const SUM_TRIPLES: &'static [(&'static [BigDigit],
           &'static [BigDigit],
           &'static [BigDigit])] = &[(&[], &[], &[]),
                                     (&[], &[1], &[1]),
                                     (&[1], &[1], &[2]),
                                     (&[1], &[1, 1], &[2, 1]),
                                     (&[1], &[N1], &[0, 1]),
                                     (&[1], &[N1, N1], &[0, 0, 1]),
                                     (&[N1, N1], &[N1, N1], &[N2, N1, 1]),
                                     (&[1, 1, 1], &[N1, N1], &[0, 1, 2]),
                                     (&[2, 2, 1], &[N1, N2], &[1, 1, 2])];

#[test]
fn test_add() {
    for elm in SUM_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert_op!(a + b == c);
        assert_op!(b + a == c);
    }
}

#[test]
fn test_sub() {
    for elm in SUM_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert_op!(c - a == b);
        assert_op!(c - b == a);
    }
}

#[test]
#[should_panic]
fn test_sub_fail_on_underflow() {
    let (a, b): (BigUint, BigUint) = (Zero::zero(), One::one());
    a - b;
}

const M: u32 = ::std::u32::MAX;
const MUL_TRIPLES: &'static [(&'static [BigDigit],
           &'static [BigDigit],
           &'static [BigDigit])] = &[(&[], &[], &[]),
                                     (&[], &[1], &[]),
                                     (&[2], &[], &[]),
                                     (&[1], &[1], &[1]),
                                     (&[2], &[3], &[6]),
                                     (&[1], &[1, 1, 1], &[1, 1, 1]),
                                     (&[1, 2, 3], &[3], &[3, 6, 9]),
                                     (&[1, 1, 1], &[N1], &[N1, N1, N1]),
                                     (&[1, 2, 3], &[N1], &[N1, N2, N2, 2]),
                                     (&[1, 2, 3, 4], &[N1], &[N1, N2, N2, N2, 3]),
                                     (&[N1], &[N1], &[1, N2]),
                                     (&[N1, N1], &[N1], &[1, N1, N2]),
                                     (&[N1, N1, N1], &[N1], &[1, N1, N1, N2]),
                                     (&[N1, N1, N1, N1], &[N1], &[1, N1, N1, N1, N2]),
                                     (&[M / 2 + 1], &[2], &[0, 1]),
                                     (&[0, M / 2 + 1], &[2], &[0, 0, 1]),
                                     (&[1, 2], &[1, 2, 3], &[1, 4, 7, 6]),
                                     (&[N1, N1], &[N1, N1, N1], &[1, 0, N1, N2, N1]),
                                     (&[N1, N1, N1],
                                      &[N1, N1, N1, N1],
                                      &[1, 0, 0, N1, N2, N1, N1]),
                                     (&[0, 0, 1], &[1, 2, 3], &[0, 0, 1, 2, 3]),
                                     (&[0, 0, 1], &[0, 0, 0, 1], &[0, 0, 0, 0, 0, 1])];

const DIV_REM_QUADRUPLES: &'static [(&'static [BigDigit],
           &'static [BigDigit],
           &'static [BigDigit],
           &'static [BigDigit])] = &[(&[1], &[2], &[], &[1]),
                                     (&[1, 1], &[2], &[M / 2 + 1], &[1]),
                                     (&[1, 1, 1], &[2], &[M / 2 + 1, M / 2 + 1], &[1]),
                                     (&[0, 1], &[N1], &[1], &[1]),
                                     (&[N1, N1], &[N2], &[2, 1], &[3])];

#[test]
fn test_mul() {
    for elm in MUL_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert_op!(a * b == c);
        assert_op!(b * a == c);
    }

    for elm in DIV_REM_QUADRUPLES.iter() {
        let (a_vec, b_vec, c_vec, d_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);
        let d = BigUint::from_slice(d_vec);

        assert!(a == &b * &c + &d);
        assert!(a == &c * &b + &d);
    }
}

#[test]
fn test_div_rem() {
    for elm in MUL_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        if !a.is_zero() {
            assert_op!(c / a == b);
            assert_op!(c % a == Zero::zero());
            assert_eq!(c.div_rem(&a), (b.clone(), Zero::zero()));
        }
        if !b.is_zero() {
            assert_op!(c / b == a);
            assert_op!(c % b == Zero::zero());
            assert_eq!(c.div_rem(&b), (a.clone(), Zero::zero()));
        }
    }

    for elm in DIV_REM_QUADRUPLES.iter() {
        let (a_vec, b_vec, c_vec, d_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);
        let d = BigUint::from_slice(d_vec);

        if !b.is_zero() {
            assert_op!(a / b == c);
            assert_op!(a % b == d);
            assert!(a.div_rem(&b) == (c, d));
        }
    }
}

#[test]
fn test_checked_add() {
    for elm in SUM_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert!(a.checked_add(&b).unwrap() == c);
        assert!(b.checked_add(&a).unwrap() == c);
    }
}

#[test]
fn test_checked_sub() {
    for elm in SUM_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert!(c.checked_sub(&a).unwrap() == b);
        assert!(c.checked_sub(&b).unwrap() == a);

        if a > c {
            assert!(a.checked_sub(&c).is_none());
        }
        if b > c {
            assert!(b.checked_sub(&c).is_none());
        }
    }
}

#[test]
fn test_checked_mul() {
    for elm in MUL_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        assert!(a.checked_mul(&b).unwrap() == c);
        assert!(b.checked_mul(&a).unwrap() == c);
    }

    for elm in DIV_REM_QUADRUPLES.iter() {
        let (a_vec, b_vec, c_vec, d_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);
        let d = BigUint::from_slice(d_vec);

        assert!(a == b.checked_mul(&c).unwrap() + &d);
        assert!(a == c.checked_mul(&b).unwrap() + &d);
    }
}

#[test]
fn test_mul_overflow() {
    /* Test for issue #187 - overflow due to mac3 incorrectly sizing temporary */
    let s = "531137992816767098689588206552468627329593117727031923199444138200403559860852242739162502232636710047537552105951370000796528760829212940754539968588340162273730474622005920097370111";
    let a: BigUint = s.parse().unwrap();
    let b = a.clone();
    let _ = a.checked_mul(&b);
}

#[test]
fn test_checked_div() {
    for elm in MUL_TRIPLES.iter() {
        let (a_vec, b_vec, c_vec) = *elm;
        let a = BigUint::from_slice(a_vec);
        let b = BigUint::from_slice(b_vec);
        let c = BigUint::from_slice(c_vec);

        if !a.is_zero() {
            assert!(c.checked_div(&a).unwrap() == b);
        }
        if !b.is_zero() {
            assert!(c.checked_div(&b).unwrap() == a);
        }

        assert!(c.checked_div(&Zero::zero()).is_none());
    }
}

#[test]
fn test_gcd() {
    fn check(a: usize, b: usize, c: usize) {
        let big_a: BigUint = FromPrimitive::from_usize(a).unwrap();
        let big_b: BigUint = FromPrimitive::from_usize(b).unwrap();
        let big_c: BigUint = FromPrimitive::from_usize(c).unwrap();

        assert_eq!(big_a.gcd(&big_b), big_c);
    }

    check(10, 2, 2);
    check(10, 3, 1);
    check(0, 3, 3);
    check(3, 3, 3);
    check(56, 42, 14);
}

#[test]
fn test_lcm() {
    fn check(a: usize, b: usize, c: usize) {
        let big_a: BigUint = FromPrimitive::from_usize(a).unwrap();
        let big_b: BigUint = FromPrimitive::from_usize(b).unwrap();
        let big_c: BigUint = FromPrimitive::from_usize(c).unwrap();

        assert_eq!(big_a.lcm(&big_b), big_c);
    }

    check(1, 0, 0);
    check(0, 1, 0);
    check(1, 1, 1);
    check(8, 9, 72);
    check(11, 5, 55);
    check(99, 17, 1683);
}

#[test]
fn test_is_even() {
    let one: BigUint = FromStr::from_str("1").unwrap();
    let two: BigUint = FromStr::from_str("2").unwrap();
    let thousand: BigUint = FromStr::from_str("1000").unwrap();
    let big: BigUint = FromStr::from_str("1000000000000000000000").unwrap();
    let bigger: BigUint = FromStr::from_str("1000000000000000000001").unwrap();
    assert!(one.is_odd());
    assert!(two.is_even());
    assert!(thousand.is_even());
    assert!(big.is_even());
    assert!(bigger.is_odd());
    assert!((&one << 64).is_even());
    assert!(((&one << 64) + one).is_odd());
}

fn to_str_pairs() -> Vec<(BigUint, Vec<(u32, String)>)> {
    let bits = big_digit::BITS;
    vec![(Zero::zero(),
          vec![(2, "0".to_string()), (3, "0".to_string())]),
         (BigUint::from_slice(&[0xff]),
          vec![(2, "11111111".to_string()),
               (3, "100110".to_string()),
               (4, "3333".to_string()),
               (5, "2010".to_string()),
               (6, "1103".to_string()),
               (7, "513".to_string()),
               (8, "377".to_string()),
               (9, "313".to_string()),
               (10, "255".to_string()),
               (11, "212".to_string()),
               (12, "193".to_string()),
               (13, "168".to_string()),
               (14, "143".to_string()),
               (15, "120".to_string()),
               (16, "ff".to_string())]),
         (BigUint::from_slice(&[0xfff]),
          vec![(2, "111111111111".to_string()),
               (4, "333333".to_string()),
               (16, "fff".to_string())]),
         (BigUint::from_slice(&[1, 2]),
          vec![(2,
                format!("10{}1", repeat("0").take(bits - 1).collect::<String>())),
               (4,
                format!("2{}1", repeat("0").take(bits / 2 - 1).collect::<String>())),
               (10,
                match bits {
                   64 => "36893488147419103233".to_string(),
                   32 => "8589934593".to_string(),
                   16 => "131073".to_string(),
                   _ => panic!(),
               }),
               (16,
                format!("2{}1", repeat("0").take(bits / 4 - 1).collect::<String>()))]),
         (BigUint::from_slice(&[1, 2, 3]),
          vec![(2,
                format!("11{}10{}1",
                        repeat("0").take(bits - 2).collect::<String>(),
                        repeat("0").take(bits - 1).collect::<String>())),
               (4,
                format!("3{}2{}1",
                        repeat("0").take(bits / 2 - 1).collect::<String>(),
                        repeat("0").take(bits / 2 - 1).collect::<String>())),
               (8,
                match bits {
                   64 => "14000000000000000000004000000000000000000001".to_string(),
                   32 => "6000000000100000000001".to_string(),
                   16 => "140000400001".to_string(),
                   _ => panic!(),
               }),
               (10,
                match bits {
                   64 => "1020847100762815390427017310442723737601".to_string(),
                   32 => "55340232229718589441".to_string(),
                   16 => "12885032961".to_string(),
                   _ => panic!(),
               }),
               (16,
                format!("3{}2{}1",
                        repeat("0").take(bits / 4 - 1).collect::<String>(),
                        repeat("0").take(bits / 4 - 1).collect::<String>()))])]
}

#[test]
fn test_to_str_radix() {
    let r = to_str_pairs();
    for num_pair in r.iter() {
        let &(ref n, ref rs) = num_pair;
        for str_pair in rs.iter() {
            let &(ref radix, ref str) = str_pair;
            assert_eq!(n.to_str_radix(*radix), *str);
        }
    }
}

#[test]
fn test_from_str_radix() {
    let r = to_str_pairs();
    for num_pair in r.iter() {
        let &(ref n, ref rs) = num_pair;
        for str_pair in rs.iter() {
            let &(ref radix, ref str) = str_pair;
            assert_eq!(n, &BigUint::from_str_radix(str, *radix).unwrap());
        }
    }

    let zed = BigUint::from_str_radix("Z", 10).ok();
    assert_eq!(zed, None);
    let blank = BigUint::from_str_radix("_", 2).ok();
    assert_eq!(blank, None);
    let plus_one = BigUint::from_str_radix("+1", 10).ok();
    assert_eq!(plus_one, Some(BigUint::from_slice(&[1])));
    let plus_plus_one = BigUint::from_str_radix("++1", 10).ok();
    assert_eq!(plus_plus_one, None);
    let minus_one = BigUint::from_str_radix("-1", 10).ok();
    assert_eq!(minus_one, None);
    let zero_plus_two = BigUint::from_str_radix("0+2", 10).ok();
    assert_eq!(zero_plus_two, None);
}

#[test]
fn test_all_str_radix() {
    use std::ascii::AsciiExt;

    let n = BigUint::new((0..10).collect());
    for radix in 2..37 {
        let s = n.to_str_radix(radix);
        let x = BigUint::from_str_radix(&s, radix);
        assert_eq!(x.unwrap(), n);

        let s = s.to_ascii_uppercase();
        let x = BigUint::from_str_radix(&s, radix);
        assert_eq!(x.unwrap(), n);
    }
}

#[test]
fn test_lower_hex() {
    let a = BigUint::parse_bytes(b"A", 16).unwrap();
    let hello = BigUint::parse_bytes("22405534230753963835153736737".as_bytes(), 10).unwrap();

    assert_eq!(format!("{:x}", a), "a");
    assert_eq!(format!("{:x}", hello), "48656c6c6f20776f726c6421");
    assert_eq!(format!("{:♥>+#8x}", a), "♥♥♥♥+0xa");
}

#[test]
fn test_upper_hex() {
    let a = BigUint::parse_bytes(b"A", 16).unwrap();
    let hello = BigUint::parse_bytes("22405534230753963835153736737".as_bytes(), 10).unwrap();

    assert_eq!(format!("{:X}", a), "A");
    assert_eq!(format!("{:X}", hello), "48656C6C6F20776F726C6421");
    assert_eq!(format!("{:♥>+#8X}", a), "♥♥♥♥+0xA");
}

#[test]
fn test_binary() {
    let a = BigUint::parse_bytes(b"A", 16).unwrap();
    let hello = BigUint::parse_bytes("224055342307539".as_bytes(), 10).unwrap();

    assert_eq!(format!("{:b}", a), "1010");
    assert_eq!(format!("{:b}", hello),
               "110010111100011011110011000101101001100011010011");
    assert_eq!(format!("{:♥>+#8b}", a), "♥+0b1010");
}

#[test]
fn test_octal() {
    let a = BigUint::parse_bytes(b"A", 16).unwrap();
    let hello = BigUint::parse_bytes("22405534230753963835153736737".as_bytes(), 10).unwrap();

    assert_eq!(format!("{:o}", a), "12");
    assert_eq!(format!("{:o}", hello), "22062554330674403566756233062041");
    assert_eq!(format!("{:♥>+#8o}", a), "♥♥♥+0o12");
}

#[test]
fn test_display() {
    let a = BigUint::parse_bytes(b"A", 16).unwrap();
    let hello = BigUint::parse_bytes("22405534230753963835153736737".as_bytes(), 10).unwrap();

    assert_eq!(format!("{}", a), "10");
    assert_eq!(format!("{}", hello), "22405534230753963835153736737");
    assert_eq!(format!("{:♥>+#8}", a), "♥♥♥♥♥+10");
}

#[test]
fn test_factor() {
    fn factor(n: usize) -> BigUint {
        let mut f: BigUint = One::one();
        for i in 2..n + 1 {
            // FIXME(#5992): assignment operator overloads
            // f *= FromPrimitive::from_usize(i);
            let bu: BigUint = FromPrimitive::from_usize(i).unwrap();
            f = f * bu;
        }
        return f;
    }

    fn check(n: usize, s: &str) {
        let n = factor(n);
        let ans = match BigUint::from_str_radix(s, 10) {
            Ok(x) => x,
            Err(_) => panic!(),
        };
        assert_eq!(n, ans);
    }

    check(3, "6");
    check(10, "3628800");
    check(20, "2432902008176640000");
    check(30, "265252859812191058636308480000000");
}

#[test]
fn test_bits() {
    assert_eq!(BigUint::new(vec![0, 0, 0, 0]).bits(), 0);
    let n: BigUint = FromPrimitive::from_usize(0).unwrap();
    assert_eq!(n.bits(), 0);
    let n: BigUint = FromPrimitive::from_usize(1).unwrap();
    assert_eq!(n.bits(), 1);
    let n: BigUint = FromPrimitive::from_usize(3).unwrap();
    assert_eq!(n.bits(), 2);
    let n: BigUint = BigUint::from_str_radix("4000000000", 16).unwrap();
    assert_eq!(n.bits(), 39);
    let one: BigUint = One::one();
    assert_eq!((one << 426).bits(), 427);
}

#[test]
fn test_rand() {
    let mut rng = thread_rng();
    let _n: BigUint = rng.gen_biguint(137);
    assert!(rng.gen_biguint(0).is_zero());
}

#[test]
fn test_rand_range() {
    let mut rng = thread_rng();

    for _ in 0..10 {
        assert_eq!(rng.gen_bigint_range(&FromPrimitive::from_usize(236).unwrap(),
                                        &FromPrimitive::from_usize(237).unwrap()),
                   FromPrimitive::from_usize(236).unwrap());
    }

    let l = FromPrimitive::from_usize(403469000 + 2352).unwrap();
    let u = FromPrimitive::from_usize(403469000 + 3513).unwrap();
    for _ in 0..1000 {
        let n: BigUint = rng.gen_biguint_below(&u);
        assert!(n < u);

        let n: BigUint = rng.gen_biguint_range(&l, &u);
        assert!(n >= l);
        assert!(n < u);
    }
}

#[test]
#[should_panic]
fn test_zero_rand_range() {
    thread_rng().gen_biguint_range(&FromPrimitive::from_usize(54).unwrap(),
                                   &FromPrimitive::from_usize(54).unwrap());
}

#[test]
#[should_panic]
fn test_negative_rand_range() {
    let mut rng = thread_rng();
    let l = FromPrimitive::from_usize(2352).unwrap();
    let u = FromPrimitive::from_usize(3513).unwrap();
    // Switching u and l should fail:
    let _n: BigUint = rng.gen_biguint_range(&u, &l);
}

fn test_mul_divide_torture_count(count: usize) {
    use rand::{SeedableRng, StdRng, Rng};

    let bits_max = 1 << 12;
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    for _ in 0..count {
        // Test with numbers of random sizes:
        let xbits = rng.gen_range(0, bits_max);
        let ybits = rng.gen_range(0, bits_max);

        let x = rng.gen_biguint(xbits);
        let y = rng.gen_biguint(ybits);

        if x.is_zero() || y.is_zero() {
            continue;
        }

        let prod = &x * &y;
        assert_eq!(&prod / &x, y);
        assert_eq!(&prod / &y, x);
    }
}

#[test]
fn test_mul_divide_torture() {
    test_mul_divide_torture_count(1000);
}

#[test]
#[ignore]
fn test_mul_divide_torture_long() {
    test_mul_divide_torture_count(1000000);
}
