// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(custom_derive, plugin)]
#![plugin(num_macros)]

extern crate num;

#[derive(Debug, PartialEq, NumFromPrimitive)]
enum Color {
    Red,
    Blue,
    Green,
}

#[test]
fn test_from_primitive() {
    let v: Vec<Option<Color>> = vec![
        num::FromPrimitive::from_u64(0),
        num::FromPrimitive::from_u64(1),
        num::FromPrimitive::from_u64(2),
        num::FromPrimitive::from_u64(3),
    ];

    assert_eq!(
        v,
        vec![Some(Color::Red), Some(Color::Blue), Some(Color::Green), None]
    );
}
