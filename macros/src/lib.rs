// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "rustc-macro"]
#![feature(rustc_macro, rustc_macro_lib)]

extern crate syn;
#[macro_use]
extern crate quote;
extern crate rustc_macro;

use rustc_macro::TokenStream;

use syn::Body::Enum;

#[rustc_macro_derive(FromPrimitive)]
pub fn from_primitive(input: TokenStream) -> TokenStream {
    let source = input.to_string();

    let ast = syn::parse_item(&source).unwrap();

    let mut idx = 0;
    let variants: Vec<_> = variants.iter()
        .map(|variant| {
            let ident = &variant.ident;
            let tt = quote!(#idx => Some(#name::#ident));
            idx += 1;
            tt
        })
        .collect();

    let res = quote! {
        #ast

        impl ::num::traits::FromPrimitive for #name {
            fn from_i64(n: i64) -> Option<Self> {
                Self::from_u64(n as u64)
            }

            fn from_u64(n: u64) -> Option<Self> {
                match n {
                    #(variants,)*
                    _ => None,
                }
            }
        }
    };

    res.to_string().parse().unwrap()
}
