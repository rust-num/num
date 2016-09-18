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
    // panic!("{:?}", ast);

macro_rules! pathvec_std {
    ($cx:expr, $first:ident :: $($rest:ident)::+) => ({
        let mut v = pathvec!($($rest)::+);
        if let Some(s) = $cx.crate_root {
            v.insert(0, s);
        }
        v
    })
}

pub fn expand_deriving_from_primitive(cx: &mut ExtCtxt,
                                      span: Span,
                                      mitem: &MetaItem,
                                      item: &Annotatable,
                                      push: &mut FnMut(Annotatable))
{
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        is_unsafe: false,
        span: span,
        attributes: Vec::new(),
        path: path!(num::FromPrimitive),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "from_i64",
                is_unsafe: false,
                unify_fieldless_variants: false,
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: vec!(Literal(path_local!(i64))),
                ret_ty: Literal(Path::new_(pathvec_std!(cx, core::option::Option),
                                           None,
                                           vec!(Box::new(Self_)),
                                           true)),
                // #[inline] liable to cause code-bloat
                attributes: attrs.clone(),
                combine_substructure: combine_substructure(Box::new(|c, s, sub| {
                    cs_from("i64", c, s, sub)
                })),
            },
            MethodDef {
                name: "from_u64",
                is_unsafe: false,
                unify_fieldless_variants: false,
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: vec!(Literal(path_local!(u64))),
                ret_ty: Literal(Path::new_(pathvec_std!(cx, core::option::Option),
                                           None,
                                           vec!(Box::new(Self_)),
                                           true)),
                // #[inline] liable to cause code-bloat
                attributes: attrs,
                combine_substructure: combine_substructure(Box::new(|c, s, sub| {
                    cs_from("u64", c, s, sub)
                })),
            }
        ),
        associated_types: Vec::new(),
        supports_unions: false,
    };

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
