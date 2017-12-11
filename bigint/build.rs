extern crate rustc_version;

use rustc_version::{Version, version};


fn main() {
    if version().unwrap() >= Version::parse("1.12.0").unwrap() {
        println!("cargo:rustc-cfg=impl_sum_product_for_bigints");
    }
}
