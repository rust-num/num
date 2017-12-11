extern crate rustc_version;

use rustc_version::version_matches;


fn main() {
    if version_matches(">=1.12.0") {
        println!("cargo:rustc-cfg=impl_sum_product_for_bigints");
    }
}
