extern crate rustc_version;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use rustc_version::{version, version_matches};


fn write_build_info() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("build_info.rs");
    let mut f = File::create(&dest_path).unwrap();

    write!(&mut f, "
        pub const RUSTC_VERSION: &'static str = {:?};
    ", version().to_string()).unwrap();
}

fn print_cfg_options() {
    if version_matches(">= 1.12.0") {
        println!("cargo:rustc-cfg=impl_sum_product_for_bigints");
    }
}


fn main() {
    write_build_info();
    print_cfg_options();
}
