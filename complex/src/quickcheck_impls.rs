use Complex;
use quickcheck::{Arbitrary, Gen};
use traits::Num;

impl<T: Arbitrary + Num> Arbitrary for Complex<T> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Complex::new(T::arbitrary(g), T::arbitrary(g))
    }

    fn shrink(&self) -> Box<Iterator<Item = Self>> {
        Box::new(Shrinker::new(self.re.clone(), self.im.clone()))
    }
}

struct Shrinker<T: Arbitrary + Num> {
    mode: bool,
    re: T,
    im: T,
    re_iter: Option<Box<Iterator<Item = T>>>,
    im_iter: Option<Box<Iterator<Item = T>>>,
}

impl<T: Arbitrary + Num> Shrinker<T> {
    fn new(re: T, im: T) -> Self {
        Shrinker {
            mode: true,
            re: re.clone(),
            im: im,
            re_iter: Some(re.shrink()),
            im_iter: None,
        }
    }
}

impl<T: Arbitrary + Num> Iterator for Shrinker<T> {
    type Item = Complex<T>;

    fn next(&mut self) -> Option<Complex<T>> {
        match self.mode {
            true => if let Some(re) = self.re_iter.as_mut().unwrap().next() {
                Some(Complex::new(re, self.im.clone()))
            } else {
                self.mode = false;
                self.re_iter = None;
                self.im_iter = Some(self.im.clone().shrink());
                self.next()
            },
            false => if let Some(im) = self.im_iter.as_mut().unwrap().next() {
                Some(Complex::new(self.re.clone(), im))
            } else if let Some(re) = self.re_iter.as_mut().unwrap().next() {
                self.re = re;
                self.next()
            } else {
                None
            }
        }
    }
}
