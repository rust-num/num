use Ratio;
use integer::Integer;
use quickcheck::{Arbitrary, Gen};

impl<T: Arbitrary + Integer> Arbitrary for Ratio<T> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Ratio::new(T::arbitrary(g), T::arbitrary(g))
    }

    fn shrink(&self) -> Box<Iterator<Item = Self>> {
        Box::new(Shrinker::new(self.numer.clone(), self.denom.clone()))
    }
}

struct Shrinker<T: Arbitrary + Integer> {
    mode: bool,
    numer: T,
    denom: T,
    numer_iter: Option<Box<Iterator<Item = T>>>,
    denom_iter: Option<Box<Iterator<Item = T>>>,
}

impl<T: Arbitrary + Integer> Shrinker<T> {
    fn new(numer: T, denom: T) -> Self {
        Shrinker {
            mode: true,
            numer: numer.clone(),
            denom: denom,
            numer_iter: Some(numer.shrink()),
            denom_iter: None,
        }
    }
}

impl<T: Arbitrary + Integer> Iterator for Shrinker<T> {
    type Item = Ratio<T>;

    fn next(&mut self) -> Option<Ratio<T>> {
        match self.mode {
            true => if let Some(numer) = self.numer_iter.as_mut().unwrap().next() {
                Some(Ratio::new(numer, self.denom.clone()))
            } else {
                self.mode = false;
                self.numer_iter = None;
                self.denom_iter = Some(self.denom.clone().shrink());
                self.next()
            },
            false => if let Some(denom) = self.denom_iter.as_mut().unwrap().next() {
                Some(Ratio::new(self.numer.clone(), denom))
            } else if let Some(numer) = self.numer_iter.as_mut().unwrap().next() {
                self.numer = numer;
                self.next()
            } else {
                None
            }
        }
    }
}
