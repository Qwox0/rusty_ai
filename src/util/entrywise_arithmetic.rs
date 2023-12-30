use matrix::Num;

pub trait EntryAdd<Rhs = Self>: Sized {
    /// performs addition entrywise and returns result.
    /// "return `self` + `rhs`"
    fn add_entries(self, rhs: Rhs) -> Self;
}

impl<X: Num> EntryAdd<&[X]> for Vec<X> {
    fn add_entries(mut self, rhs: &[X]) -> Self {
        assert_eq!(self.len(), rhs.len());
        for (x, &rhs) in self.iter_mut().zip(rhs) {
            *x += rhs;
        }
        self
    }
}
