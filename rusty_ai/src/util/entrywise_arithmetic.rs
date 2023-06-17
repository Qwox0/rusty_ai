pub trait EntryAdd<Rhs = Self>: Sized {
    /// performs addition entrywise and returns result.
    /// "return `self` + `rhs`"
    fn add_entries(self, rhs: Rhs) -> Self;
}

impl EntryAdd<&Vec<f64>> for Vec<f64> {
    fn add_entries(mut self, rhs: &Vec<f64>) -> Self {
        assert_eq!(self.len(), rhs.len());
        for (x, rhs) in self.iter_mut().zip(rhs) {
            *x += rhs;
        }
        self
    }
}
