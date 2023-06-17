pub trait EntryAdd<Rhs = Self>: Sized {
    /// performs addition entrywise by mutating `self` in place.
    /// "`self` = `self` + `rhs`"
    fn add_entries_mut(&mut self, rhs: Rhs) -> &mut Self;
    /// performs addition entrywise and returns result.
    /// "return `self` + `rhs`"
    fn add_entries(mut self, rhs: Rhs) -> Self {
        self.add_entries_mut(rhs);
        self
    }
}

trait EntrySub<Rhs = Self>: Sized {
    /// performs subtraction entrywise by mutating `self` in place.
    /// "`self` = `self` - `rhs`"
    fn sub_entries_mut(&mut self, rhs: Rhs) -> &mut Self;
    /// performs subtraction entrywise and returns result.
    /// "return `self` - `rhs`"
    fn sub_entries(mut self, rhs: Rhs) -> Self {
        self.sub_entries_mut(rhs);
        self
    }
}

trait EntryMul<Rhs = Self>: Sized {
    /// performs multiplication entrywise by mutating `self` in place.
    /// "`self` = `self` * `rhs`"
    fn mul_entries_mut(&mut self, rhs: Rhs) -> &mut Self;
    /// performs multiplication entrywise and returns result.
    /// "return `self` * `rhs`"
    fn mul_entries(mut self, rhs: Rhs) -> Self {
        self.mul_entries_mut(rhs);
        self
    }
}

trait EntryDiv<Rhs = Self>: Sized {
    /// performs division entrywise by mutating `self` in place.
    /// "`self` = `self` / `rhs`"
    fn div_entries_mut(&mut self, rhs: Rhs) -> &mut Self;
    /// performs division entrywise and returns result.
    /// "return `self` / `rhs`"
    fn div_entries(mut self, rhs: Rhs) -> Self {
        self.div_entries_mut(rhs);
        self
    }
}

macro_rules! impl_entry_arithmetic_trait {
    ( $trait:ident : $trait_fn:ident $op:tt ) => {
        impl $trait<&f64> for f64 {
            fn $trait_fn(&mut self, rhs: &f64) -> &mut Self {
                *self $op rhs;
                self
            }
        }

        impl<'a, T: $trait<&'a T> + 'a> $trait<&'a Vec<T>> for Vec<T> {
            fn $trait_fn(&mut self, rhs: &'a Vec<T>) -> &mut Self {
                assert_eq!(self.len(), rhs.len());
                for (x, rhs) in self.iter_mut().zip(rhs) {
                    x.$trait_fn(rhs);
                }
                self
            }
        }

        impl<T: for<'a> $trait<&'a T>> $trait<T> for T {
            fn $trait_fn(&mut self, rhs: T) -> &mut Self {
                self.$trait_fn(&rhs)
            }
        }
    };
}

impl_entry_arithmetic_trait! { EntryAdd : add_entries_mut += }
impl_entry_arithmetic_trait! { EntrySub : sub_entries_mut -= }
impl_entry_arithmetic_trait! { EntryMul : mul_entries_mut *= }
impl_entry_arithmetic_trait! { EntryDiv : div_entries_mut /= }
