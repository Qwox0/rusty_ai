pub trait AssignEach<'a, T: 'a>: Iterator<Item = &'a mut T> + Sized {
    fn assign_each_ref<F>(self, mut f: F)
    where F: FnMut(&T) -> T {
        self.for_each(|x| *x = f(&x))
    }

    fn assign_each<F>(self, mut f: F)
    where
        T: Copy,
        F: FnMut(T) -> T,
    {
        self.for_each(|x| *x = f(x.clone()))
    }
}

impl<'a, T: 'a, I: Iterator<Item = &'a mut T>> AssignEach<'a, T> for I {}
