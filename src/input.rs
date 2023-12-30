use anyhow::anyhow;
use std::ops::Deref;

/*
pub trait AsInput<const IN: usize> {
    fn try_as_input<'a>(&'a self) -> Option<Input<'a, IN>>;

    fn as_input<'a>(&'a self) -> Input<'a, IN> {
        Self::try_as_input(&self).expect("The length must match IN")
    }
}

impl<const IN: usize, T: AsRef<[f64]>> AsInput<IN> for T {
    /// # Panics
    ///
    /// Panics if the length of the `val` slice doesn't match `IN`.
    fn try_as_input<'a>(&'a self) -> Option<Input<'a, IN>> {
        Input::try_new(self.as_ref())
    }
}

impl AsInput<X, 1> for f64 {
    fn try_as_input<'a>(&'a self) -> Option<Input<'a, 1>> {
        Some(self.as_input())
    }

    fn as_input<'a>(&'a self) -> Input<'a, 1> {
        Input::new_unchecked(std::slice::from_ref(self))
    }
}
*/

/// an array stored on the heap.
#[derive(Debug, Clone, derive_more::Index)]
pub struct Input<X, const N: usize>(Box<[X; N]>);

impl<X, const N: usize> From<[X; N]> for Input<X, N> {
    fn from(value: [X; N]) -> Self {
        Self(Box::new(value))
    }
}

impl<X, const N: usize> TryFrom<Vec<X>> for Input<X, N> {
    type Error = anyhow::Error;

    fn try_from(value: Vec<X>) -> Result<Self, Self::Error> {
        value
            .try_into()
            .map(Self)
            .map_err(|v| anyhow!("Input length must be {N} elements. Got {}", v.len()))
    }
}

impl<'a, X, const N: usize> Input<X, N> {
    /// Create a new neural network [`Input`].
    pub fn new(elements: Box<[X; N]>) -> Self {
        Input(elements)
    }

    /// # Panics
    ///
    /// Panics if the length of the `val` slice doesn't match `IN`.
    pub fn try_from_vec(vec: Vec<X>) -> Self {
        Self::try_from(vec).unwrap()
    }

    /// Returns all input elements as a slice.
    pub fn as_slice(&self) -> &[X] {
        self
    }
}

impl<X, const N: usize> AsRef<[X; N]> for Input<X, N> {
    fn as_ref(&self) -> &[X; N] {
        &self.0
    }
}

impl<X, const I: usize> Deref for Input<X, I> {
    type Target = [X];

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

#[cfg(test)]
mod tests {
    use super::Input;

    const TOO_BIG_FOR_STACK: usize = 1_000_000;

    #[test]
    fn get_arr_ref() {
        let input =
            Input::<f64, TOO_BIG_FOR_STACK>::try_from(vec![1.0; TOO_BIG_FOR_STACK]).unwrap();
        let arr = input.as_ref();
        println!("{}", arr[TOO_BIG_FOR_STACK - 1]);
    }

    #[test]
    #[should_panic = "array is too big for the stack"]
    #[ignore = "array is too big for the stack"]
    fn get_arr_panic() {
        let input =
            Input::<f64, TOO_BIG_FOR_STACK>::try_from(vec![1.0; TOO_BIG_FOR_STACK]).unwrap();
        let arr = *input.0;
        println!("{}", arr[TOO_BIG_FOR_STACK - 1]);
    }
}
