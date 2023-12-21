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

impl AsInput<1> for f64 {
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
pub struct Input<const N: usize>(Box<[f64; N]>);

impl<const N: usize> From<[f64; N]> for Input<N> {
    fn from(value: [f64; N]) -> Self {
        Self(Box::new(value))
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for Input<N> {
    type Error = anyhow::Error;

    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        value
            .try_into()
            .map(Self)
            .map_err(|v| anyhow!("Input length must be {N} elements. Got {}", v.len()))
    }
}

impl<'a, const N: usize> Input<N> {
    /// # Panics
    ///
    /// Panics if the length of the `val` slice doesn't match `IN`.
    pub fn new(vec: Vec<f64>) -> Self {
        Self::try_from(vec).unwrap()
    }

    /// Returns all input elements as a slice.
    pub fn as_slice(&self) -> &[f64] {
        self
    }
}

impl<const N: usize> AsRef<[f64; N]> for Input<N> {
    fn as_ref(&self) -> &[f64; N] {
        &self.0
    }
}

impl<const I: usize> Deref for Input<I> {
    type Target = [f64];

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
        let input = Input::<TOO_BIG_FOR_STACK>::try_from(vec![1.0; TOO_BIG_FOR_STACK]).unwrap();
        let arr = input.as_ref();
        println!("{}", arr[TOO_BIG_FOR_STACK - 1]);
    }

    #[test]
    #[should_panic = "array is too big for the stack"]
    #[ignore = "array is too big for the stack"]
    fn get_arr_panic() {
        let input = Input::<TOO_BIG_FOR_STACK>::try_from(vec![1.0; TOO_BIG_FOR_STACK]).unwrap();
        let arr = *input.0;
        println!("{}", arr[TOO_BIG_FOR_STACK - 1]);
    }
}
