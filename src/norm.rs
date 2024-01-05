/*
use matrix::{Float, Num};
use serde::{Deserialize, Serialize};

/// see <https://en.wikipedia.org/wiki/Norm_(mathematics)>
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Norm {
    /// Infinity norm
    Infinity,

    /// Absolute-value norm
    One,

    /// Euclidean norm
    Two,

    /// p-norm for any integer
    Integer(i32),

    /// p-norm for any float
    Float(f64),
}

impl Norm {
    /// Calculates the Norm of type `self` for the `elements`.
    pub fn calculate<X: Float>(self, elements: impl IntoIterator<Item = X>) -> X {
        let elements = elements.into_iter();
        match self {
            Norm::Infinity => elements.map(X::abs).reduce(X::max).unwrap_or(X::zero()),
            Norm::One => elements.map(X::abs).sum(),
            Norm::Two => elements.map(|x| x * x).sum::<X>().sqrt(),
            Norm::Integer(i) => {
                elements.map(X::abs).map(|x| x.powi(i)).sum::<X>().powf(i.cast::<X>().recip())
            },
            Norm::Float(f) => {
                let f = f.cast();
                elements.map(X::abs).map(|x| x.powf(f)).sum::<X>().powf(f.recip())
            },
        }
    }
}
*/
