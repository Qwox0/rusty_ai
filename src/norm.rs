use const_tensor::{Float, Num};
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
    pub fn calculate<F: Float>(self, elements: impl IntoIterator<Item = F>) -> F {
        let elements = elements.into_iter();
        match self {
            Norm::Infinity => elements.map(F::abs).reduce(F::max).unwrap_or(F::zero()),
            Norm::One => elements.map(F::abs).sum(),
            Norm::Two => elements.map(|x| x * x).sum::<F>().sqrt(),
            Norm::Integer(i) => {
                elements.map(F::abs).map(|x| x.powi(i)).sum::<F>().powf(i.cast::<F>().recip())
            },
            Norm::Float(f) => {
                let f = f.cast();
                elements.map(F::abs).map(|x| x.powf(f)).sum::<F>().powf(f.recip())
            },
        }
    }
}
