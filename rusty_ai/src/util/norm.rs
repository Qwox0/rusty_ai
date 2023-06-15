/// see [https://en.wikipedia.org/wiki/Norm_(mathematics)]
#[derive(Debug, Clone, Copy)]
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
    pub fn calculate(&self, elements: impl Iterator<Item = f64>) -> f64 {
        match self {
            Norm::Infinity => elements.map(f64::abs).reduce(f64::max).unwrap_or(0.0),
            Norm::One => elements.map(f64::abs).sum(),
            Norm::Two => elements.map(|x| x * x).sum::<f64>().sqrt(),
            Norm::Integer(i) => elements
                .map(f64::abs)
                .map(|x| x.powi(*i))
                .sum::<f64>()
                .powf(1.0 / *i as f64),
            Norm::Float(f) => elements
                .map(f64::abs)
                .map(|x| x.powf(*f))
                .sum::<f64>()
                .powf(f.recip()),
        }
    }

    pub fn calculate_clip_factor(&self, ) {

    }
}
