use crate::util::impl_fn_traits;

#[derive(Debug, Clone, Copy, Default)]
pub enum ActivationFn {
    /// Identity(x) = x
    /// Identity(x) = 1
    #[default]
    Identity,

    /// values: (ReLU'(0))
    /// ReLU(x) = max(0, x)
    /// ReLU'(0) := self.0
    ReLU(f64),

    /// values: (leak_rate, LeakyReLU'(0))
    /// LeakyReLU(x) = max(self.0 * x, x)
    /// LeakyReLU'(0) := self.1
    LeakyReLU(f64, f64),

    /// Sigmoid(x) = 1/(1 + exp(-x)) = exp(x)/(exp(x) + 1)
    /// Sigmoid'(x) = e^(-x)/(1+e^(-x))^2 = e^x/(1+e^x)^2
    Sigmoid,
}

impl ActivationFn {
    pub const fn default_relu() -> ActivationFn {
        ActivationFn::ReLU(1.0)
    }
    pub const fn default_leaky_relu() -> ActivationFn {
        ActivationFn::LeakyReLU(0.01, 1.0)
    }

    pub fn calculate(&self, input: f64) -> f64 {
        use ActivationFn::*;
        match self {
            Identity => input,
            ReLU(_) => match input {
                x if x.is_sign_positive() => x,
                _ => 0.0,
            },
            LeakyReLU(leak_rate, _) => match input {
                x if x.is_sign_positive() => x,
                x => leak_rate * x,
            },
            Sigmoid => 1.0 / (1.0 + f64::exp(-input)),
        }
    }

    pub fn derivative(&self, input: f64) -> f64 {
        use ActivationFn::*;
        #[allow(illegal_floating_point_literal_pattern)] // for pattern: 0.0 => ...
        match self {
            Identity => 1.0,
            ReLU(d0) => match input {
                0.0 => *d0,
                x => x.is_sign_positive() as u8 as f64,
            },
            LeakyReLU(leak_rate, d0) => match input {
                0.0 => *d0,
                x if x.is_sign_positive() => 1.0,
                _ => *leak_rate,
            },
            Sigmoid => {
                let exp = input.exp();
                let exp_plus_1 = exp + 1.0;
                exp / (exp_plus_1 * exp_plus_1) // Sigmoid'(x) = e^x/(1+e^x)^2
            }
        }
    }

    pub fn call(&self, args: (f64,)) -> f64 {
        self.calculate(args.0)
    }
    pub fn call_ref(&self, args: (&f64,)) -> f64 {
        self.calculate(*args.0)
    }
}

impl_fn_traits!(Fn<(f64,)> -> f64: ActivationFn => call);
impl_fn_traits!(Fn<(&f64,)> -> f64: ActivationFn => call_ref);

impl std::fmt::Display for ActivationFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ActivationFn::*;
        match self {
            Identity => write!(f, "Identity"),
            ReLU(d0) => write!(f, "ReLU (ReLU'(0)={})", d0),
            LeakyReLU(a, d0) => write!(f, "Leaky ReLU (a={}; f'(0)={})", a, d0),
            Sigmoid => write!(f, "Sigmoid"),
        }
    }
}
