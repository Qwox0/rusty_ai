#[derive(Debug)]
pub enum ActivationFunction {
    /// ReLU(x) = max(0, x)
    /// ReLU'(0) = 0
    ReLU,
    /// ReLU2(x) = max(0, x)
    /// ReLU2'(0) = 1
    ReLU2,
    /// LeakyReLU(x) = max(leak_rate*x, x)
    /// LeakyReLU'(0) = 0
    LeakyReLU(f64), // leak_rate
    /// LeakyReLU2(x) = max(leak_rate*x, x)
    /// LeakyReLU2'(0) = 1
    LeakyReLU2(f64),
    /// Sigmoid(x) = 1/(1 + exp(-x)) = exp(x)/(exp(x) + 1)
    /// Sigmoid'(x) = e^(-x)/(1+e^(-x))^2 = e^x/(1+e^x)^2
    Sigmoid,
}

impl ActivationFunction {
    pub fn calculate(&self, input: f64) -> f64 {
        use ActivationFunction::*;
        match self {
            ReLU | ReLU2 => match input {
                x if x.is_sign_positive() => x,
                _ => 0.0,
            },
            LeakyReLU(leak_rate) | LeakyReLU2(leak_rate) => match input {
                x if x.is_sign_positive() => x,
                x => leak_rate * x,
            },
            Sigmoid => 1.0 / (1.0 + f64::exp(-input)),
        }
    }

    pub fn derivative(&self, input: f64) -> f64 {
        use ActivationFunction::*;
        match self {
            ReLU => input.is_sign_positive() as i32 as f64, // ReLU'(0) := 0
            ReLU2 => !input.is_sign_negative() as i32 as f64, // ReLU2'(0) := 1
            LeakyReLU(leak_rate) => match input {
                x if x.is_sign_positive() => 1.0,
                _ => *leak_rate, // LeakyReLU'(0) = leak_rate
            },
            LeakyReLU2(leak_rate) => match input {
                x if x.is_sign_positive() => 1.0,
                0.0 => 1.0,
                _ => *leak_rate,
            },
            Sigmoid => {
                let exp = input.exp();
                let exp_plus_1 = exp + 1.0;
                exp / (exp_plus_1 * exp_plus_1) // Sigmoid'(x) = e^x/(1+e^x)^2
            }
        }
    }
}

impl FnOnce<(f64,)> for ActivationFunction {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        self.calculate(args.0)
    }
}

impl FnMut<(f64,)> for ActivationFunction {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        self.calculate(args.0)
    }
}

impl Fn<(f64,)> for ActivationFunction {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        self.calculate(args.0)
    }
}
