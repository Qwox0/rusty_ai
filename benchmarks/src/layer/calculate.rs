use crate::constants::*;
use crate::macros::make_benches;
use rusty_ai::layer::Layer;

pub trait Apply: Sized {
    type Elem: Clone;
    fn apply_mut(self, f: impl FnMut(&mut Self::Elem)) -> Self;
    fn apply(self, mut f: impl FnMut(&Self::Elem) -> Self::Elem) -> Self {
        self.apply_mut(|a| *a = f(a))
    }
    fn apply2(self, mut f: impl FnMut(Self::Elem) -> Self::Elem) -> Self {
        self.apply_mut(|a| *a = f(a.clone()))
    }
}

impl<T: Clone> Apply for Vec<T> {
    type Elem = T;

    fn apply_mut(mut self, f: impl FnMut(&mut Self::Elem)) -> Self {
        self.iter_mut().for_each(f);
        self
    }
}

trait MatrixBenchmarks {
    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64>;
    fn calculate2(&self, inputs: Vec<f64>) -> Vec<f64>;
    fn calculate3(&self, inputs: Vec<f64>) -> Vec<f64>;
}

impl MatrixBenchmarks for Layer {
    fn calculate(&self, inputs: Vec<f64>) -> Vec<f64> {
        (self.get_weights() * inputs)
            .apply(|x| x + self.get_bias())
            .apply(|a| (self.get_activation_function())(*a))
    }
    fn calculate2(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut res = self.get_weights() * inputs;
        for x in res.iter_mut() {
            *x = (self.get_activation_function())(*x + self.get_bias())
        }
        res
    }
    fn calculate3(&self, inputs: Vec<f64>) -> Vec<f64> {
        (self.get_weights() * inputs)
            .into_iter()
            .map(|x| x + self.get_bias())
            .map(&self.get_activation_function())
            .collect()
    }
}

make_benches! {
    Layer;
    Layer::new_hidden(LAYER_CALC_IN, LAYER_CALC_OUT, rusty_ai::activation_function::ActivationFunction::ReLU2);
    calculate: fill_rand(LAYER_CALC_VEC)
    calculate2: fill_rand(LAYER_CALC_VEC)
    calculate3: fill_rand(LAYER_CALC_VEC)
}
