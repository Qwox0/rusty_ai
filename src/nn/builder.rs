//! # Neural network builder module

use super::{flatten::Flatten, linear::Linear, relu::ReLU, NNComponent, NN};
use crate::Initializer;
use const_tensor::{matrix, vector, Element, Float, Len, Matrix, Tensor, TensorData, Vector};
use half::{bf16, f16};
use markers::*;
use rand::{
    rngs::{StdRng, ThreadRng},
    SeedableRng,
};
use std::marker::PhantomData;

/// Markers uses by [`NNBuilder`].
pub mod markers {
    /// Seed used for RNG.
    pub struct NoRng;
}

/// Neural Network Builder
#[derive(Debug, Clone)]
pub struct NNBuilder<X: Element, NNIN: Tensor<X>, OUT: Tensor<X>, C, RNG> {
    components: C,
    rng: RNG,
    _out: PhantomData<OUT>,
    _marker: PhantomData<(X, NNIN)>,
}

impl<X: Element, NNIN: Tensor<X>> Default for NNBuilder<X, NNIN, NNIN, (), NoRng> {
    #[inline]
    fn default() -> Self {
        Self { components: (), rng: NoRng, _out: PhantomData, _marker: PhantomData }
    }
}

impl<X: Element, NNIN: Tensor<X>, RNG> NNBuilder<X, NNIN, NNIN, (), RNG> {
    /// Sets the [`rand::Rng`] used during initialization.
    #[inline]
    pub fn rng<R: rand::Rng>(self, rng: R) -> NNBuilder<X, NNIN, NNIN, (), R> {
        NNBuilder { rng, ..self }
    }

    /// Uses [`rand::thread_rng`] for during initialization.
    #[inline]
    pub fn thread_rng(self) -> NNBuilder<X, NNIN, NNIN, (), ThreadRng> {
        self.rng(rand::thread_rng())
    }

    /// Note: currently the same as `.thread_rng()`
    #[inline]
    pub fn default_rng(self) -> NNBuilder<X, NNIN, NNIN, (), ThreadRng> {
        self.thread_rng()
    }

    /// Uses seeded rng during initialization.
    #[inline]
    pub fn seeded_rng(self, seed: u64) -> NNBuilder<X, NNIN, NNIN, (), StdRng> {
        self.rng(StdRng::seed_from_u64(seed))
    }

    /// Sets `NX` as nn element type.
    pub fn element_type<X2: Element, NNIN2: Tensor<X2>>(
        self,
    ) -> NNBuilder<X2, NNIN2, NNIN2, (), RNG> {
        NNBuilder { _marker: PhantomData, _out: PhantomData, ..self }
    }

    /// Sets [`f32`] as nn element type.
    pub fn normal_precision<NNIN2: Tensor<f32>>(self) -> NNBuilder<f32, NNIN2, NNIN2, (), RNG> {
        self.element_type()
    }

    /// Sets [`f64`] as nn element type.
    pub fn double_precision<NNIN2: Tensor<f64>>(self) -> NNBuilder<f64, NNIN2, NNIN2, (), RNG> {
        self.element_type()
    }

    /// Sets [`half::f16`] as nn element type.
    pub fn half_precision<NNIN2: Tensor<f16>>(self) -> NNBuilder<f16, NNIN2, NNIN2, (), RNG> {
        self.element_type()
    }

    /// Sets [`half::bf16`] as nn element type.
    pub fn bhalf_precision<NNIN2: Tensor<bf16>>(self) -> NNBuilder<bf16, NNIN2, NNIN2, (), RNG> {
        self.element_type()
    }
}

impl<X: Element, NNIN: Tensor<X>, OUT: Tensor<X>, PREV: NNComponent<X, NNIN, OUT>, RNG>
    NNBuilder<X, NNIN, OUT, PREV, RNG>
{
    #[inline]
    pub fn add_component<C: NNComponent<X, NNIN, OUT2>, OUT2: Tensor<X>>(
        self,
        c: impl FnOnce(PREV) -> C,
    ) -> NNBuilder<X, NNIN, OUT2, C, RNG> {
        NNBuilder { components: c(self.components), _out: PhantomData, ..self }
    }

    pub fn relu(self) -> NNBuilder<X, NNIN, OUT, ReLU<PREV>, RNG> {
        NNBuilder { components: ReLU { prev: self.components }, ..self }
    }

    pub fn flatten(
        self,
    ) -> NNBuilder<X, NNIN, Vector<X, { <OUT::Data as TensorData<X>>::LEN }>, Flatten<OUT, PREV>, RNG>
    {
        NNBuilder {
            components: Flatten { prev: self.components, _tensor: PhantomData },
            _out: PhantomData,
            ..self
        }
    }

    pub fn build(self) -> NN<X, NNIN, OUT, PREV> {
        NN { components: self.components, _marker: PhantomData }
    }
}

impl<X, NNIN, const IN: usize, PREV, RNG> NNBuilder<X, NNIN, Vector<X, IN>, PREV, RNG>
where
    X: Element,
    NNIN: Tensor<X>,
    PREV: NNComponent<X, NNIN, Vector<X, IN>>,
    RNG: rand::Rng,
{
    pub fn layer_from_parts<const N: usize>(
        self,
        weights: Matrix<X, IN, N>,
        bias: Vector<X, N>,
    ) -> NNBuilder<X, NNIN, Vector<X, N>, Linear<X, IN, N, PREV>, RNG> {
        let components = Linear { prev: self.components, weights, bias };
        NNBuilder { components, _out: PhantomData, ..self }
    }

    pub fn layer_from_arr<const N: usize>(
        self,
        weights: [[X; IN]; N],
        bias: [X; N],
    ) -> NNBuilder<X, NNIN, Vector<X, N>, Linear<X, IN, N, PREV>, RNG> {
        self.layer_from_parts(Matrix::new(weights), Vector::new(bias))
    }

    /// Uses [`Initializer`] to add a new [`Linear`] layer to the neural network.
    fn layer<const N: usize>(
        mut self,
        weights_init: Initializer<X, Matrix<X, IN, N>>,
        bias_init: Initializer<X, Vector<X, N>>,
    ) -> NNBuilder<X, NNIN, Vector<X, N>, Linear<X, IN, N, PREV>, RNG>
    where
        X: Float,
        matrix<X, IN, N>: Len<{ IN * N }>,
        vector<X, N>: Len<N>,
        rand_distr::StandardNormal: rand_distr::Distribution<X>,
    {
        let weights = weights_init.init(&mut self.rng, IN, N);
        let bias = bias_init.init(&mut self.rng, IN, N);
        self.layer_from_parts(weights, bias)
    }
}

/*



macro_rules! activation_function {
    ( $( $fn_name:ident -> $variant:ident $( { $($arg:ident : $ty:ty),+ } )? : $variant_str:expr );+ $(;)? ) => {
        $(
            #[doc = "Sets the `"]
            #[doc = $variant_str]
            #[doc = "` activation function for the previously defined layer."]
            #[inline]
            pub fn $fn_name(self $(, $($arg : $ty),+)? ) -> BuilderNoParts<X, IN, RNG> {
                self.activation_function(ActivationFn::$variant $({ $($arg),+ })?)
            }
         )+
    };
}

impl<X: Element, const IN: usize, RNG: rand::Rng> BuilderWithParts<X, IN, RNG> {
    activation_function! {
        identity -> Identity : "Identity" ;
        relu -> ReLU : "ReLU" ;
        leaky_relu -> LeakyReLU { leak_rate: X } : "LeakyReLU" ;
        sigmoid -> Sigmoid : "Sigmoid" ;
        softmax -> Softmax : "Softmax" ;
        log_softmax -> LogSoftmax : "LogSoftmax";
    }

    /// Sets the [`ActivationFn`] for the previously defined layer.
    pub fn activation_function(self, af: ActivationFn<X>) -> BuilderNoParts<X, IN, RNG> {
        let LayerParts { weights, bias } = self.layer_parts;
        let layer = Layer::new(weights, bias, af);
        NNBuilder { layer_parts: NoLayerParts, ..self }._layer(layer)
    }

    /// Uses the `self.default_activation_function` for the previously defined layer.
    ///
    /// This function gets called automatically if no activation function is provided.
    #[inline]
    pub fn use_default_activation_function(self) -> BuilderNoParts<X, IN, RNG> {
        let default = self.default_activation_function;
        self.activation_function(default)
    }
}
*/
