//! # Neural network builder module

use super::{flatten::Flatten, linear::Linear, relu::ReLU, LeakyReLU, NNComponent, Sigmoid, NN};
use crate::Initializer;
use const_tensor::{
    matrix, vector, Element, Float, Len, Matrix, Shape, Tensor, TensorData, Vector,
};
use half::{bf16, f16};
use markers::*;
use rand::{
    rngs::{StdRng, ThreadRng},
    SeedableRng,
};
use std::marker::PhantomData;

/// Markers uses by [`NNBuilder`].
pub mod markers {
    #[allow(unused_imports)]
    use super::NNBuilder;

    /// Marker for [`NNBuilder`] without RNG.
    pub struct NoRng;

    /// Marker for [`NNBuilder`] without Shape.
    pub struct NoShape;
}

/// Neural Network Builder
#[derive(Debug, Clone)]
pub struct NNBuilder<X, NNIN, OUT, C, RNG> {
    components: C,
    rng: RNG,
    _element: PhantomData<X>,
    _nnin: PhantomData<NNIN>,
    _out: PhantomData<OUT>,
}

impl Default for NNBuilder<f32, NoShape, NoShape, (), NoRng> {
    #[inline]
    fn default() -> Self {
        Self {
            components: (),
            rng: NoRng,
            _element: PhantomData,
            _nnin: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<X: Element, NNIN, OUT, C, RNG> NNBuilder<X, NNIN, OUT, C, RNG> {
    /// Sets the [`rand::Rng`] used during initialization.
    #[inline]
    pub fn rng<R: rand::Rng>(self, rng: R) -> NNBuilder<X, NNIN, OUT, C, R> {
        NNBuilder { rng, ..self }
    }

    /// Uses [`rand::thread_rng`] for during initialization.
    #[inline]
    pub fn thread_rng(self) -> NNBuilder<X, NNIN, OUT, C, ThreadRng> {
        self.rng(rand::thread_rng())
    }

    /// Note: currently the same as `.thread_rng()`
    #[inline]
    pub fn default_rng(self) -> NNBuilder<X, NNIN, OUT, C, ThreadRng> {
        self.thread_rng()
    }

    /// Uses seeded rng during initialization.
    #[inline]
    pub fn seeded_rng(self, seed: u64) -> NNBuilder<X, NNIN, OUT, C, StdRng> {
        self.rng(StdRng::seed_from_u64(seed))
    }
}

impl<X: Element, RNG> NNBuilder<X, NoShape, NoShape, (), RNG> {
    /// Sets `NX` as nn element type.
    pub fn element_type<X2: Element>(self) -> NNBuilder<X2, NoShape, NoShape, (), RNG> {
        NNBuilder { _element: PhantomData, ..self }
    }

    /// Sets [`f32`] as nn element type.
    pub fn normal_precision(self) -> NNBuilder<f32, NoShape, NoShape, (), RNG> {
        self.element_type()
    }

    /// Sets [`f64`] as nn element type.
    pub fn double_precision(self) -> NNBuilder<f64, NoShape, NoShape, (), RNG> {
        self.element_type()
    }

    /// Sets [`half::f16`] as nn element type.
    pub fn half_precision(self) -> NNBuilder<f16, NoShape, NoShape, (), RNG> {
        self.element_type()
    }

    /// Sets [`half::bf16`] as nn element type.
    pub fn bhalf_precision(self) -> NNBuilder<bf16, NoShape, NoShape, (), RNG> {
        self.element_type()
    }

    /// Sets the [`Shape`] of the input tensor.
    pub fn input_shape<S: Shape>(self) -> NNBuilder<X, S, S, (), RNG> {
        NNBuilder { _nnin: PhantomData, _out: PhantomData, ..self }
    }
}

impl<X: Element, NNIN: Shape, OUT: Shape, PREV: NNComponent<X, NNIN, OUT>, RNG>
    NNBuilder<X, NNIN, OUT, PREV, RNG>
{
    /// Adds a new [`NNComponent`] to the neural network.
    #[inline]
    pub fn add_component<C: NNComponent<X, NNIN, OUT2>, OUT2: Shape>(
        self,
        c: impl FnOnce(PREV) -> C,
    ) -> NNBuilder<X, NNIN, OUT2, C, RNG> {
        NNBuilder { components: c(self.components), _out: PhantomData, ..self }
    }

    /// Adds a new [`ReLU`] component to the neural network.
    pub fn relu(self) -> NNBuilder<X, NNIN, OUT, ReLU<PREV>, RNG>
    where ReLU<PREV>: NNComponent<X, NNIN, OUT> {
        self.add_component(ReLU::new)
    }

    /// Adds a new [`LeakyReLU`] component to the neural network.
    pub fn leaky_relu(self, leak_rate: X) -> NNBuilder<X, NNIN, OUT, LeakyReLU<X, PREV>, RNG>
    where LeakyReLU<X, PREV>: NNComponent<X, NNIN, OUT> {
        self.add_component(|prev| LeakyReLU { prev, leak_rate })
    }

    /// Adds a new [`Sigmoid`] component to the neural network.
    pub fn sigmoid(self) -> NNBuilder<X, NNIN, OUT, Sigmoid<PREV>, RNG>
    where Sigmoid<PREV>: NNComponent<X, NNIN, OUT> {
        self.add_component(Sigmoid::new)
    }

    /// Adds a new [`Flatten`] component to the neural network.
    pub fn flatten(self) -> NNBuilder<X, NNIN, [(); OUT::LEN], Flatten<OUT, PREV>, RNG>
    where Flatten<OUT, PREV>: NNComponent<X, NNIN, [(); OUT::LEN]> {
        self.add_component(Flatten::new)
    }

    /// Consumes the builder to create a new [`NN`].
    pub fn build(self) -> NN<X, NNIN, OUT, PREV> {
        NN::new(self.components)
    }
}

impl<X, NNIN, const IN: usize, PREV, RNG> NNBuilder<X, NNIN, [(); IN], PREV, RNG>
where
    X: Element,
    NNIN: Shape,
    PREV: NNComponent<X, NNIN, [(); IN]>,
{
    /// Adds a new [`Linear`] layer to the neural network.
    pub fn layer_from_parts<const N: usize>(
        self,
        weights: Matrix<X, IN, N>,
        bias: Vector<X, N>,
    ) -> NNBuilder<X, NNIN, [(); N], Linear<X, IN, N, PREV>, RNG> {
        let components = Linear { prev: self.components, weights, bias };
        NNBuilder { components, _out: PhantomData, ..self }
    }

    /// Adds a new [`Linear`] layer to the neural network.
    pub fn layer_from_arr<const N: usize>(
        self,
        weights: [[X; IN]; N],
        bias: [X; N],
    ) -> NNBuilder<X, NNIN, [(); N], Linear<X, IN, N, PREV>, RNG> {
        self.layer_from_parts(Matrix::new(weights), Vector::new(bias))
    }
}

impl<F, NNIN, const IN: usize, PREV, RNG> NNBuilder<F, NNIN, [(); IN], PREV, RNG>
where
    F: Float,
    NNIN: Shape,
    PREV: NNComponent<F, NNIN, [(); IN]>,
    RNG: rand::Rng,
    rand_distr::StandardNormal: rand_distr::Distribution<F>,
{
    /// Uses [`Initializer`] to add a new [`Linear`] layer to the neural network.
    pub fn layer<const N: usize>(
        mut self,
        weights_init: Initializer<F, [[(); IN]; N]>,
        bias_init: Initializer<F, [(); N]>,
    ) -> NNBuilder<F, NNIN, [(); N], Linear<F, IN, N, PREV>, RNG>
    where
        [[(); IN]; N]: Len<{ IN * N }>,
        [(); N]: Len<N>,
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
