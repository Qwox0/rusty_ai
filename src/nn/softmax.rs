use super::{component_new, Data, NN};
use crate::optimizer::Optimizer;
use const_tensor::{
    Float, Len, Multidimensional, MultidimensionalOwned, Num, Shape, Tensor, Vector,
};
use core::fmt;
use serde::{Deserialize, Serialize};

/// TODO: higher dimensions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Softmax<PREV> {
    pub(super) prev: PREV,
}

component_new! { Softmax }

impl<X, const LEN: usize, NNIN, PREV> NN<X, NNIN, [(); LEN]> for Softmax<PREV>
where
    X: Float,
    NNIN: Shape,
    PREV: NN<X, NNIN, [(); LEN]>,
{
    type Grad = PREV::Grad;
    type In = [(); LEN];
    type OptState<O: Optimizer<X>> = PREV::OptState<O>;
    /// Bool Tensor contains whether propagation output elements where unequal to zero or not.
    type StoredData = Data<Vector<bool, LEN>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, [(); LEN]> {
        let mut input = self.prev.prop(input);
        let mut sum = X::ZERO;
        for x in input.iter_elem_mut() {
            *x = x.exp();
            sum += *x;
        }
        input.scalar_div(sum)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Vector<X, LEN>, Self::StoredData) {
        let (mut input, prev_data) = self.prev.train_prop(input);
        let mut sum = X::ZERO;
        let mut data = Vector::default();
        for x in input.iter_elem_mut() {
            *x = x.exp();
            sum += *x;
        }
        let out = input.map_inplace(|exp| exp / sum);
        (out, Data { data, prev: prev_data })
    }

    /// ```text
    /// y_i = softmax(x_i) = e^x_i/sum(e^x_j)
    /// dy_i/dx_i = relu'(a_i)
    /// dL/dx_i   = dL/dy_i * dy_i/dx_i
    ///
    /// -----
    ///
    /// y_i: output component i
    /// x_i: input component i
    ///
    /// L: total loss
    /// ```
    #[inline]
    fn backprop_inplace(
        &self,
        out_grad: Tensor<X, [(); LEN]>,
        data: Self::StoredData,
        grad: &mut Self::Grad,
    ) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= X::from_bool(is_pos);
        }
        self.prev.backprop_inplace(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: &mut Self::OptState<O>,
    ) {
        self.prev.optimize(grad, optimizer, opt_state)
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        self.prev.init_zero_grad()
    }

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {
        self.prev.init_opt_state()
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        self.prev.iter_param()
    }
}

impl<PREV: fmt::Display> fmt::Display for Softmax<PREV> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", &self.prev)?;
        write!(f, "Softmax")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LogSoftmax<PREV> {
    pub(super) prev: PREV,
}

component_new! { LogSoftmax }

impl<X, const LEN: usize, NNIN, PREV> NN<X, NNIN, [(); LEN]> for LogSoftmax<PREV>
where
    X: Float,
    NNIN: Shape,
    PREV: NN<X, NNIN, [(); LEN]>,
{
    type Grad = PREV::Grad;
    type In = [(); LEN];
    type OptState<O: Optimizer<X>> = PREV::OptState<O>;
    type StoredData = Data<Vector<X, LEN>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<X, NNIN>) -> Tensor<X, [(); LEN]> {
        // ln(e^(x_i)/exp_sum) == x_i - ln(exp_sum)
        let input = self.prev.prop(input);
        let ln_sum = input.iter_elem().copied().map(X::exp).sum::<X>().ln();
        input.scalar_sub(ln_sum)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<X, NNIN>) -> (Vector<X, LEN>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let ln_sum = input.iter_elem().copied().map(X::exp).sum::<X>().ln();
        let out = input.scalar_sub(ln_sum);
        (out.clone(), Data { data: out, prev: prev_data })
    }

    /// ```text
    /// y_i = ln(softmax(x_i)) = ln(e^x_i/sum(e^x_j)) = x_i - ln(sum(e^x_j))
    /// dy_i/dx_i = 1 - e^x_i/sum(e^x_i) = 1 - softmax(x_i)
    /// dy_i/dx_j =   - e^x_i/sum(e^x_i) =   - softmax(x_i)
    /// dL/dx_i   = dL/dy_i * dy_i/dx_i
    ///
    /// -----
    ///
    /// y_i: output component i
    /// x_i: input component i
    ///
    /// L: total loss
    /// ```
    #[inline]
    fn backprop_inplace(
        &self,
        out_grad: Tensor<X, [(); LEN]>,
        data: Self::StoredData,
        grad: &mut Self::Grad,
    ) {
        let Data { prev: prev_data, data: prop_out } = data;
        let input_grad = prop_out.map_inplace(X::exp).add_elem(&out_grad);
        self.prev.backprop_inplace(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<X>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        opt_state: &mut Self::OptState<O>,
    ) {
        self.prev.optimize(grad, optimizer, opt_state)
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        self.prev.init_zero_grad()
    }

    #[inline]
    fn init_opt_state<O: Optimizer<X>>(&self) -> Self::OptState<O> {
        self.prev.init_opt_state()
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &X> {
        self.prev.iter_param()
    }
}

impl<PREV: fmt::Display> fmt::Display for LogSoftmax<PREV> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", &self.prev)?;
        write!(f, "LogSoftmax")
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::NNHead;

    use super::*;
    use const_tensor::vector;

    #[test]
    fn logsoftmax_prop()
    where [(); 3]: Len<3> {
        let log_softmax = LogSoftmax::new(NNHead);

        let input = Vector::new([2.0, 0.0, 0.35]);

        let out = log_softmax.prop(input);

        println!("out: {:?}", out);
        assert_eq!(
            out.as_ref(),
            vector::literal([-0.2832109859173988, -2.283210985917399, -1.9332109859173987])
        );
    }
}
