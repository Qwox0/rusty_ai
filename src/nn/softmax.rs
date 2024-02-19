use super::{component_new, Data, NN};
use crate::optimizer::Optimizer;
use const_tensor::{Float, Multidimensional, MultidimensionalOwned, Shape, Tensor, Vector};
use core::fmt;
use serde::{Deserialize, Serialize};

/// The softmax activation function.
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
    type StoredData = Data<Vector<X, LEN>, PREV::StoredData>;

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
        for x in input.iter_elem_mut() {
            *x = x.exp();
            sum += *x;
        }
        let out = input.scalar_div(sum);
        (out.clone(), Data { data: out, prev: prev_data })
    }

    /// ```text
    /// y_i = softmax(x_i) = e^x_i/sum e^x over x = e^x_i/s where s = sum e^x over x
    ///
    /// dy_i/dx_i = (e^x_i * s - e^x_i * e^x_i)/s^2
    ///           = e^x_i/s * (s - e^x_i)/s
    ///           = softmax(x_i) * (1 - softmax(x_i))
    ///           = softmax(x_i) - softmax(x_i)^2
    /// dy_j/dx_i = (0 - e^x_i * e^x_j)/s^2
    /// dy_j/dx_i = - softmax(x_i) * softmax(x_j)
    /// dL/dx_i   = sum dL/dy * dy/dx_i over y
    ///           = dL/dy_i * dy_i/dx_i + sum dL/dy_j * dy_j/dx_i where j!=i
    ///           = dL/dy_i * softmax(x_i) - sum dL/dy_j * softmax(x_i) * softmax(x_j) over j
    ///           = dL/dy_i * y_i - sum dL/dy_j * y_i * y_j over j
    ///           = y_i * (dL/dy_i - sum dL/dy_j * y_j over j)
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
        let Data { prev: prev_data, data: self_out } = data;
        // sum = sum dL/dy * y over y
        let sum: X = out_grad.clone().mul_elem(&self_out).iter_elem().sum();
        let input_grad = out_grad.scalar_sub(sum).mul_elem(&self_out);
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

/// The log softmax activation function.
///
/// currently only works if followed by `NLLLoss`! TODO
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
    /// y_i = ln(softmax(x_i)) = ln(e^x_i/sum e^x over x) = x_i - ln(sum e^x over x)
    /// dy_i/dx_i = 1 - e^x_i/sum e^x over x = 1 - softmax(x_i)
    /// dy_j/dx_i =   - e^x_j/sum e^x over x =   - softmax(x_j)
    /// dL/dx_i   = sum dL/dy * dy/dx_i over y
    /// dL/dx_i   = dL/dy_i * (1 - softmax(x_i)) - sum dL/dy_j * softmax(x_j) where j!=i
    /// dL/dx_i   = dL/dy_i - sum dL/dy_j * softmax(x_j) over j
    /// dL/dx_i   = dL/dy_i - sum dL/dy_j * e^y_j over j
    ///
    /// NLLLoss:
    /// dL/dx_i = -1
    /// dL/dx_j = 0
    ///
    /// dL/dx_i   = dL/dy_i + e^y_i
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
        let input_grad = prop_out.map(X::exp).add_elem(&out_grad);
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
    use super::*;
    use crate::nn::NNHead;
    use const_tensor::vector;

    #[test]
    fn logsoftmax_prop() {
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
