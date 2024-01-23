use super::component::{component_new, Data, NNComponent};
use crate::optimizer::Optimizer;
use const_tensor::{Float, Len, Num, Shape, Tensor, Vector};
use serde::{Deserialize, Serialize};

/// TODO: higher dimensions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Softmax<PREV> {
    pub(super) prev: PREV,
}

component_new! { Softmax }

impl<F, const LEN: usize, NNIN, PREV> NNComponent<F, NNIN, [(); LEN]> for Softmax<PREV>
where
    F: Float,
    [(); LEN]: Len<LEN>,
    NNIN: Shape,
    PREV: NNComponent<F, NNIN, [(); LEN]>,
{
    type Grad = PREV::Grad;
    type OptState<O: Optimizer<F>> = PREV::OptState<O>;
    /// Bool Tensor contains whether propagation output elements where unequal to zero or not.
    type StoredData = Data<Vector<bool, LEN>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<F, NNIN>) -> Tensor<F, [(); LEN]> {
        let mut input = self.prev.prop(input);
        let mut sum = F::ZERO;
        for x in input.iter_elem_mut() {
            *x = x.exp();
            sum += *x;
        }
        input.scalar_div(sum)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<F, NNIN>) -> (Vector<F, LEN>, Self::StoredData) {
        let (mut input, prev_data) = self.prev.train_prop(input);
        let mut sum = F::ZERO;
        let mut data = Vector::default();
        for x in input.iter_elem_mut() {
            *x = x.exp();
            sum += *x;
        }
        let out = input.map_inplace(|exp| exp / sum);
        (out, Data { data, prev: prev_data })
    }

    /// ```
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
    fn backprop(
        &self,
        out_grad: Tensor<F, [(); LEN]>,
        data: Self::StoredData,
        grad: &mut Self::Grad,
    ) {
        let Data { prev: prev_data, data } = data;
        let mut input_grad = out_grad;
        for (out, &is_pos) in input_grad.iter_elem_mut().zip(data.iter_elem()) {
            *out *= F::from_bool(is_pos);
        }
        self.prev.backprop(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<F>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        mut opt_state: Self::OptState<O>,
    ) -> Self::OptState<O> {
        self.prev.optimize(grad, optimizer, opt_state)
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        self.prev.init_zero_grad()
    }

    #[inline]
    fn init_opt_state<O: Optimizer<F>>(&self) -> Self::OptState<O> {
        self.prev.init_opt_state()
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &F> {
        self.prev.iter_param()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LogSoftmax<PREV> {
    pub(super) prev: PREV,
}

component_new! { LogSoftmax }

impl<F, const LEN: usize, NNIN, PREV> NNComponent<F, NNIN, [(); LEN]> for LogSoftmax<PREV>
where
    F: Float,
    [(); LEN]: Len<LEN>,
    NNIN: Shape,
    PREV: NNComponent<F, NNIN, [(); LEN]>,
{
    type Grad = PREV::Grad;
    type OptState<O: Optimizer<F>> = PREV::OptState<O>;
    type StoredData = Data<Vector<F, LEN>, PREV::StoredData>;

    #[inline]
    fn prop(&self, input: Tensor<F, NNIN>) -> Tensor<F, [(); LEN]> {
        // ln(e^(x_i)/exp_sum) == x_i - ln(exp_sum)
        let input = self.prev.prop(input);
        let ln_sum = input.iter_elem().copied().map(F::exp).sum::<F>().ln();
        input.scalar_sub(ln_sum)
    }

    #[inline]
    fn train_prop(&self, input: Tensor<F, NNIN>) -> (Vector<F, LEN>, Self::StoredData) {
        let (input, prev_data) = self.prev.train_prop(input);
        let ln_sum = input.iter_elem().copied().map(F::exp).sum::<F>().ln();
        let out = input.scalar_sub(ln_sum);
        (out.clone(), Data { data: out, prev: prev_data })
    }

    /// ```
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
    fn backprop(
        &self,
        out_grad: Tensor<F, [(); LEN]>,
        data: Self::StoredData,
        grad: &mut Self::Grad,
    ) {
        let Data { prev: prev_data, data: prop_out } = data;
        let input_grad = prop_out.map_inplace(F::exp).add_elem(&out_grad);
        self.prev.backprop(input_grad, prev_data, grad)
    }

    #[inline]
    fn optimize<O: Optimizer<F>>(
        &mut self,
        grad: &Self::Grad,
        optimizer: &O,
        mut opt_state: Self::OptState<O>,
    ) -> Self::OptState<O> {
        self.prev.optimize(grad, optimizer, opt_state)
    }

    #[inline]
    fn init_zero_grad(&self) -> Self::Grad {
        self.prev.init_zero_grad()
    }

    #[inline]
    fn init_opt_state<O: Optimizer<F>>(&self) -> Self::OptState<O> {
        self.prev.init_opt_state()
    }

    #[inline]
    fn iter_param(&self) -> impl Iterator<Item = &F> {
        self.prev.iter_param()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use const_tensor::vector;

    #[test]
    fn logsoftmax_prop()
    where [(); 3]: Len<3> {
        let log_softmax = LogSoftmax::new(());

        let input = Vector::new([2.0, 0.0, 0.35]);

        let out = log_softmax.prop(input);

        println!("out: {:?}", out);
        assert_eq!(
            out.as_ref(),
            vector::literal([-0.2832109859173988, -2.283210985917399, -1.9332109859173987])
        );
    }
}
