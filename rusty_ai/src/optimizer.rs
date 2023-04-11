use crate::{layer::Layer, results::GradientLayer, util::{EntryAdd, ScalarMul}};

pub const LEARNING_RATE: f64 = 0.5;

pub(crate) fn optimize_weights<'a>(
    layers: impl IntoIterator<Item = &'a mut Layer>,
    gradient: Vec<GradientLayer>,
) {
    for (layer, gradient) in layers.into_iter().zip(gradient) {
        //println!("{:?}", gradient);
        //println!("before: {} ; {:?}", layer.get_bias(), layer.get_weights());

        *layer.get_bias_mut() -= LEARNING_RATE * gradient.bias_change;
        layer
            .get_weights_mut()
            .add_into(gradient.weights_change.mul_scalar(-LEARNING_RATE));
        //println!("after: {} ; {:?}", layer.get_bias(), layer.get_weights());
    }
}
