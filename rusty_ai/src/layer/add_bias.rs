use super::LayerBias;
use crate::util::EntryAdd;

pub trait AddBias {
    fn add_bias_mut(&mut self, bias: &LayerBias) -> &mut Self;
}

impl AddBias for Vec<f64> {
    fn add_bias_mut(&mut self, bias: &LayerBias) -> &mut Self {
        match bias {
            LayerBias::OnePerLayer(bias) => self.iter_mut().for_each(|x| *x += bias),
            LayerBias::OnePerNeuron(bias) => {
                self.add_entries_mut(bias);
            }
        }
        self
    }
}
