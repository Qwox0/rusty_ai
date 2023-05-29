mod bias;
mod weights;

pub use bias::*;
pub use weights::*;

use crate::util::RngWrapper;

pub trait RandomMarker {
    fn set_seed(&mut self, seed: u64);
}

pub trait Buildable {
    type OUT;
    fn build(self, rng: &mut RngWrapper) -> Self::OUT;
    fn clone_build(&mut self, rng: &mut RngWrapper) -> Self::OUT;
}
