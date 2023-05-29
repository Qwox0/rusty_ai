use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug, Clone)]
pub enum RngWrapper {
    Normal(rand::rngs::ThreadRng),
    Seeded(rand::rngs::StdRng),
}

impl Default for RngWrapper {
    fn default() -> Self {
        RngWrapper::Normal(rand::thread_rng())
    }
}

impl RngWrapper {
    pub fn new(seed: Option<u64>) -> Self {
        match seed {
            Some(seed) => RngWrapper::Seeded(StdRng::seed_from_u64(seed)),
            None => RngWrapper::Normal(rand::thread_rng()),
        }
    }
}

impl rand::RngCore for RngWrapper {
    fn next_u32(&mut self) -> u32 {
        match self {
            RngWrapper::Normal(rng) => rng.next_u32(),
            RngWrapper::Seeded(rng) => rng.next_u32(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        match self {
            RngWrapper::Normal(rng) => rng.next_u64(),
            RngWrapper::Seeded(rng) => rng.next_u64(),
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            RngWrapper::Normal(rng) => rng.fill_bytes(dest),
            RngWrapper::Seeded(rng) => rng.fill_bytes(dest),
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            RngWrapper::Normal(rng) => rng.try_fill_bytes(dest),
            RngWrapper::Seeded(rng) => rng.try_fill_bytes(dest),
        }
    }
}
