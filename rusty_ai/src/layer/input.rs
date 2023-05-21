#[derive(Debug, Clone, Copy)]
pub struct InputLayer<const IN: usize>;

impl<const IN: usize> std::fmt::Display for InputLayer<IN> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let plural_s = if IN == 1 { "" } else { "s" };
        write!(f, "{} Input{}", IN, plural_s)
    }
}
