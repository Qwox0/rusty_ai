use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    #[arg(long, short = 'c', default_value_t = 10)]
    pub training_count: usize,
}

