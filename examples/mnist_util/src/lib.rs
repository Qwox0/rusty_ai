#![feature(iter_array_chunks)]

use std::ops::Range;

#[cfg(not(target_os = "windows"))]
const NL: &'static str = "\n";
#[cfg(target_os = "windows")]
const NL: &'static str = "\r\n";

pub use load_mnist::Mnist;

pub fn get_mnist() -> load_mnist::Mnist {
    let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/mnist-raw");

    load_mnist::MnistBuilder::new()
        .base_path(data_path)
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize()
}

/// each pixel has a size of 2x1 (`XX`).
///
/// # Panics
///
/// Panics if `IMAGE_SIZE != IMAGE_SIDE ^ 2`
pub fn image_to_string<const IMAGE_SIDE: usize, const IMAGE_SIZE: usize>(
    image: &[f64; IMAGE_SIZE],
    val_range: Range<f64>,
) -> String {
    assert_eq!(IMAGE_SIZE, IMAGE_SIDE * IMAGE_SIDE);

    let v_border = "─".repeat(IMAGE_SIDE * 2);
    let v_border = v_border.as_str();

    let border_bytes = (IMAGE_SIDE * 6 + 4) * 3;
    let newline_bytes = (IMAGE_SIDE + 1) * NL.len();
    let mut buf = String::with_capacity(IMAGE_SIDE * IMAGE_SIDE * 2 + border_bytes + newline_bytes);

    let range_width = val_range.end - val_range.start;
    buf.push('┌');
    buf.push_str(v_border);
    buf.push('┐');
    buf.push_str(NL);
    #[allow(illegal_floating_point_literal_pattern)]
    image
        .iter()
        .map(|x| (x - val_range.start) / range_width)
        .map(|x| match x {
            ..=0.0751 => ' ',
            ..=0.0829 => '`',
            ..=0.0848 => '.',
            ..=0.1227 => '-',
            ..=0.1403 => '\'',
            ..=0.1559 => ':',
            ..=0.185 => '_',
            ..=0.2183 => ',',
            ..=0.2417 => '^',
            ..=0.2571 => '=',
            ..=0.2852 => ';',
            ..=0.2902 => '>',
            ..=0.2919 => '<',
            ..=0.3099 => '+',
            ..=0.3192 => '!',
            ..=0.3232 => 'r',
            ..=0.3294 => 'c',
            ..=0.3384 => '*',
            ..=0.3609 => '/',
            ..=0.3619 => 'z',
            ..=0.3667 => '?',
            ..=0.3737 => 's',
            ..=0.3747 => 'L',
            ..=0.3838 => 'T',
            ..=0.3921 => 'v',
            ..=0.396 => ')',
            ..=0.3984 => 'J',
            ..=0.3993 => '7',
            ..=0.4075 => '(',
            ..=0.4091 => '|',
            ..=0.4101 => 'F',
            ..=0.42 => 'i',
            ..=0.423 => '{',
            ..=0.4247 => 'C',
            ..=0.4274 => '}',
            ..=0.4293 => 'f',
            ..=0.4328 => 'I',
            ..=0.4382 => '3',
            ..=0.4385 => '1',
            ..=0.442 => 't',
            ..=0.4473 => 'l',
            ..=0.4477 => 'u',
            ..=0.4503 => '[',
            ..=0.4562 => 'n',
            ..=0.458 => 'e',
            ..=0.461 => 'o',
            ..=0.4638 => 'Z',
            ..=0.4667 => '5',
            ..=0.4686 => 'Y',
            ..=0.4693 => 'x',
            ..=0.4703 => 'j',
            ..=0.4833 => 'y',
            ..=0.4881 => 'a',
            ..=0.4944 => ']',
            ..=0.4953 => '2',
            ..=0.4992 => 'E',
            ..=0.5509 => 'S',
            ..=0.5567 => 'w',
            ..=0.5569 => 'q',
            ..=0.5591 => 'k',
            ..=0.5602 => 'P',
            //..=0.5602 => '6',
            ..=0.565 => 'h',
            ..=0.5776 => '9',
            ..=0.5777 => 'd',
            ..=0.5818 => '4',
            ..=0.587 => 'V',
            ..=0.5972 => 'p',
            ..=0.5999 => 'O',
            ..=0.6043 => 'G',
            ..=0.6049 => 'b',
            ..=0.6093 => 'U',
            ..=0.6099 => 'A',
            ..=0.6465 => 'K',
            ..=0.6561 => 'X',
            ..=0.6595 => 'H',
            ..=0.6631 => 'm',
            ..=0.6714 => '8',
            ..=0.6759 => 'R',
            ..=0.6809 => 'D',
            ..=0.6816 => '#',
            ..=0.6925 => '$',
            ..=0.7039 => 'B',
            ..=0.7086 => 'g',
            ..=0.7235 => '0',
            ..=0.7302 => 'M',
            ..=0.7332 => 'N',
            ..=0.7602 => 'W',
            ..=0.7834 => 'Q',
            ..=0.8037 => '%',
            //..=0.9999 => '&',
            ..=0.95 => '&',
            _ => '@',
        })
        .array_chunks::<IMAGE_SIDE>()
        .for_each(|line| {
            buf.push('│');
            for char in line {
                buf.push(char);
                buf.push(char);
            }
            buf.push('│');
            buf.push_str(NL);
        });
    buf.push('└');
    buf.push_str(v_border);
    buf.push('┘');
    assert_eq!(buf.capacity(), buf.len()); // not necessary
    buf
}
