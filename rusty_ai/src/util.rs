/*
fn dot_product<'a, T, const N: usize>(vec1: &'a [T; N], vec2: &'a [T; N]) -> T
where
    T: Default + std::ops::Add<&'a T, Output = T> + 'a,
    &'a T: std::ops::Mul<Output = &'a T>,
{
    debug_assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1 * x2)
}
pub fn dot_product<T, const N: usize>(vec1: &[T; N], vec2: &[T; N]) -> T
where
    T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    debug_assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
}
*/

pub fn dot_product<T>(vec1: &Vec<T>, vec2: &Vec<T>) -> T
where
    T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    debug_assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
}

pub fn dot_product2<T, const N: usize>(vec1: &[T; N], vec2: &[T; N]) -> T
where
    T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    debug_assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
}

pub fn relu(x: f64) -> f64 {
    if x.is_sign_positive() {
        x
    } else {
        0.0
    }
}

pub trait SetLength {
    type Item;
    fn set_length(self, length: usize, default: Self::Item) -> Self;
}

impl<T: Clone> SetLength for Vec<T> {
    type Item = T;
    fn set_length(self, length: usize, default: Self::Item) -> Self {
        let missing_len = length - self.len();
        let missing_elements = (0..missing_len).into_iter().map(|_| default.clone());
        self.into_iter()
            .take(length)
            .chain(missing_elements)
            .collect()
    }
}

/*
pub trait SetLengthDefault {
    fn set_length_default(self, length: usize) -> Self;
}

impl<T, U> SetLengthDefault for T
where
    T: SetLength<Item = U>,
    U: Default + Clone,
{
    fn set_length_default(self, length: usize) -> Self {
        self.set_length(length, U::default())
    }
}
*/

pub mod macros {
    macro_rules! __get_getter {
        ($name:ident -> $attr: ident: $attr_type: ty) => {
            #[inline(always)]
            pub fn $name(&self) -> &$attr_type {
                &self.$attr
            }
        };
    }
    pub(crate) use __get_getter;

    macro_rules! impl_getter {
        ( $type:ident: $( $getter: ident -> $attr: ident: $attr_type: ty ),+ ) => {
            impl $type {
                $( $crate::util::macros::__get_getter! { $getter -> $attr: $attr_type } )+
            }
        };
        ( $type:ident < $( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+ >: $( $getter: ident -> $attr: ident: $attr_type: ty ),+ ) => {
            impl<$( $lt $( : $clt $(+ $dlt )* )? ),+> $type<$( $lt ),+> {
                $( $crate::util::macros::__get_getter! { $getter -> $attr: $attr_type } )+
            }
        };
    }
    pub(crate) use impl_getter;
}
