pub fn dot_product<T>(vec1: &Vec<T>, vec2: &Vec<T>) -> T
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
    fn set_length(self, new_length: usize, default: Self::Item) -> Self;
    fn to_arr<const N: usize>(self, default: Self::Item) -> [Self::Item; N];
}

impl<T: Clone> SetLength for Vec<T> {
    type Item = T;
    fn set_length(mut self, new_length: usize, default: Self::Item) -> Self {
        self.resize(new_length, default);
        self
    }

    fn to_arr<const N: usize>(self, default: Self::Item) -> [Self::Item; N] {
        let mut arr = std::array::from_fn::<_, N, _>(|_| default.clone());
        for (idx, elem) in self.into_iter().enumerate() {
            arr[idx] = elem;
        }
        arr
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
    /// impl_getter! { Matrix<T>: get_elements -> elements: Vec<Vec<T>>, }
    macro_rules! impl_getter {
        ( $type:ident: $( $getter:ident -> $attr:ident: $attr_type:ty ),+ $(,)? ) => {
            impl $type {
                $( $crate::util::macros::impl_getter! { inner $getter -> $attr: $attr_type } )+
            }
        };
        ( $type:ident < $( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+ >: $( $getter: ident -> $attr: ident: $attr_type: ty ),+ $(,)? ) => {
            impl<$( $lt $( : $clt $(+ $dlt )* )? ),+> $type<$( $lt ),+> {
                $( $crate::util::macros::impl_getter! { inner $getter -> $attr: $attr_type } )+
            }
        };
        ( inner $name:ident -> $attr:ident : $attr_type:ty ) => {
            #[inline(always)]
            pub fn $name(&self) -> &$attr_type {
                &self.$attr
            }
        };
    }
    pub(crate) use impl_getter;

    /// impl_fn_traits!(Fn<(f64,)> -> f64: ActivationFunction => call);
    macro_rules! impl_fn_traits {
        ( FnOnce < $in:ty > -> $out:ty : $type:ty => $method:ident ) => {
            $crate::util::macros::impl_fn_traits! { inner FnOnce<$in> -> $out; $type; $method; call_once; }
        };
        ( FnMut < $in:ty > -> $out:ty : $type:ty => $method:ident ) => {
            $crate::util::macros::impl_fn_traits!(FnOnce<$in> -> $out: $type => $method);
            $crate::util::macros::impl_fn_traits! { inner FnMut<$in>; $type; $method; call_mut; &mut}
        };
        ( Fn < $in:ty > -> $out:ty : $type:ty => $method:ident ) => {
            $crate::util::macros::impl_fn_traits!(FnMut<$in> -> $out: $type => $method);
            $crate::util::macros::impl_fn_traits!(inner Fn<$in>; $type; $method; call; &);
        };
        ( inner $fn_trait:ident < $in:ty > $( -> $out:ty )? ; $type:ty ; $method:ident ; $call:ident ; $( $self:tt )* ) => {
            impl $fn_trait<$in> for $type {
                $( type Output = $out; )?
                extern "rust-call" fn $call( $($self)* self, args: $in) -> Self::Output {
                    <$type>::$method(&self, args)
                }
            }
        };
    }
    pub(crate) use impl_fn_traits;

    macro_rules! impl_new {
        ( $vis:vis $type:ident : $( $attr:ident : $attrty: ty ),* $(,)? ) => {
            impl $type {
                $vis fn new( $( $attr : $attrty ),* ) -> $type {
                    $type { $( $attr ),* }
                }
            }
        };
        ( $vis:vis $type:ident : $( $attr:ident : $attrty: ty ),* ; Default ) => {
            impl $type {
                $vis fn new( $( $attr : $attrty ),* ) -> $type {
                    $type {
                        $( $attr ),* ,
                        ..Default::default()
                    }
                }
            }
        };
    }
    pub(crate) use impl_new;
}
