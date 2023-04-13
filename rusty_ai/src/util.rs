pub fn dot_product<T>(vec1: &Vec<T>, vec2: &Vec<T>) -> T
where
    T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
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
/// can be used for scalar multiplication ( vec * Scalar(t) / Scalar(t) * vec )
#[derive(Debug, Clone, Copy)]
pub struct Scalar(pub f64);

macro_rules! impl_scalar_mul {
    ( Scalar * $rhs:ty = $out:ty ) => {
        impl Mul<Scalar> for $rhs {
            type Output = $out;
            fn mul(self, rhs: Scalar) -> Self::Output {
                self.into_iter().map(|x| x * rhs.0).collect()
            }
        }
        impl Mul<$rhs> for Scalar {
            type Output = $out;
            fn mul(self, rhs: $rhs) -> Self::Output {
                rhs * self
            }
        }
    };
}

impl_scalar_mul! { Scalar * Vec<f64> = Vec<f64> }
impl_scalar_mul! { Scalar * &Vec<f64> = Vec<f64> }
impl_scalar_mul! { Scalar * Vec<&f64> = Vec<f64> }

impl MulAssign<Scalar> for Vec<f64> {
    fn mul_assign(&mut self, rhs: Scalar) {
        *self = self.iter().map(|x| x * rhs.0).collect();
    }
}

impl Mul<Scalar> for Matrix<f64> {
    type Output = Matrix<f64>;
    fn mul(mut self, rhs: Scalar) -> Self::Output {
        for row in self.iter_rows_mut() {
            *row *= rhs;
        }
        self
    }
}

impl Mul<Matrix<f64>> for Scalar {
    type Output = Matrix<f64>;
    fn mul(self, rhs: Matrix<f64>) -> Self::Output {
        rhs * self
    }
}

// ------------------------

impl<T: Mul<Scalar, Output = T>> Mul<T> for Scalar {
    type Output = T;
    fn mul(self, rhs: T) -> Self::Output {
        rhs * self
    }
}

impl<'a, T: 'a> Mul<&T> for Scalar
where
    &'a T: Mul<Scalar, Output = T>,
{
    type Output = T;
    fn mul(self, rhs: &T) -> Self::Output {
        rhs * self
    }
}
*/

//pub struct ArithWrapper<T>(pub T);

pub trait EntryAdd<Rhs = Self>: Sized {
    /// performs addition entrywise by mutating `self` in place.
    /// "`self` = `self` + `rhs`"
    fn mut_add_entries(&mut self, rhs: Rhs) -> &mut Self;
    /// performs addition entrywise and returns result.
    /// "return `self` + `rhs`"
    fn add_entries(mut self, rhs: Rhs) -> Self {
        self.mut_add_entries(rhs);
        self
    }
}

pub trait EntrySub<Rhs = Self>: Sized {
    /// performs addition entrywise by mutating `self` in place.
    /// "`self` = `self` - `rhs`"
    fn mut_sub_entries(&mut self, rhs: Rhs) -> &mut Self;
    /// performs subtraction entrywise and returns result.
    /// "return `self` - `rhs`"
    fn sub_entries(mut self, rhs: Rhs) -> Self {
        self.mut_sub_entries(rhs);
        self
    }
}

pub trait EntryMul<Rhs = Self>: Sized {
    /// performs addition entrywise by mutating `self` in place.
    /// "`self` = `self` + `rhs`"
    fn mut_mul_entries(&mut self, rhs: Rhs) -> &mut Self;
    /// performs addition entrywise and returns result.
    /// "return `self` + `rhs`"
    fn mul_entries(mut self, rhs: Rhs) -> Self {
        self.mut_mul_entries(rhs);
        self
    }
}

macro_rules! impl_entry_arithmetic_trait {
    ( $trait:ident : $trait_fn:ident $op:tt ) => {
        impl $trait<&f64> for f64 {
            fn $trait_fn(&mut self, rhs: &f64) -> &mut Self {
                *self $op rhs;
                self
            }
        }

        impl<'a, T: $trait<&'a T> + 'a> $trait<&'a Vec<T>> for Vec<T> {
            fn $trait_fn(&mut self, rhs: &'a Vec<T>) -> &mut Self {
                assert_eq!(self.len(), rhs.len());
                for (x, rhs) in self.iter_mut().zip(rhs) {
                    x.$trait_fn(rhs);
                }
                self
            }
        }

        impl<T: for<'a> $trait<&'a T>> $trait<T> for T {
            fn $trait_fn(&mut self, rhs: T) -> &mut Self {
                self.$trait_fn(&rhs)
            }
        }
    };
}

impl_entry_arithmetic_trait! { EntryAdd : mut_add_entries += }
impl_entry_arithmetic_trait! { EntrySub : mut_sub_entries -= }
impl_entry_arithmetic_trait! { EntryMul : mut_mul_entries *= }

pub trait ScalarMul: Sized {
    /// performs scalar multiplication by mutating `self` in place.
    /// "`self` = `self` * `scalar`"
    fn mut_mul_scalar(&mut self, scalar: f64) -> &mut Self;
    /// performs scalar multiplication and returns result.
    /// "return `self` * `scalar`"
    fn mul_scalar(mut self, scalar: f64) -> Self {
        self.mut_mul_scalar(scalar);
        self
    }
}

impl ScalarMul for f64 {
    fn mut_mul_scalar(&mut self, scalar: f64) -> &mut Self {
        *self *= scalar;
        self
    }
}

impl<T: ScalarMul> ScalarMul for Vec<T> {
    fn mut_mul_scalar(&mut self, scalar: f64) -> &mut Self {
        for x in self.iter_mut() {
            x.mut_mul_scalar(scalar);
        }
        self
    }
}

/*
impl EntrySub for f64 {
    fn sub_into(&mut self, rhs: &Self) {
        *self -= rhs;
    }
}

impl<T: EntrySub> EntrySub for Vec<T> {
    fn sub_into(&mut self, rhs: &Self) {
        assert_eq!(self.len(), rhs.len());
        for (x, rhs) in self.iter_mut().zip(rhs) {
            x.sub_into(rhs)
        }
    }
}
*/

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
    /// ```rust
    /// impl Type {
    ///     impl_getter! { get_num -> num: usize }
    ///     impl_getter! { get_num_mut -> elements: &mut usize }
    /// }
    /// ```
    macro_rules! impl_getter {
        ( $name:ident -> $attr:ident : &mut $( $attr_type:tt )+ ) => {
            #[allow(unused)]
            #[inline(always)]
            pub fn $name(&mut self) -> &mut $($attr_type)+ {
                &mut self.$attr
            }
        };
        ( $name:ident -> $attr:ident : & $( $attr_type:tt )+ ) => {
            #[allow(unused)]
            #[inline(always)]
            pub fn $name(&self) -> & $($attr_type)+ {
                &self.$attr
            }
        };
        ( $name:ident -> $attr:ident : $( $attr_type:tt )+ ) => {
            #[allow(unused)]
            #[inline(always)]
            pub fn $name(&self) -> $($attr_type)+ {
                self.$attr
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
            $crate::util::macros::impl_fn_traits! { FnOnce<$in> -> $out: $type => $method }
            $crate::util::macros::impl_fn_traits! { inner FnMut<$in>; $type; $method; call_mut; &mut}
        };
        ( Fn < $in:ty > -> $out:ty : $type:ty => $method:ident ) => {
            $crate::util::macros::impl_fn_traits! { FnMut<$in> -> $out: $type => $method }
            $crate::util::macros::impl_fn_traits! { inner Fn<$in>; $type; $method; call; & }
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

    /// ```rust
    /// impl Point {
    ///     impl_new! { pub x: usize, y: usize; Default }
    /// }
    /// ```
    macro_rules! impl_new {
        ( $vis:vis $( $attr:ident : $attrty: ty ),* $(,)? ) => {
            $vis fn new( $( $attr : $attrty ),* ) -> Self {
                Self { $( $attr ),* }
            }
        };
        ( $vis:vis $( $attr:ident : $attrty: ty ),+ ; Default ) => {
            $vis fn new( $( $attr : $attrty ),* ) -> Self {
                Self {
                    $( $attr ),* ,
                    ..Default::default()
                }
            }
        };
    }
    pub(crate) use impl_new;
}
