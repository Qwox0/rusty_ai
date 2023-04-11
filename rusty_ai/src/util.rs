
pub fn dot_product<T>(vec1: &Vec<T>, vec2: &Vec<T>) -> T
where
    T: Default + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    assert_eq!(vec1.len(), vec2.len());
    vec1.iter()
        .zip(vec2.iter())
        .fold(T::default(), |acc, (x1, x2)| acc + x1.clone() * x2.clone())
}

/// Mean squarred error: E = 0.5 * ∑ (o_i - t_i)^2 from i = 1 to n
pub fn mean_squarred_error<const N: usize>(output: &[f64; N], expected_output: &[f64; N]) -> f64 {
    0.5 * output
        .iter()
        .zip(expected_output)
        .map(|(out, expected)| out - expected)
        .map(|x| x * x)
        .sum::<f64>()
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
    fn add_into(&mut self, rhs: Rhs);
    /// performs addition entrywise and returns result.
    /// "return `self` + `rhs`"
    fn add(mut self, rhs: Rhs) -> Self {
        self.add_into(rhs);
        self
    }
}

impl EntryAdd<&f64> for f64 {
    fn add_into(&mut self, rhs: &f64) {
        *self += rhs;
    }
}

impl<'a, T: EntryAdd<&'a T> + 'a> EntryAdd<&'a Vec<T>> for Vec<T> {
    fn add_into(&mut self, rhs: &'a Vec<T>) {
        assert_eq!(self.len(), rhs.len());
        for (x, rhs) in self.iter_mut().zip(rhs) {
            x.add_into(rhs)
        }
    }
}

impl<T: for<'a> EntryAdd<&'a T>> EntryAdd<T> for T {
    fn add_into(&mut self, rhs: T) {
        self.add_into(&rhs)
    }
}

pub trait ScalarMul: Sized {
    /// performs scalar multiplication by mutating `self` in place.
    /// "`self` = `self` * `scalar`"
    fn mul_scalar_into(&mut self, scalar: f64);
    /// performs scalar multiplication and returns result.
    /// "return `self` * `scalar`"
    fn mul_scalar(mut self, scalar: f64) -> Self {
        self.mul_scalar_into(scalar);
        self
    }
}

impl ScalarMul for f64 {
    fn mul_scalar_into(&mut self, scalar: f64) {
        *self *= scalar;
    }
}

impl<T: ScalarMul> ScalarMul for Vec<T> {
    fn mul_scalar_into(&mut self, scalar: f64) {
        self.iter_mut().for_each(|x| x.mul_scalar_into(scalar));
    }
}

pub trait EntrySub<Rhs = Self>: Sized {
    /// performs addition entrywise by mutating `self` in place.
    /// "`self` = `self` - `rhs`"
    fn sub_into(&mut self, rhs: Rhs);
    /// performs subtraction entrywise and returns result.
    /// "return `self` - `rhs`"
    fn sub(mut self, rhs: Rhs) -> Self {
        self.sub_into(rhs);
        self
    }
}

impl<'a, T> EntrySub<&'a T> for T
where
    T: EntryAdd<&'a T> + ScalarMul + 'a,
{
    fn sub_into(&mut self, rhs: &'a T) {
        self.mul_scalar_into(-1.0);
        self.add_into(rhs);
        self.mul_scalar_into(-1.0);
    }
}

impl<T: for<'a> EntrySub<&'a T>> EntrySub<T> for T {
    fn sub_into(&mut self, rhs: T) {
        self.sub_into(&rhs);
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

    /// ```rust
    /// impl Point {
    ///     impl_new! { pub x: usize, y: usize }
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
