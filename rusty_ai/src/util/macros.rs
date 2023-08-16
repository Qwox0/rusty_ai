/// ```rust, ignore
/// struct Type { num: usize }
/// impl Type {
///     impl_getter! { get_num -> num: usize }
///     impl_getter! { get_num_mut -> num: &mut usize }
/// }
/// ```
macro_rules! impl_getter {
    ( $vis:vis $name:ident -> $attr:ident : &mut $( $attr_type:tt )+ ) => {
        #[allow(unused)]
        #[inline(always)]
        $vis fn $name(&mut self) -> &mut $($attr_type)+ {
            &mut self.$attr
        }
    };
    ( $vis:vis $name:ident -> $attr:ident : & $( $attr_type:tt )+ ) => {
        #[allow(unused)]
        #[inline(always)]
        $vis fn $name(&self) -> & $($attr_type)+ {
            &self.$attr
        }
    };
    ( $vis:vis $name:ident -> $attr:ident : $( $attr_type:tt )+ ) => {
        #[allow(unused)]
        #[inline(always)]
        $vis fn $name(&self) -> $($attr_type)+ {
            self.$attr
        }
    };
}
pub(crate) use impl_getter;

/*
/// impl_fn_traits!(Fn<(f64,)> -> f64: ActivationFunction => call);
macro_rules! impl_fn_traits {
    ( FnOnce < $in:ty > -> $out:ty : $type:ty => $method:ident ) => {
        $crate::util::impl_fn_traits! { inner FnOnce<$in> -> $out; $type; $method; call_once; }
    };
    ( FnMut < $in:ty > -> $out:ty : $type:ty => $method:ident ) => {
        $crate::util::impl_fn_traits! { FnOnce<$in> -> $out: $type => $method }
        $crate::util::impl_fn_traits! { inner FnMut<$in>; $type; $method; call_mut; &mut}
    };
    ( Fn < $in:ty > -> $out:ty : $type:ty => $method:ident ) => {
        $crate::util::impl_fn_traits! { FnMut<$in> -> $out: $type => $method }
        $crate::util::impl_fn_traits! { inner Fn<$in>; $type; $method; call; & }
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
/// impl_fn_traits! { ActivationFunction : Fn<(f64,)> -> f64 ; call }
macro_rules! impl_fn_traits {
    ($type:ty : $method:ident => FnOnce < $in:ty > -> $out:ty) => {
        impl FnOnce<$in> for $type {
            type Output = $out;

            extern "rust-call" fn call_once(self, args: $in) -> Self::Output {
                Self::$method(&self, args)
            }
        }
    };
    ($type:ty : $method:ident => FnMut < $in:ty > -> $out:ty) => {
        $crate::util::impl_fn_traits! { $type : $method => FnOnce<$in> -> $out }
        impl FnMut<$in> for $type {
            extern "rust-call" fn call_mut(&mut self, args: $in) -> Self::Output {
                Self::$method(&self, args)
            }
        }
    };
    ($type:ty : $method:ident => Fn < $in:ty > -> $out:ty) => {
        $crate::util::impl_fn_traits! { $type : $method => FnMut<$in> -> $out }
        impl Fn<$in> for $type {
            extern "rust-call" fn call(&self, args: $in) -> Self::Output {
                Self::$method(&self, args)
            }
        }
    };
}
pub(crate) use impl_fn_traits;

macro_rules! impl_fn {
    ( $type:ty : Fn ( $( $in:ty ),* ) -> $out:ty ; $method:expr ) => {
        impl FnOnce<( $( $in , )* )> for $type {
            type Output = $out;
            extern "rust-call" fn call_once(self, args: $in) -> Self::Output {
                $method(&self, args)
            }
        }
        impl FnMut<$in> for $type {
            extern "rust-call" fn call_mut(&mut self, args: $in) -> Self::Output {
                $method(&self, args)
            }
        }
        impl Fn<$in> for $type {
            extern "rust-call" fn call(&self, args: $in) -> Self::Output {
                $method(&self, args)
            }
        }
    };
}
pub(crate) use impl_fn;
*/

/// ```rust, ignore
/// struct Point { x: usize, y: usize, other: i32 }
/// impl Point {
///     constructor! { pub new -> x: usize, y: usize; Default }
/// }
/// ```
macro_rules! constructor {
        ( $vis:vis $name:ident -> $( $attr:ident : $attrty: ty ),* $(,)? ) => {
            $vis fn $name( $( $attr : $attrty ),* ) -> Self {
                Self { $( $attr ),* }
            }
        };
        ( $vis:vis $name:ident -> $( $attr:ident : $attrty: ty ),+ ; Default ) => {
            $vis fn $name( $( $attr : $attrty ),* ) -> Self {
                Self {
                    $( $attr ),* ,
                    ..Default::default()
                }
            }
        };
    }
pub(crate) use constructor;
