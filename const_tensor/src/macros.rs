macro_rules! count {
    () => { 0usize };
    ( $x:tt $($xs:tt)* ) => { 1usize + count!($($xs)*) };
}
pub(crate) use count;

pub trait ArrDefault {
    fn arr_default() -> Self;
}

impl<T: ArrDefault + Copy, const N: usize> ArrDefault for [T; N] {
    #[inline]
    fn arr_default() -> Self {
        [T::arr_default(); N]
    }
}

impl<T: Element> ArrDefault for T {
    #[inline]
    fn arr_default() -> Self {
        T::default()
    }
}

macro_rules! make_tensor {
    (
        $name:ident $data_name:ident : $($dim_name:ident)* => $shape:ty,
        Sub: $sub_data:ty
        $(,)?
    ) => {
        /// implements [`TensorData`]
        #[derive(Debug, Clone, PartialEq, Eq)]
        #[allow(non_camel_case_types)]
        #[repr(transparent)]
        pub struct $data_name<X: Element, $( const $dim_name: usize ),*> {
            data: $shape
        }

        unsafe impl<X: Element, $( const $dim_name: usize ),*> Len<{ $($dim_name * )* 1 }> for $data_name<X, $( $dim_name ),*> {}

        unsafe impl<X: Element, $( const $dim_name: usize ),*> TensorData<X> for $data_name<X, $( $dim_name ),*> {
            type Owned = $name<X, $( $dim_name ),*>;
            type Shape = $shape;
            type SubData = $sub_data;

            type Mapped<X_: Element> = $data_name<X_, $( $dim_name ),*>;

            const DIM: usize = count!($($dim_name)*);
            const LEN: usize = $($dim_name * )* 1;

            #[inline]
            fn new_boxed(data: Self::Shape) -> Box<Self> {
                Box::new(Self { data })
            }
        }

        /// Owned [`Tensor`].
        #[derive(Debug, Clone, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name<X: Element, $( const $dim_name: usize ),*> {
            data: Box<$data_name<X, $( $dim_name ),*>>,
        }

        impl<X: Element, $( const $dim_name: usize ),*> Default for $name<X, $( $dim_name ),*> {
            fn default() -> Self {
                Self::new(ArrDefault::arr_default())
            }
        }

        unsafe impl<X: Element, $( const $dim_name: usize ),*> Tensor<X> for $name<X, $( $dim_name ),*> {
            type Data = $data_name<X, $( $dim_name ),*>;

            type Mapped<X_: Element> = $name<X_, $( $dim_name ),*>;

            #[inline]
            fn from_box(data: Box<Self::Data>) -> Self {
                Self { data }
            }
        }

        impl<X: Element, $( const $dim_name: usize ),*> Deref for $name<X, $( $dim_name ),*> {
            type Target = $data_name<X, $( $dim_name ),*>;

            #[inline]
            fn deref(&self) -> &Self::Target { &self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> DerefMut for $name<X, $( $dim_name ),*> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> AsRef<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn as_ref(&self) -> &$data_name<X, $( $dim_name ),*> { &self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> AsMut<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn as_mut(&mut self) -> &mut $data_name<X, $( $dim_name ),*> { &mut self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> Borrow<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn borrow(&self) -> &$data_name<X, $( $dim_name ),*> { &self.data }
        }

        impl<X: Element, $( const $dim_name: usize ),*> BorrowMut<$data_name<X, $( $dim_name ),*>> for $name<X, $( $dim_name ),*> {
            #[inline]
            fn borrow_mut(&mut self) -> &mut $data_name<X, $( $dim_name ),*> { &mut self.data }
        }
    };
}
use crate::Element;
pub(crate) use make_tensor;
