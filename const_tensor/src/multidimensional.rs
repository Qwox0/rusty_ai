/*
#[cfg(test)]
mod tests {
    use crate::{multidim_arr::*, Element, Num};
    use core::{fmt, slice};
    use std::{marker::PhantomData, mem};

    // =====

    /*
    trait MyShape {
        type Data<X: Element>;
    }

    impl<SA: ShapeArr + Len<LEN>, const LEN: usize> MyShape for Shape<SA, LEN> {
        type Data<X: Element> = SA::Data<X>;
    }
    */

    /// # SAFETY
    /// Ensure `S::LEN == LEN`
    pub unsafe trait Shape {
        type AsArr: MultidimArr<Element = ()>;
    }

    /// # SAFETY
    ///
    /// When using this struct (in a type), the wrapping type has to ensure that only valid
    /// [`ShapeS`] are possible. Valid means `S: Len<LEN>` (S::LEN == LEN).
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ShapeS<S: MultidimArr<Element = ()>, const LEN: usize> {
        _marker: PhantomData<S>,
    }

    unsafe impl<S: MultidimArr<Element = ()>, const LEN: usize> Shape for ShapeS<S, LEN> {
        type AsArr = S;
    }

    /*
    unsafe trait MyShape2 {
        type Data<X: Element>;
    }

    /// # Safety
    /// SA::LEN == LEN
    unsafe impl<SA: ShapeArr, const LEN: usize> MyShape2 for Shape<SA, LEN> {
        type Data<X: Element> = SA::Data<X>;
    }

    impl<SA: ShapeArr + Len<LEN>, const LEN: usize> ShapeS<SA, LEN> {
        fn as_1d(tensor: &SA::Data<i32>) -> &[i32; LEN] {
            unsafe { mem::transmute(tensor) }
        }
    }
    */

    #[test]
    fn test() {
        /// `S`: [`Shape`]
        #[derive(Debug)]
        struct Tensor<X: Element, S: Shape> {
            //val: Box<<S::AsArr as MultidimArr>::Mapped<X>>,
            val: <S::AsArr as MultidimArr>::Mapped<X>,
        }

        type Vector<X, const LEN: usize> = Tensor<X, ShapeS<[(); LEN], LEN>>;
        type Matrix<X, const W: usize, const H: usize> = Tensor<X, ShapeS<[[(); W]; H], { W * H }>>;

        // ensure that only tensors with `S::LEN == LEN` are possible to create.

        impl<X: Element, S: MultidimArr<Element = ()> + Len<LEN>, const LEN: usize>
            Tensor<X, ShapeS<S, LEN>>
        {
            fn new(val: impl MultidimArr<Element = X, Mapped<()> = S>) -> Self {
                Tensor { val: Box::new(val.type_hint()) }
            }
        }

        // all following implementation assume `S::LEN == LEN`!

        impl<X: Element, S: MultidimArr<Element = ()>, const LEN: usize> Tensor<X, ShapeS<S, LEN>> {
            fn val(&self) -> &S::Mapped<X> {
                &self.val
            }

            fn into_1d(self) -> Vector<X, LEN> {
                let a = mem::ManuallyDrop::new(self);
                unsafe { mem::transmute_copy(&a) }
            }

            fn as_1d(&self) -> &Vector<X, LEN> {
                unsafe { mem::transmute(self) }
            }

            fn as_1d_mut(&mut self) -> &mut Vector<X, LEN> {
                unsafe { mem::transmute(self) }
            }

            fn iter(&self) -> impl Iterator<Item = &X> {
                self.as_1d().val.iter()
            }

            fn iter_mut(&mut self) -> impl Iterator<Item = &mut X> {
                self.as_1d_mut().val.iter_mut()
            }

            fn iter_sub2(&self) -> slice::Iter<'_, <S::Mapped<X> as MultidimArr>::Sub> {
                self.val().as_sub_slice().iter()
            }

            fn index_sub2(&self, idx: usize) -> <S::Mapped<X> as MultidimArr>::Sub {
                self.val().as_sub_slice()[idx]
            }
        }

        impl<X: Element, SUB: MultidimArr<Element = ()>, const N: usize, const LEN: usize>
            Tensor<X, ShapeS<[SUB; N], LEN>>
        {
            fn iter_sub(&self) -> slice::Iter<'_, SUB::Mapped<X>> {
                self.val().as_sub_slice().iter()
            }

            fn index_sub(&self, idx: usize) -> SUB::Mapped<X> {
                self.val().as_sub_slice()[idx]
            }

            /*
            fn index_sub_wrap(&self, idx: usize) -> Tensor<X, ShapeS<SUB, SUBLEN>> {
                self.val().as_sub_slice()[idx]
            }
            */
        }

        impl<X: Num, S: MultidimArr<Element = ()>, const LEN: usize> Tensor<X, ShapeS<S, LEN>> {
            fn sqare(&mut self) {
                self.iter_mut().for_each(|x| *x *= *x);
            }
        }

        let mut a = Tensor::new([[6, 5, 4], [3, 2, 1]]);
        let data = a.iter().copied().collect::<Vec<_>>();
        println!("{:?}", data);
        assert_eq!(data, &[6, 5, 4, 3, 2, 1]);

        a.sqare();
        let data = a.iter().copied().collect::<Vec<_>>();
        println!("{:?}", data);
        assert_eq!(data, &[36, 25, 16, 9, 4, 1]);

        println!("{:?}", a);

        println!("{:?}", a.iter_sub().collect::<Vec<_>>());
        println!("{:?}", a.iter_sub2().collect::<Vec<_>>());

        let mut a = Matrix::new([[6, 5, 4], [3, 2, 1]]);
        let data = a.iter().copied().collect::<Vec<_>>();
        println!("{:?}", data);
        assert_eq!(data, &[6, 5, 4, 3, 2, 1]);

        a.sqare();
        let data = a.iter().copied().collect::<Vec<_>>();
        println!("{:?}", data);
        assert_eq!(data, &[36, 25, 16, 9, 4, 1]);

        println!("{:?}", a);

        println!("{:?}", a.iter_sub().collect::<Vec<_>>());
        println!("{:?}", a.iter_sub2().collect::<Vec<_>>());
    }
}
*/

/*
use crate::{Element, Len, Tensor};

/// An owned multidimensional structure.
///
/// This is implemented for [`Tensor`], but this can be implemented for tensor like structure:
///
/// ```rust
/// struct MyLayer {
///     weights: Matrix<f32, 2, 4>,
///     bias: Vector<f32, 4>,
/// }
///
/// impl Multidimensional for MyLayer { ... }
/// ```
pub trait Multidimensional<X: Element> {
    /// Applies a function to every element of the tensor.
    // TODO: bench vs tensor::map_clone
    #[inline]
    fn map_inplace<const LEN: usize>(mut self, mut f: impl FnMut(X) -> X) -> Self
    where S: Len<LEN> {
        self.map_mut(|x| *x = f(*x));
        self
    }

    /// Adds a scalar to every element of the tensor.
    #[inline]
    fn scalar_add<const LEN: usize>(mut self, scalar: X) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.scalar_add_mut(scalar);
        self
    }

    /// Subtracts a scalar from every element of the tensor.
    #[inline]
    fn scalar_sub<const LEN: usize>(mut self, scalar: X) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.scalar_sub_mut(scalar);
        self
    }

    /// Multiplies the tensor by a scalar value.
    #[inline]
    fn scalar_mul<const LEN: usize>(mut self, scalar: X) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.scalar_mul_mut(scalar);
        self
    }

    /// Divides the tensor by a scalar value.
    #[inline]
    fn scalar_div<const LEN: usize>(mut self, scalar: X) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.scalar_div_mut(scalar);
        self
    }

    /// Adds `other` to `self` elementwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat1 = Matrix::new([[1, 2], [3, 4]]);
    /// let mat2 = Matrix::new([[4, 3], [2, 1]]);
    /// let res = mat1.add_elem(&mat2);
    /// assert_eq!(res._as_inner(), &[[5, 5], [5, 5]]);
    /// ```
    #[inline]
    fn add_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.add_elem_mut(other);
        self
    }

    /// Subtracts `other` from `self` elementwise.
    #[inline]
    fn sub_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.sub_elem_mut(other);
        self
    }

    /// Multiplies `other` to `self` elementwise.
    #[inline]
    fn mul_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.mul_elem_mut(other);
        self
    }

    /// Divides `other` to `self` elementwise.
    #[inline]
    fn div_elem<const LEN: usize>(mut self, other: &Self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.div_elem_mut(other);
        self
    }

    /// Squares `self` elementwise.
    #[inline]
    fn square_elem<const LEN: usize>(mut self) -> Self
    where
        S: Len<LEN>,
        X: Num,
    {
        self.square_elem_mut();
        self
    }

    /// Calculates the reciprocal of every element in `self`.
    #[inline]
    fn recip_elem<const LEN: usize>(mut self) -> Self
    where
        S: Len<LEN>,
        X: Float,
    {
        self.recip_elem_mut();
        self
    }

    /// Calculates the negative of the tensor.
    #[inline]
    fn neg<const LEN: usize>(mut self) -> Self
    where
        S: Len<LEN>,
        X: Float,
    {
        self.neg_mut();
        self
    }

    /// `self * t + other * (1 - t)` (same as `t * (self - other) + other`)
    #[inline]
    fn lerp<const LEN: usize>(mut self, other: &tensor<X, S>, blend: X) -> Self
    where
        S: Len<LEN>,
        X: Float,
    {
        self.lerp_mut(other, blend);
        self
    }
}
*/
