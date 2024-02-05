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
        type AsArr: MultidimArr;
    }

    /// # SAFETY
    ///
    /// When using this struct (in a type), the wrapping type has to ensure that only valid
    /// [`ShapeS`] are possible. Valid means `S: Len<LEN>` (S::LEN == LEN).
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ShapeS<S: MultidimArr, const LEN: usize> {
        _marker: PhantomData<S>,
    }

    unsafe impl<S: MultidimArr, const LEN: usize> Shape for ShapeS<S, LEN> {
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

    /*
    mod shape3 {
        /*
        pub trait ShapeArr {
            const LEN: usize;
            const Mapped
        }

        impl ShapeArr for () {
            const LEN: usize = 1;
        }

        impl<SUB: ShapeArr, const N: usize> ShapeArr for [SUB; N] {
            const LEN: usize = SUB::LEN * N;
        }
        */
        use super::{Len, MultidimArr};
        use crate::Element;
        use std::{marker::PhantomData, mem};

        /*
        pub struct ShapeS<SUB: ShapeArr, const LEN: usize> {
            _marker: PhantomData<SUB>,
        }

        struct tensor<X: Element, S: Shape> {
            val: [X; 1],
            _marker: PhantomData<S>,
        }
        */

        pub trait Tensor<const LEN: usize> {
            type Element: Element;
            type Sub;
            type Data: MultidimArr<Element = Self::Element>;

            fn new(data: Self::Data) -> Self
            where Self::Data: Len<LEN>;

            /// # SAFETY
            /// `Self::Data::LEN == LEN`
            unsafe fn wrap_ref_unchecked(data: &Self::Data) -> &Self;

            fn wrap_ref(data: &Self::Data) -> &Self
            where Self::Data: Len<LEN> {
                Self::wrap_ref_unchecked(data)
            }

            fn into_1d(self) -> Vector<Self::Element, LEN>;

            fn get_sub(&self, idx: usize) -> &Self::Sub;
        }

        struct Vector<X: Element, const LEN: usize> {
            data: [X; LEN],
        }

        impl<X: Element, const LEN: usize> Tensor<LEN> for Vector<X, LEN> {
            type Data = [X; LEN];
            type Element = X;
            type Sub = X;

            /// ensure that only tensors with `Self::Data::LEN == LEN` are possible to create.
            fn new(data: [X; LEN]) -> Self
            where Self::Data: Len<LEN> {
                Self { data }
            }

            unsafe fn wrap_ref_unchecked(data: &Self::Data) -> &Self {
                unsafe { mem::transmute(data) }
            }

            fn into_1d(self) -> Vector<Self::Element, LEN> {
                todo!()
            }

            fn get_sub(&self, idx: usize) -> &Self::Sub {
                let data = &self.data[idx];
                data
            }
        }

        struct Matrix<X: Element, const W: usize, const H: usize, const LEN: usize> {
            data: [[X; W]; H],
        }

        impl<X: Element, const W: usize, const H: usize, const LEN: usize> Tensor<LEN>
            for Matrix<X, W, H, LEN>
        {
            type Data = [[X; W]; H];
            type Element = X;
            type Sub = Vector<X, W>;

            /// ensure that only tensors with `Self::Data::LEN == LEN` are possible to create.
            fn new(data: [[X; W]; H]) -> Self
            where Self::Data: Len<LEN> {
                Self { data }
            }

            unsafe fn wrap_ref_unchecked(data: &Self::Data) -> &Self {
                unsafe { mem::transmute(data) }
            }

            fn into_1d(self) -> Vector<Self::Element, LEN> {
                todo!()
            }

            fn get_sub(&self, idx: usize) -> &Self::Sub {
                let data = &self.data[idx];
                // SAFETY: `<[X; W]>::LEN == Self::Sub::LEN == W`
                unsafe { Self::Sub::wrap_ref_unchecked(data) }
            }
        }

        struct Tensor3<X: Element, const A: usize, const B: usize, const C: usize, const LEN: usize> {
            data: [[[X; A]; B]; C],
        }

        impl<X: Element, const A: usize, const B: usize, const C: usize, const LEN: usize>
            Tensor<LEN> for Tensor3<X, A, B, C, LEN>
        {
            type Data = [[[X; A]; B]; C];
            type Element = X;
            type Sub = Matrix<X, A, B, { A * B }>;

            fn new(data: Self::Data) -> Self
            where Self::Data: Len<LEN> {
                todo!()
            }

            unsafe fn wrap_ref_unchecked(data: &Self::Data) -> &Self {
                todo!()
            }

            fn into_1d(self) -> Vector<Self::Element, LEN> {
                todo!()
            }

            fn get_sub(&self, idx: usize) -> &Self::Sub {
                todo!()
            }
        }
    }
    */

    #[cfg(test)]
    #[cfg(dont_compile)]
    mod shape3 {
        use super::{Len, MultidimArr};
        use crate::{Element, Num};
        use core::slice;
        use std::{mem, ops::Mul};

        trait Length<const LEN: usize> {
            type Vector<X: Element>: MultidimArr<Element = X>;
        }
        impl<const N: usize> Length<N> for [(); N] {
            type Vector<X: Element> = [X; N];
        }

        #[derive(Debug, Clone)]
        struct tensor<S: MultidimArr> {
            data: S,
        }

        impl<X: Element, S: MultidimArr<Element = X>> tensor<S> {
            fn new(data: S) -> Self {
                tensor { data }
            }

            fn wrap_ref(data: &S::Mapped<X>) -> &Self {
                todo!()
            }

            fn into_1d(self) -> tensor<[X; S::LEN]> {
                let t = mem::ManuallyDrop::new(self); // TODO
                unsafe { mem::transmute_copy(&t) }
            }

            fn as_1d(&self) -> &tensor<[X; S::LEN]> {
                unsafe { mem::transmute(self) }
            }

            fn as_1d_mut(&mut self) -> &mut tensor<[X; S::LEN]> {
                unsafe { mem::transmute(self) }
            }

            fn as_1d_mut2<const LEN: usize>(&mut self) -> &mut tensor<[X; LEN]> {
                unsafe { mem::transmute(self) }
            }

            pub fn iter_elem(&self) -> slice::Iter<'_, X>
            where [(); S::LEN]: {
                self.as_1d().data.iter()
            }

            pub fn iter_elem_mut(&mut self) -> slice::IterMut<'_, X>
//where [(); S::LEN]:
            {
                self.as_1d_mut2::<4>().data.iter_mut()
            }

            fn get_sub(&self, idx: usize) -> &tensor<S::Sub> {
                let data = &self.data.as_sub_slice()[idx];
                unsafe { mem::transmute(data) }
            }

            fn map<Y: Element>(self, f: impl FnMut(X) -> Y) -> tensor<S::Mapped<Y>> {
                todo!()
            }

            fn square_mut(&mut self)
            where
                [(); S::LEN]:,
                X: Num,
            {
                for x in self.iter_elem_mut() {
                    *x *= *x;
                }
            }
        }

        fn concatenate<const COUNT1: usize, const COUNT2: usize>(
            a: [i32; COUNT1],
            b: [i32; COUNT2],
        ) -> [i32; COUNT1 + COUNT2] {
            let mut output = [0i32; { COUNT1 + COUNT2 }];
            output
                .copy_from_slice(&a.iter().chain(b.iter()).map(|&item| item).collect::<Vec<i32>>());
            output
        }

        fn remove<const COUNT1: usize, const COUNT2: usize>(
            a: [i32; COUNT1],
        ) -> [i32; COUNT1 - COUNT2] {
            let mut output = [0i32; { COUNT1 - COUNT2 }];
            for x in 0..COUNT1 - COUNT2 {
                output[x] = a[x];
            }
            output
        }

        fn index_sub<const LEN: usize, const SUB: usize>(
            arr: &[i32; LEN],
            idx: usize,
        ) -> &[i32; SUB] {
            let ptr = arr.as_ptr();
            unsafe { mem::transmute(ptr.add(idx * SUB).as_ref()) }
        }

        fn as_sub<const LEN: usize, const COUNT: usize>(
            arr: &[i32; LEN * COUNT],
        ) -> &[[i32; LEN]; COUNT] {
            unsafe { mem::transmute(arr) }
        }

        fn iter_sub<const LEN: usize, const COUNT: usize>(
            arr: &[i32; LEN * COUNT],
        ) -> impl Iterator<Item = &[i32; LEN]> {
            as_sub(arr).iter()
        }

        fn add_n_zeros<const LEN: usize, const N: usize>(arr: [i32; LEN]) -> [i32; LEN + N] {
            concatenate(arr, [0; N])
        }

        #[test]
        fn asdfsdfds() {
            println!("{:?}", remove::<6, 3>([1, 2, 3, 4, 5, 6]));
            let arr = [1, 2, 3, 4, 5, 6];

            for a in iter_sub::<3, 2>(&[1, 2, 3, 4, 5, 6]) {
                println!("{:?}", a);
            }
        }

        #[test]
        #[cfg(dont_compile)]
        fn test() {
            let t3 = tensor::new([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
            println!("{:?}", t3);

            let arr = t3.clone().into_1d();
            println!("{:?}", arr);

            let sub = t3.get_sub(1);
            println!("{:?}", sub);

            panic!()
        }
    }

    mod shape4 {
        use crate::Element;
        use core::slice;
        use std::{mem, ptr::slice_from_raw_parts};

        trait Shape: Sized {
            type Sub: Shape;
            type Data<X: Element>;

            const LEN: usize;

            fn get_sub(&self) -> &Self::Sub;

            fn as_1d(&self) -> &[(); Self::LEN];

            fn iter_elem(&self) -> impl Iterator<Item = &()>;
            fn iter_elem_mut(&mut self) -> impl Iterator<Item = &mut ()>;

            fn double_mut(&mut self) {
                for x in self.iter_elem_mut() {
                    todo!()
                }
            }

            //fn as_1d_t<X: Element>(t: &tensor<X, Self>) -> &[(); Self::LEN];
        }

        impl Shape for () {
            type Data<X: Element> = X;
            type Sub = ();

            const LEN: usize = 1;

            fn get_sub(&self) -> &Self::Sub {
                self
            }

            fn as_1d(&self) -> &[(); Self::LEN] {
                unsafe { mem::transmute(self) }
            }

            fn iter_elem(&self) -> impl Iterator<Item = &()> {
                Some(self).into_iter()
            }

            fn iter_elem_mut(&mut self) -> impl Iterator<Item = &mut ()> {
                Some(self).into_iter()
            }
        }

        impl<SUB: Shape, const N: usize> Shape for [SUB; N] {
            type Data<X: Element> = [SUB::Data<X>; N];
            type Sub = SUB;

            const LEN: usize = SUB::LEN * N;

            fn get_sub(&self) -> &Self::Sub {
                &self[0]
            }

            fn as_1d(&self) -> &[(); Self::LEN] {
                unsafe { mem::transmute(self) }
            }

            fn iter_elem(&self) -> impl Iterator<Item = &()> {
                let ptr = self.as_ptr() as *const ();
                unsafe { slice::from_raw_parts(ptr, Self::LEN) }.iter()
            }

            fn iter_elem_mut(&mut self) -> impl Iterator<Item = &mut ()> {
                let ptr = self.as_mut_ptr() as *mut ();
                unsafe { slice::from_raw_parts_mut(ptr, Self::LEN) }.iter_mut()
            }
        }

        pub struct tensor<X: Element, S: Shape> {
            data: S::Data<X>,
        }

        #[test]
        fn test() {
            let s = [[[(); 2]; 4]; 3];
            println!("{:?}", s);
            println!("{:?}", s.len());
            println!("{:?}", s.get_sub());
            println!("{:?}", s.as_1d());

            panic!()
        }
    }

    #[test]
    #[cfg(dont_compile)]
    fn test() {
        struct tensor<X: Element, S: Shape> {
            val: <S::AsArr as MultidimArr>::Mapped<X>,
        }

        // ensure that only tensors with `S::LEN == LEN` are possible to create.

        impl<X: Element, S: MultidimArr<Element = ()> + Len<LEN>, const LEN: usize>
            tensor<X, ShapeS<S, LEN>>
        {
            fn new(val: impl MultidimArr<Element = X, Mapped<()> = S>) -> Self {
                Tensor { val: Box::new(val.type_hint()) }
            }
        }

        // all following implementation assume `S::LEN == LEN`!

        impl<X: Element, S: MultidimArr<Element = ()>, const LEN: usize> tensor<X, ShapeS<S, LEN>> {
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

            fn index_sub2(&self, idx: usize) -> &<S::Mapped<X> as MultidimArr>::Sub {
                &self.val.as_sub_slice()[idx]
            }
        }

        // ==================================

        /// `S`: [`Shape`]
        #[derive(Debug)]
        struct Tensor<X: Element, S: Shape> {
            val: Box<<S::AsArr as MultidimArr>::Mapped<X>>,
            //val: <S::AsArr as MultidimArr>::Mapped<X>,
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

            fn from_1d(vec: Vector<X, LEN>) -> Self {
                todo!()
            }
        }

        // all following implementation assume `S::LEN == LEN`!

        impl<X: Element, S: MultidimArr<Element = ()>, const LEN: usize> Tensor<X, ShapeS<S, LEN>> {
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

            fn index_sub2(&self, idx: usize) -> &<S::Mapped<X> as MultidimArr>::Sub {
                &self.val.as_sub_slice()[idx]
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

use crate::{Element, Float, Len, Num, Tensor};
use std::{
    io::Write,
    mem,
    ops::{Deref, DerefMut, Neg},
};

pub trait Multidimensional<X: Element>: Sized {
    //type Element: Element;

    /// Creates an [`Iterator`] over the references to the elements of `self`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mat = Matrix::new([[1, 2], [3, 4]]);
    /// let mut iter = mat.iter_elem();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn iter_elem(&self) -> impl Iterator<Item = &X>;

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    fn iter_elem_mut(&mut self) -> impl Iterator<Item = &mut X>;

    /// Sets every element of the tensor to the scalar value `val`.
    #[inline]
    fn fill(&mut self, val: X) {
        for x in self.iter_elem_mut() {
            *x = val;
        }
    }

    /// Sets every element of the tensor to the scalar value `0`.
    #[inline]
    fn fill_zero(&mut self)
    where X: Num {
        self.fill(X::ZERO)
    }

    /// Sets every element of the tensor to the scalar value `1`.
    #[inline]
    fn fill_one(&mut self)
    where X: Num {
        self.fill(X::ONE)
    }

    /// Applies a function to every element of the tensor.
    #[inline]
    fn map_mut(&mut self, f: impl FnMut(&mut X)) {
        self.iter_elem_mut().for_each(f);
    }

    /// Adds a scalar to every element of the tensor inplace.
    #[inline]
    fn scalar_add_mut(&mut self, scalar: X)
    where X: Num {
        for x in self.iter_elem_mut() {
            *x += scalar;
        }
    }

    /// Subtracts a scalar from every element of the tensor inplace.
    #[inline]
    fn scalar_sub_mut(&mut self, scalar: X)
    where X: Num {
        for x in self.iter_elem_mut() {
            *x -= scalar;
        }
    }

    /// Multiplies the tensor by a scalar value inplace.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat = Matrix::new([[1, 2], [3, 4]]);
    /// mat.scalar_mul_mut(10);
    /// assert_eq!(mat._as_inner(), &[[10, 20], [30, 40]]);
    /// ```
    #[inline]
    fn scalar_mul_mut(&mut self, scalar: X)
    where X: Num {
        for x in self.iter_elem_mut() {
            *x *= scalar;
        }
    }

    /// Divides the tensor by a scalar value inplace.
    #[inline]
    fn scalar_div_mut(&mut self, scalar: X)
    where X: Num {
        for x in self.iter_elem_mut() {
            *x /= scalar;
        }
    }

    /// Adds `other` to `self` elementwise and inplace.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat1 = Matrix::new([[1, 2], [3, 4]]);
    /// let mat2 = Matrix::new([[4, 3], [2, 1]]);
    /// mat1.add_elem_mut(&mat2);
    /// assert_eq!(mat1._as_inner(), &[[5, 5], [5, 5]]);
    /// ```
    #[inline]
    fn add_elem_mut(&mut self, other: &Self)
    where X: Num {
        for (x, y) in self.iter_elem_mut().zip(other.iter_elem()) {
            *x += *y;
        }
    }

    /// Subtracts `other` from `self` elementwise and inplace.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use const_tensor::{Matrix, Tensor, TensorData};
    /// let mut mat1 = Matrix::new([[5, 5], [5, 5]]);
    /// let mat2 = Matrix::new([[4, 3], [2, 1]]);
    /// mat1.sub_elem_mut(&mat2);
    /// assert_eq!(mat1._as_inner(), &[[1, 2], [3, 4]]);
    /// ```
    #[inline]
    fn sub_elem_mut(&mut self, other: &Self)
    where X: Num {
        for (x, y) in self.iter_elem_mut().zip(other.iter_elem()) {
            *x -= *y;
        }
    }

    /// Multiplies `other` to `self` elementwise and inplace.
    #[inline]
    fn mul_elem_mut(&mut self, other: &Self)
    where X: Num {
        for (x, y) in self.iter_elem_mut().zip(other.iter_elem()) {
            *x *= *y;
        }
    }

    /// Divides `other` to `self` elementwise and inplace.
    #[inline]
    fn div_elem_mut(&mut self, other: &Self)
    where X: Num {
        for (x, y) in self.iter_elem_mut().zip(other.iter_elem()) {
            *x /= *y;
        }
    }

    /// Squares `self` elementwise and inplace.
    #[inline]
    fn square_elem_mut(&mut self)
    where X: Num {
        for x in self.iter_elem_mut() {
            *x *= *x;
        }
    }

    /// Calculates the reciprocal of every element in `self` inplace.
    #[inline]
    fn recip_elem_mut(&mut self)
    where X: Float {
        use num::Float;
        for x in self.iter_elem_mut() {
            *x = x.recip();
        }
    }

    /// Calculates the negative of the tensor inplace.
    #[inline]
    fn neg_mut(&mut self)
    where X: Float {
        for x in self.iter_elem_mut() {
            *x = x.neg();
        }
    }

    /// Linear interpolation between `self` and `other` with blend value `blend`.
    ///
    /// `self * t + other * (1 - t)` (same as `t * (self - other) + other`)
    #[inline]
    fn lerp_mut(&mut self, other: &Self, blend: X)
    where X: Float {
        for (a, b) in self.iter_elem_mut().zip(other.iter_elem()) {
            *a = blend.mul_add(*a - *b, *b)
        }
    }
}

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
pub trait MultidimensionalOwned<X: Element>: Sized + DerefMut<Target = Self::Data> {
    type Data: Multidimensional<X>;
    type Uninit: MultidimensionalOwned<crate::maybe_uninit::MaybeUninit<X>>;

    fn new_uninit() -> Self::Uninit;

    /// Creates a new Tensor filled with the values in `iter`.
    /// If the [`Iterator`] it too small, the rest of the elements contain the value `X::default()`.
    #[inline]
    fn from_iter(iter: impl IntoIterator<Item = X>) -> Self {
        let mut t = Self::new_uninit();
        for (x, val) in t.iter_elem_mut().zip(iter) {
            x.write(val);
        }
        let t = mem::ManuallyDrop::new(t);
        unsafe { mem::transmute_copy(&t) }
    }

    /// Creates a new Tensor filled with the scalar value.
    #[inline]
    fn full(val: X) -> Self {
        //Self::from_1d(Vector::new([val; LEN])) // TODO: bench
        let mut t = Self::new_uninit();
        for x in t.iter_elem_mut() {
            x.write(val);
        }
        let t = mem::ManuallyDrop::new(t);
        unsafe { mem::transmute_copy(&t) }
    }

    /// Creates a new Tensor filled with the scalar value `0`.
    #[inline]
    fn zeros() -> Self
    where X: Num {
        Self::full(X::ZERO)
    }

    /// Creates a new Tensor filled with the scalar value `1`.
    #[inline]
    fn ones() -> Self
    where X: Num {
        Self::full(X::ONE)
    }

    /// Applies a function to every element of the tensor.
    // TODO: bench vs tensor::map_clone
    #[inline]
    fn map_inplace(mut self, mut f: impl FnMut(X) -> X) -> Self {
        self.map_mut(|x| *x = f(*x));
        self
    }

    /// Adds a scalar to every element of the tensor.
    #[inline]
    fn scalar_add(mut self, scalar: X) -> Self
    where X: Num {
        self.scalar_add_mut(scalar);
        self
    }

    /// Subtracts a scalar from every element of the tensor.
    #[inline]
    fn scalar_sub(mut self, scalar: X) -> Self
    where X: Num {
        self.scalar_sub_mut(scalar);
        self
    }

    /// Multiplies the tensor by a scalar value.
    #[inline]
    fn scalar_mul(mut self, scalar: X) -> Self
    where X: Num {
        self.scalar_mul_mut(scalar);
        self
    }

    /// Divides the tensor by a scalar value.
    #[inline]
    fn scalar_div(mut self, scalar: X) -> Self
    where X: Num {
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
    fn add_elem(mut self, other: &Self) -> Self
    where X: Num {
        self.add_elem_mut(other);
        self
    }

    /// Subtracts `other` from `self` elementwise.
    #[inline]
    fn sub_elem(mut self, other: &Self) -> Self
    where X: Num {
        self.sub_elem_mut(other);
        self
    }

    /// Multiplies `other` to `self` elementwise.
    #[inline]
    fn mul_elem(mut self, other: &Self) -> Self
    where X: Num {
        self.mul_elem_mut(other);
        self
    }

    /// Divides `other` to `self` elementwise.
    #[inline]
    fn div_elem(mut self, other: &Self) -> Self
    where X: Num {
        self.div_elem_mut(other);
        self
    }

    /// Squares `self` elementwise.
    #[inline]
    fn square_elem(mut self) -> Self
    where X: Num {
        self.square_elem_mut();
        self
    }

    /// Calculates the reciprocal of every element in `self`.
    #[inline]
    fn recip_elem(mut self) -> Self
    where X: Float {
        self.recip_elem_mut();
        self
    }

    /// Calculates the negative of the tensor.
    #[inline]
    fn neg(mut self) -> Self
    where X: Float {
        self.neg_mut();
        self
    }

    /// `self * t + other * (1 - t)` (same as `t * (self - other) + other`)
    #[inline]
    fn lerp(mut self, other: &Self::Data, blend: X) -> Self
    where X: Float {
        self.lerp_mut(other, blend);
        self
    }
}
