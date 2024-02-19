use crate::{Element, Float, Num};
use std::{iter, mem, ops::DerefMut};

/// represents an object containing multiple [`Element`]s.
pub trait Multidimensional<X: Element>: Sized {
    /// Type of an [`Iterator`] over references to the inner [`Element`]s.
    type Iter<'a>: Iterator<Item = &'a X>
    where Self: 'a;
    /// Type of an [`Iterator`] over mutable references to the inner [`Element`]s.
    type IterMut<'a>: Iterator<Item = &'a mut X> + 'a
    where Self: 'a;

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
    fn iter_elem(&self) -> Self::Iter<'_>;

    /// Creates an [`Iterator`] over the mutable references to the elements of `self`.
    fn iter_elem_mut(&mut self) -> Self::IterMut<'_>;

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

impl<X: Element> Multidimensional<X> for () {
    type Iter<'a> = iter::Empty<&'a X>;
    type IterMut<'a> = iter::Empty<&'a mut X>;

    fn iter_elem(&self) -> Self::Iter<'_> {
        iter::empty()
    }

    fn iter_elem_mut(&mut self) -> Self::IterMut<'_> {
        iter::empty()
    }
}

/// An owned multidimensional structure.
///
/// This is implemented for [`Tensor`], but this can be implemented for tensor like structure:
///
/// ```rust,ignore
/// struct MyLayer {
///     weights: Matrix<f32, 2, 4>,
///     bias: Vector<f32, 4>,
/// }
///
/// impl Multidimensional for MyLayer { ... }
/// ```
pub trait MultidimensionalOwned<X: Element>: Sized + DerefMut<Target = Self::Data> {
    /// Type of the Data of this owned multidimensional object.
    type Data: Multidimensional<X>;
    /// this type but with possibly uninitialized values.
    type Uninit: MultidimensionalOwned<crate::maybe_uninit::MaybeUninit<X>>;

    /// Creates a new uninitialized multidimensional object.
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
    fn map(mut self, mut f: impl FnMut(X) -> X) -> Self {
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
