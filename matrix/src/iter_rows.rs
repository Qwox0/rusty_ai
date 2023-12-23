use core::slice;
use std::{fmt::Debug, iter::FusedIterator, marker::PhantomData, ptr::NonNull};

iter_rows_impl! {
    /// Matrix row slice [`Iterator`].
    ///
    /// This implementation is faster than simply using [`std::slice::Chunks`] or
    /// [`std::slice::ChunksExact`].
    ///
    /// implemented similar to [`core::slice::Iter`].
    ///
    /// ```rust
    /// # use matrix::Matrix;
    /// let m = Matrix::from([[0.3, 1.0], [0.1, 2.0]]);
    /// let v = &[10.0, 1.0];
    /// assert_eq!(&m.mul_vec(v), &[4.0, 3.0]);
    /// ```
    ///
    /// ```rust
    /// # use matrix::Matrix;
    /// let m = Matrix::from([[0.3, 1.0], [0.1, 2.0]]);
    /// let mut iter = m.iter_rows();
    /// assert_eq!(iter.next(), Some([0.3, 1.0].as_slice()));
    /// assert_eq!(iter.next(), Some([0.1, 2.0].as_slice()));
    /// assert_eq!(iter.next(), None);
    /// ```
    struct IterRows:
        slice::from_raw_parts => &'a [T],
        &'a T,
        as_ptr => *const T,
        const,
}

iter_rows_impl! {
    /// Mutable matrix row slice [`Iterator`].
    struct IterRowsMut:
        slice::from_raw_parts_mut => &'a mut [T],
        &'a mut T,
        as_mut_ptr => *mut T,
}

macro_rules! _iter_rows_impl {
    (
        $(#[doc = $struct_doc:expr])*
        struct
        $name:ident :
        $make_slice:path =>
        $slice:ty,
        $ref:ty,
        $as_ptr:ident =>
        $ptr:ty,
        $($const:tt)? $(,)?
    ) => {
        $(#[doc = $struct_doc])*
        #[derive(Clone)]
        pub struct $name<'a, T: Sized> {
            ptr: NonNull<T>,
            row_len: usize,
            end: $ptr,
            _marker: PhantomData<$ref>,
        }

        impl<'a, T> $name<'a, T> {
            /// # Safety
            ///
            /// `data.len()` must be a multiple of `row_len`.
            #[inline]
            pub(super) $($const)? unsafe fn new(data: $slice, row_len: usize) -> $name<'a, T> {
                let ptr = data.$as_ptr();
                // SAFETY: see `core::slice::Iter::new()`
                unsafe {
                    let end = ptr.add(data.len());
                    let ptr = NonNull::new_unchecked(ptr as *mut T);
                    $name { ptr, row_len, end, _marker: PhantomData }
                }
            }
        }

        impl<T: Debug> Debug for $name<'_, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let rows = (0..self.len())
                    .map(|idx| {
                        let start =
                            unsafe { self.ptr.add(idx * self.row_len).as_ptr() } as *const T;
                        unsafe { std::slice::from_raw_parts(start, self.row_len) }
                    })
                    .collect::<Vec<_>>();
                f.debug_tuple(stringify!($name)).field(&rows).finish()
            }
        }

        unsafe impl<T: Sync> Sync for $name<'_, T> {}
        unsafe impl<T: Sync> Send for $name<'_, T> {}

        impl<T> ExactSizeIterator for $name<'_, T> {
            #[inline]
            fn len(&self) -> usize {
                unsafe { self.end.sub_ptr(self.ptr.as_ptr() as *const T) }
                    .checked_div(self.row_len)
                    .unwrap_or(
                        0, // data.len() == row_len == 0
                    )
            }

            #[inline]
            fn is_empty(&self) -> bool {
                self.ptr.as_ptr() as *const T == self.end
            }
        }

        impl<'a, T> Iterator for $name<'a, T> {
            type Item = $slice;

            fn next(&mut self) -> Option<Self::Item> {
                if self.is_empty() {
                    None
                } else {
                    let start = self.ptr.as_ptr() as $ptr;
                    unsafe { self.ptr = self.ptr.add(self.row_len) };
                    Some(unsafe { $make_slice(start, self.row_len) })
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let exact = self.len();
                (exact, Some(exact))
            }

            #[inline]
            fn count(self) -> usize {
                self.len()
            }

            /*
            #[inline]
            fn nth(&mut self, n: usize) -> Option<&'a T> {
                if n >= len!(self) {
                    if_zst!(mut self,len =>  *len = 0,end => self.ptr =  *end,);
                    return None;
                }
                unsafe {
                    self.post_inc_start(n);
                    Some(next_unchecked!(self))
                }
            }

            #[inline]
            fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
                let advance = cmp::min(len!(self), n);
                unsafe { self.post_inc_start(advance) };
                NonZeroUsize::new(n - advance).map_or(Ok(()), Err)
            }

            #[inline]
            fn last(mut self) -> Option<&'a T> {
                self.next_back()
            }

            #[inline]
            fn fold<B, F>(self, init: B, mut f: F) -> B
            where F: FnMut(B, Self::Item) -> B {
                if is_empty!(self) {
                    return init;
                }
                let mut acc = init;
                let mut i = 0;
                let len = len!(self);
                loop {
                    acc = f(acc, unsafe { &*self.ptr.add(i).as_ptr() });
                    i = unsafe { i.unchecked_add(1) };
                    if i == len {
                        break;
                    }
                }
                acc
            }

            #[inline]
            fn for_each<F>(mut self, mut f: F)
            where
                Self: Sized,
                F: FnMut(Self::Item),
            {
                while let Some(x) = self.next() {
                    f(x);
                }
            }

            #[inline]
            fn all<F>(&mut self, mut f: F) -> bool
            where
                Self: Sized,
                F: FnMut(Self::Item) -> bool,
            {
                while let Some(x) = self.next() {
                    if !f(x) {
                        return false;
                    }
                }
                true
            }

            #[inline]
            fn any<F>(&mut self, mut f: F) -> bool
            where
                Self: Sized,
                F: FnMut(Self::Item) -> bool,
            {
                while let Some(x) = self.next() {
                    if f(x) {
                        return true;
                    }
                }
                false
            }

            #[inline]
            fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
            where
                Self: Sized,
                P: FnMut(&Self::Item) -> bool,
            {
                while let Some(x) = self.next() {
                    if predicate(&x) {
                        return Some(x);
                    }
                }
                None
            }

            #[inline]
            fn find_map<B, F>(&mut self, mut f: F) -> Option<B>
            where
                Self: Sized,
                F: FnMut(Self::Item) -> Option<B>,
            {
                while let Some(x) = self.next() {
                    if let Some(y) = f(x) {
                        return Some(y);
                    }
                }
                None
            }

            #[inline]
            #[rustc_inherit_overflow_checks]
            fn position<P>(&mut self, mut predicate: P) -> Option<usize>
            where
                Self: Sized,
                P: FnMut(Self::Item) -> bool,
            {
                let n = len!(self);
                let mut i = 0;
                while let Some(x) = self.next() {
                    if predicate(x) {
                        unsafe { assume(i < n) };
                        return Some(i);
                    }
                    i += 1;
                }
                None
            }

            #[inline]
            fn rposition<P>(&mut self, mut predicate: P) -> Option<usize>
            where
                P: FnMut(Self::Item) -> bool,
                Self: Sized + ExactSizeIterator + DoubleEndedIterator,
            {
                let n = len!(self);
                let mut i = n;
                while let Some(x) = self.next_back() {
                    i -= 1;
                    if predicate(x) {
                        unsafe { assume(i < n) };
                        return Some(i);
                    }
                }
                None
            }

            #[inline]
            unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
                unsafe { &*self.ptr.as_ptr().add(idx) }
            }

            fn is_sorted_by<F>(self, mut compare: F) -> bool
            where
                Self: Sized,
                F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>,
            {
                self.as_slice().is_sorted_by(|a, b| compare(&a, &b))
            }
            */
        }

        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$slice> {
                if self.is_empty() {
                    None
                } else {
                    self.end = unsafe { self.end.sub(self.row_len) };
                    Some(unsafe { $make_slice(self.end, self.row_len) })
                }
            }

            /*
            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<&'a T> {
                if n >= len!(self) {
                    if_zst!(mut self,len =>  *len = 0,end =>  *end = self.ptr,);
                    return None;
                }
                unsafe {
                    self.pre_dec_end(n);
                    Some(next_back_unchecked!(self))
                }
            }

            #[inline]
            fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
                let advance = cmp::min(len!(self), n);
                unsafe { self.pre_dec_end(advance) };
                NonZeroUsize::new(n - advance).map_or(Ok(()), Err)
            }
            */
        }

        impl<T> FusedIterator for $name<'_, T> {}
    };
}
use _iter_rows_impl as iter_rows_impl;
