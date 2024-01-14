use crate::Element;
use core::{fmt, mem};

/// Wrapper for [`mem::MaybeUninit`] which implements [`Element`].
#[derive(Debug, Copy)]
#[repr(transparent)]
pub struct MaybeUninit<T>(pub mem::MaybeUninit<T>);

impl<T: Element> Element for MaybeUninit<T> {}

impl<T: Copy> Clone for MaybeUninit<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: fmt::Display> fmt::Display for MaybeUninit<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: The wrapper will probably never be displayed.
        write!(f, "{}", unsafe { self.0.assume_init_ref() })
    }
}

impl<T: Default> Default for MaybeUninit<T> {
    fn default() -> Self {
        Self(mem::MaybeUninit::uninit())
    }
}

impl<T> MaybeUninit<T> {
    pub const fn uninit() -> MaybeUninit<T> {
        Self(mem::MaybeUninit::uninit())
    }

    pub const unsafe fn assume_init(self) -> T {
        self.0.assume_init()
    }

    pub fn write(&mut self, val: T) -> &mut T {
        self.0.write(val)
    }
}
