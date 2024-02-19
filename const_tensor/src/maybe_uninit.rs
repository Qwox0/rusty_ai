use crate::Element;
use core::{fmt, mem};
use serde::{Deserialize, Deserializer, Serialize};

/// Wrapper for [`mem::MaybeUninit`] which implements [`Element`].
///
/// Some trait implementations ([`fmt::Display`], [`Serialize`], ...) just assume that the content
/// is initialized. This struct should only be used when initiallizing tensors.
///
/// But using these trait implementations is still UNSAFE!
#[derive(Debug, Copy)]
#[repr(transparent)]
pub struct MaybeUninit<T>(pub mem::MaybeUninit<T>);

impl<T: Element> Element for MaybeUninit<T> {}

impl<T: Copy> Clone for MaybeUninit<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Default> Default for MaybeUninit<T> {
    fn default() -> Self {
        Self(mem::MaybeUninit::uninit())
    }
}

impl<T: fmt::Display> fmt::Display for MaybeUninit<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: The wrapper will probably never be displayed.
        write!(f, "{}", unsafe { self.0.assume_init_ref() })
    }
}

impl<T: PartialEq> PartialEq for MaybeUninit<T> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.0.assume_init_ref() == other.0.assume_init_ref() }
    }
}

impl<T: Eq> Eq for MaybeUninit<T> {}

impl<T: Serialize> Serialize for MaybeUninit<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        unsafe { self.0.assume_init_ref() }.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for MaybeUninit<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        T::deserialize(deserializer).map(MaybeUninit::new)
    }
}

impl<T> MaybeUninit<T> {
    pub const fn new(val: T) -> MaybeUninit<T> {
        MaybeUninit(mem::MaybeUninit::new(val))
    }

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
