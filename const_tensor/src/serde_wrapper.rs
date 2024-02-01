use crate::{tensor, Element, Tensor};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/*
/// [`Tensor`] helper which implements [`Serialize`].
///
/// To deserialize the Data use [`DeTensor`].
#[derive(Debug, Clone, Serialize)]
pub struct SerTensor<'a, X: Element> {
    data: &'a [X],
}

/// [`Tensor`] helper which implements [`Deserialize`].
#[derive(Debug, Clone, Deserialize)]
pub struct DeTensor<X: Element> {
    data: Box<[X]>,
}

impl<'a, X: Element, S: Shape> From<&'a tensor<X, S>> for SerTensor<'a, X>
where S: Len<{ S::LEN }>
{
    fn from(value: &'a tensor<X, S>) -> Self {
        let data = value.as_1d().0.as_slice();
        SerTensor { data }
    }
}

impl<X: Element, S: Shape> TryFrom<DeTensor<X>> for Tensor<X, S>
where S: Len<{ S::LEN }>
{
    type Error = Box<[X]>;

    fn try_from(value: DeTensor<X>) -> Result<Self, Self::Error> {
        let box_arr: Box<[X; S::LEN]> = value.data.try_into()?;
        Ok(Tensor::from_1d(Vector::from(vector::wrap_box(box_arr))))
    }
}
*/

// 2

/*
#[derive(Debug, Clone, Serialize)]
pub struct SerTensor<'a, X: Element, S: Shape>
where S::Data<X>: Serialize
{
    data: &'a S::Data<X>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeTensor<X: Element, S: Shape>
where S::Data<X>: DeserializeOwned
{
    data: Box<S::Data<X>>,
}

impl<'a, X: Element, S: Shape> From<&'a tensor<X, S>> for SerTensor<'a, X, S>
where S::Data<X>: Serialize
{
    fn from(tensor(data): &'a tensor<X, S>) -> Self {
        SerTensor { data }
    }
}

impl<X: Element, S: Shape> From<DeTensor<X, S>> for Tensor<X, S>
where S::Data<X>: DeserializeOwned
{
    fn from(value: DeTensor<X, S>) -> Self {
        Tensor::from(tensor::wrap_box(value.data))
    }
}
*/
