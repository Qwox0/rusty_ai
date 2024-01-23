use crate::{tensor, Tensor};

macro_rules! make_aliases {
    ( $(
        $owned_name:ident $data_name:ident :
        $($dim_name:ident)* =>
        $shape_name:ident : $shape:ty
    ),* $(,)? ) => { $(
        /// tensor data
        #[allow(non_camel_case_types)]
        pub type $data_name<X, $(const $dim_name: usize),*> = tensor<X, $shape>;

        /// owned tensor
        pub type $owned_name<X, $(const $dim_name: usize),*> = Tensor<X, $shape>;

        /// tensor shape
        pub type $shape_name<$(const $dim_name: usize),*> = $shape;
    )* };
}

make_aliases! {
    Scalar scalar: => ScalarShape: (),
    Vector vector: N => VectorShape: [(); N],
    Matrix matrix: W H => MatrixShape: [[(); W]; H],
    Tensor3 tensor3: A B C => Tensor3Shape: [[[(); A]; B]; C],
    Tensor4 tensor4: A B C D => Tensor4Shape: [[[[(); A]; B]; C]; D],
    Tensor5 tensor5: A B C D E => Tensor5Shape: [[[[[(); A]; B]; C]; D]; E],
    Tensor6 tensor6: A B C D E F => Tensor6Shape: [[[[[[(); A]; B]; C]; D]; E]; F],
    Tensor7 tensor7: A B C D E F G => Tensor7Shape: [[[[[[[(); A]; B]; C]; D]; E]; F]; G],
    Tensor8 tensor8: A B C D E F G H =>
        Tensor8Shape: [[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H],
    Tensor9 tensor9: A B C D E F G H I =>
        Tensor9Shape: [[[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H]; I],
    Tensor10 tensor10: A B C D E F G H I J =>
        Tensor10Shape: [[[[[[[[[[(); A]; B]; C]; D]; E]; F]; G]; H]; I]; J],
}
