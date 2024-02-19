use super::*;

#[test]
fn basic_usage() {
    println!("\n# transmute_into:");
    let vec = Vector::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let mat: Tensor3<i32, 3, 2, 2> = vec.transmute_into();
    println!("{:#?}", mat);
    assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

    println!("\n# transmute_as:");
    let vec = Vector::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let vec = vec.as_ref(); // Optional
    let mat: &tensor3<i32, 3, 2, 2> = vec.transmute_as();
    println!("{:#?}", mat);
    assert_eq!(mat, tensor3::literal([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

    println!("\n# from_1d:");
    let vec = Tensor::new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let mat = Tensor3::from_1d(vec);
    println!("{:#?}", mat);
    assert_eq!(mat, Tensor::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]));

    println!("\n# iter components:");
    let mut iter = mat.iter_sub_tensors();
    assert_eq!(iter.next(), Some(matrix::literal([[1, 2, 3], [4, 5, 6]])));
    assert_eq!(iter.next(), Some(matrix::literal([[7, 8, 9], [10, 11, 12]])));
    assert_eq!(iter.next(), None);
    println!("works");

    println!("\n# iter components:");
    assert!(mat.iter_elem().enumerate().all(|(idx, elem)| *elem == 1 + idx as i32));
    println!("works");

    println!("\n# add one:");
    let mat = Matrix::new([[1i32, 2], [3, 4], [5, 6]]);
    let mat = mat.map(|x| x + 1);
    println!("{:#?}", mat);
    assert_eq!(mat, Tensor::new([[2, 3], [4, 5], [6, 7]]));

    println!("\n# dot_product:");
    let vec1 = Vector::new([1, 9, 2, 2]);
    let vec2 = Vector::new([1, 0, 5, 1]);
    let res = vec1.dot_product(vec2.as_ref());
    println!("{:#?}", res);
    assert_eq!(res, 13);

    println!("\n# mat_mul_vec:");
    let vec = Vector::new([2, 1]);
    let res = mat.mul_vec(&vec);
    println!("{:#?}", res);
    assert_eq!(res, Tensor::new([7, 13, 19]));
}
