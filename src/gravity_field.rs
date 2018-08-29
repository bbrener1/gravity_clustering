use std::sync::Arc;
use ndarray::{Array,Ix1,Ix2,Zip,ArrayView};
use std::f64;

use io::Parameters;

use pathfinder::Pathfinder;


pub struct GravityField<'a> {
    gravity_points: Arc<Array<f64,Ix2>>,
    pub final_positions: Option<Array<f64,Ix2>>,
    pub clusters: Vec<(Array<f64,Ix1>,usize)>,
    parameters: &'a Parameters,
}

impl<'a> GravityField<'a> {

    pub fn init(gravity_points: Array<f64,Ix2>, parameters:&'a Parameters) -> GravityField {
        assert!(!gravity_points.iter().any(|x| x.is_nan()));
        GravityField {
            gravity_points: Arc::new(gravity_points),
            final_positions: None,
            clusters: vec![],
            parameters: parameters,
        }
    }


    pub fn fit(&mut self) -> &Option<Array<f64,Ix2>> {

        let mut final_positions: Array<f64,Ix2> = Array::zeros((self.gravity_points.shape()[0],self.gravity_points.shape()[1]));

        for (i,(point, mut store_final)) in self.gravity_points.outer_iter().zip(final_positions.outer_iter_mut()).enumerate() {
            let mut pathfinder = Pathfinder::init(point, self.gravity_points.clone(), i, self.parameters);
            eprintln!("Fitting sample {}", i);
            store_final.assign(&pathfinder.descend());
        }

        self.final_positions = Some(final_positions);

        &self.final_positions
    }

    pub fn predict(&mut self) -> Array<usize,Ix1> {

        let final_positions = self.final_positions.as_ref().unwrap();
        let merge_distance = self.parameters.merge_distance.unwrap_or(self.parameters.scaling_factor.unwrap_or(self.gravity_points.shape()[1] as f64 / 5.) * 10.) ;

        let mut cluster_labels = Array::zeros(final_positions.shape()[0]);

        for (point, mut cluster_label) in final_positions.outer_iter().zip(cluster_labels.iter_mut()) {
            // eprintln!("Merge distance: {}",merge_distance);
            // eprintln!("Local merge distance: {}", self.merge_distance.unwrap());
            for (key_point,existing_cluster) in &self.clusters {
                if distance(point, key_point.view()) < merge_distance {
                    *cluster_label = *existing_cluster
                }
            }
            if *cluster_label == 0 {
                let new_cluster = self.clusters.last().as_ref().map(|x| x.1).unwrap_or(0) + 1;
                self.clusters.push((point.to_owned(),new_cluster));
                *cluster_label = new_cluster;
            }
        }

        cluster_labels

    }

    // pub fn manual_test() {
    //     let raw = Array::from_shape_vec((5,2),vec![10.,8.,9.,10.,3.,4.,3.,3.,5.,6.]).expect("impossible");
    //     let mut field = GravityField::init(raw, Some(0.3), None);
    //     println!("{:?}",field.gravity_points);
    //     println!("{:?}",field.fit());
    //     println!("{:?}",field.final_positions);
    //     println!("{:?}",field.predict());
    // }

}

pub fn distance(pa1:ArrayView<f64,Ix1>,pa2:ArrayView<f64,Ix1>) -> f64 {
    // (p1 - p2).mapv_inplace(|x| x.powi(2)).scalar_sum().sqrt()
    let mut acc = 0.;
    Zip::from(pa1).and(pa2).apply(|p1,p2| acc += (*p1 - *p2).powi(2));
    acc = acc.sqrt();
    acc
}


#[cfg(test)]
mod gravity_testing {

    use super::*;

    pub fn basic_field() -> Array<f64, Ix2> {
        Array::from_shape_vec((5,2),vec![10.,8.,9.,10.,3.,4.,3.,3.,5.,6.]).expect("impossible")
    }

    #[test]
    pub fn test_distance() {
        let p1 = array![0.,0.];
        let p2 = array![3.,4.];
        let p3 = array![-5.,-12.];
        let p4 = array![-3.,4.];
        let p5 = array![3.,-4.];
        assert_eq!(0.,distance(p1.view(), p1.view()));
        assert_eq!(5.,distance(p1.view(), p2.view()));
        assert_eq!(13.,distance(p1.view(), p3.view()));
        assert_eq!(6.,distance(p2.view(), p4.view()));
        assert_eq!(8.,distance(p2.view(), p5.view()));
    }

    #[test]
    pub fn initization() {
        let parameters = Parameters::empty();
        let raw = basic_field();
        GravityField::init(raw, &parameters);
    }

    #[test]
    #[should_panic]
    pub fn initization_nan() {
        let parameters = Parameters::empty();
        let mut raw = basic_field();
        raw[[1,2]] = f64::NAN;
        GravityField::init(raw, &parameters);
    }

    #[test]
    pub fn fit() {
        let parameters = Parameters::empty();
        let raw = basic_field();
        let mut field = GravityField::init(raw, &parameters);
        field.fit();
    }

    #[test]
    pub fn predict() {
        let parameters = Parameters::empty();
        let raw = basic_field();
        let mut field = GravityField::init(raw, &parameters);
        field.fit();
        field.predict();
    }
}
