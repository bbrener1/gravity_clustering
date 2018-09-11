use ndarray::{Array,ArrayView,Ix1,Ix2,Axis};
use std::f64;
use std::sync::Arc;
use rand::{Rng,ThreadRng,thread_rng};
use std::cmp::{min,max};
use io::Parameters;
use ndarray_parallel::prelude::*;
use distance;
use length;

#[derive(Debug)]
pub struct Pathfinder {
    point: Array<f64,Ix1>,
    origin: Array<f64,Ix1>,
    skip: usize,
    samples: usize,
    sample_indecies: Vec<usize>,
    features: usize,
    // feature_indecies: Vec<usize>,
    points: Arc<Array<f64,Ix2>>,
    sample_subsamples: Vec<usize>,
    // feature_subsamples: Vec<usize>,
    previous_jump: Option<f64>,
    rng: ThreadRng,
    parameters: Arc<Parameters>,
}

impl Pathfinder {
    // pub fn init(origin: ArrayView<'a,f64,Ix1>, gravity_points: Arc<Array<f64,Ix2>>, skip: usize, scaling_factor:Option<f64>, subsample_arg: Option<usize>, convergence_arg: Option<f64>,locality: Option<f64>) -> Pathfinder {
    pub fn init(origin: Array<f64,Ix1>, points: Arc<Array<f64,Ix2>>, skip: usize, parameters: Arc<Parameters>) -> Pathfinder {

        let point = origin.to_owned();

        let rng = thread_rng();

        let samples = points.shape()[0];
        let features = points.shape()[1];

        let mut sample_indecies: Vec<usize> = (0..samples).collect();
        sample_indecies.remove(skip);

        let mut subsample_size = sample_indecies.len();
        if samples > 1000 {
            subsample_size = max(min(1000, subsample_size/10),2);
        };
        if parameters.sample_subsample.is_some() {
            subsample_size = parameters.sample_subsample.unwrap();
        };
        let sample_subsamples = Vec::with_capacity(subsample_size);

        assert!(parameters.scaling_factor.as_ref().map(|x| if *x == 0. {0.} else {1.}).unwrap_or(1.) != 0.);

        // if point.iter().sum::<f64>() == 0. {
        //     eprintln!("Initialized with zero:{:?}",point);
        //     panic!();
        // }

        Pathfinder {
            point: point.clone(),
            origin: origin,
            skip: skip,
            samples: samples,
            sample_indecies: sample_indecies,
            features: features,
            points: points,
            sample_subsamples: sample_subsamples,
            previous_jump: None,
            rng: rng,
            parameters: parameters,
        }

    }

    fn subsample(&mut self) {
        self.rng.shuffle(&mut self.sample_indecies);
        self.sample_subsamples = self.sample_indecies.iter().cloned().take(self.parameters.sample_subsample.unwrap_or(self.samples)).collect();
        // for (i,ss) in self.sample_subsamples.iter().enumerate() {
        //     self.sub_points.row_mut(i).assign(&self.points.row(i));
        // }
    }

    fn subsampled_nearest(&mut self, _previous_jump: f64) -> Option<(Array<f64,Ix1>,f64)> {

        self.subsample();

        let mut jump_point = None;

        let mut maximum_jump_distance = f64::MAX;

        for mut sub_point_index in self.sample_subsamples.iter() {
            // eprintln!("P:{:?}",self.point.view());
            // eprintln!("S1:{:?}",self.points.row(*sub_point_index).view());
            // eprintln!("D:{:?}",distance(self.point.view(),self.points.row(*sub_point_index).view()));
            let point_distance = distance(self.point.view(),self.points.row(*sub_point_index).view());
            if point_distance < maximum_jump_distance && point_distance > 0. {
                maximum_jump_distance = point_distance;
                jump_point = Some(sub_point_index);
            }
        }

        jump_point.map(|x| (self.points.row(*x).to_owned(),maximum_jump_distance))

    }

    pub fn step(&mut self) -> Option<()> {

        if self.point.iter().sum::<f64>() == 0. {
            eprintln!("Moved to zero:{:?}",self.point);
            panic!();
        }

        let mut previous_jump = self.previous_jump.unwrap_or(f64::MAX);

        let mut jump_point = Array::zeros(self.features);
        let mut total_distance = 0.;
        let mut bag_counter = 0;

        for _i in 0..10 {
            if let Some((bagged_point,jump_distance)) = self.subsampled_nearest(previous_jump) {
                // eprintln!("BP:{:?}",bagged_point);
                // eprintln!("BC1:{:?}",bag_counter);
                // eprintln!("JP1:{:?}",jump_point);
                jump_point += &bagged_point;
                total_distance += jump_distance;
                bag_counter += 1;
                // eprintln!("BC2:{:?}",bag_counter);
                // eprintln!("JP2:{:?}",jump_point);
            }
        };

        if bag_counter > 0 {

            jump_point /= bag_counter as f64;
            let mean_distance = total_distance / bag_counter as f64;

            // eprintln!("J:{:?}",jump_point);

            let jump_distance = distance(jump_point.view(),self.point.view());

            if jump_distance > (mean_distance / self.parameters.convergence_factor.unwrap_or(10.)) {

                // eprintln!("P:{:?}",self.point);
                // eprintln!("J:{:?}",jump_point);
                // eprintln!("MD:{:?}",mean_distance);
                // eprintln!("JD:{:?}",jump_distance);

                self.previous_jump = Some(jump_distance);
                self.point = jump_point;
                return Some(())
            }
        }

        return None

    }

    pub fn descend(&mut self) -> Array<f64,Ix1> {
        let mut motionless_count = 0;
        let mut step_count = 0;
        loop {
            if step_count % 100 == 0 {
                // eprintln!("S:{:?}",self.point);
                eprintln!("Step:{:?}",step_count);
            }
            if self.step().is_none() {
            //     motionless_count += 1;
            // };
            // if motionless_count > 50 {
                break
            };
            step_count += 1;
            if step_count > 20000000 {
                panic!("Failed to converge after 20 million steps, adjust parameters")
            }
        };
        // eprintln!("{:?}",self.point);
        self.point.to_owned()
    }

    pub fn fuzzy_descend(&mut self, fuzz:usize) -> (Array<f64,Ix1>,(f64,f64)) {

        let mut final_points: Array<f64,Ix2> = Array::zeros((fuzz,self.features));

        for i in 0..fuzz {
            self.reset();
            final_points.row_mut(i).assign(&self.descend());
        }

        let average_point = final_points.sum_axis(Axis(0))/fuzz as f64;

        // eprintln!("Average:{:?}",average_point);

        let mut av_deviation = 0.;

        for mut final_point in final_points.outer_iter_mut() {
            final_point.scaled_add(-1.,&average_point);
            av_deviation += length(final_point.view())/fuzz as f64;
        }

        eprintln!("Deviation:{:?}",av_deviation);

        let displacement = length((&self.origin - &average_point).view());

        eprintln!("Displacement:{:?}",displacement);

        (average_point,(av_deviation,displacement))

    }

    pub fn reset(&mut self) {
        self.point = self.origin.clone();
        self.previous_jump = None;
    }



}

#[cfg(test)]
mod pathfinder_tesing {

    use super::*;
    // use ndarray::{Array,Ix1,Ix2,Axis};
    // use std::collections::VecDeque;
    // use std::sync::Arc;
    // use rand::{Rng,ThreadRng,thread_rng};
    // use std::cmp::{min,max};

    fn basic_point() -> Array<f64,Ix1> {
        Array::from_vec(vec![7.,8.])
    }

    fn basic_gravity(point: Array<f64,Ix1>,parameters: Arc<Parameters>) -> Pathfinder {
        let gravity_points = Arc::new(Array::from_shape_vec((5,2),vec![10.,8.,9.,10.,3.,4.,3.,3.,5.,6.]).expect("wtf?"));
        Pathfinder::init(point, gravity_points, 6, parameters)
    }

    #[test]
    fn step_test() {
        let parameters = Parameters::empty();
        let point = basic_point();
        let mut path = basic_gravity(point,Arc::new(parameters));
        for i in 0..1000 {
            println!("{:?}",path.step());
            println!("{:?}",path.point);
        }
        // panic!()
    }
    #[test]
    fn descent_test() {
        let parameters = Parameters::empty();
        let point = basic_point();
        let mut path = basic_gravity(point,Arc::new(parameters));
        let end = path.descend();

        let hopefully_end = array![5.,6.];

        println!("{:?}",end);
        println!("{:?}",&end - &hopefully_end);

        assert!((&end - &hopefully_end)[0] < 0.1 && (&end - &hopefully_end)[1] < 0.1)
    }

}
