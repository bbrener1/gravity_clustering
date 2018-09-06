use ndarray::{Array,Ix1,Ix2,Axis,Zip};
use std::collections::VecDeque;
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
    feature_indecies: Vec<usize>,
    gravity_points: Arc<Array<f64,Ix2>>,
    sub_gravity_points: Array<f64,Ix2>,
    sample_subsamples: Vec<usize>,
    feature_subsamples: Vec<usize>,
    previous_steps: VecDeque<Array<f64,Ix1>>,
    rng: ThreadRng,
    parameters: Arc<Parameters>,
}

impl Pathfinder {
    // pub fn init(origin: ArrayView<'a,f64,Ix1>, gravity_points: Arc<Array<f64,Ix2>>, skip: usize, scaling_factor:Option<f64>, subsample_arg: Option<usize>, convergence_arg: Option<f64>,locality: Option<f64>) -> Pathfinder {
    pub fn init(origin: Array<f64,Ix1>, gravity_points: Arc<Array<f64,Ix2>>, skip: usize, parameters: Arc<Parameters>) -> Pathfinder {

        let point = origin.to_owned();

        let rng = thread_rng();

        let samples = gravity_points.shape()[0];
        let features = gravity_points.shape()[1];

        let mut sample_indecies: Vec<usize> = (0..samples).collect();
        sample_indecies.remove(skip);
        let feature_indecies = (0..features).collect();

        let mut subsample_size = gravity_points.shape()[0];
        if samples > 1000 {
            subsample_size = max(min(1000, samples/10),2);
        };
        if parameters.sample_subsample.is_some() {
            subsample_size = parameters.sample_subsample.unwrap();
        };
        let feature_subsample_size = parameters.feature_subsample.unwrap();
        let sample_subsamples = Vec::with_capacity(subsample_size);
        let feature_subsamples = Vec::with_capacity(parameters.feature_subsample.unwrap_or(features));

        let mut sub_gravity_points = Array::zeros((subsample_size,features));

        assert!(parameters.scaling_factor.as_ref().map(|x| if *x == 0. {0.} else {1.}).unwrap_or(1.) != 0.);

        Pathfinder {
            point: point.clone(),
            origin: origin,
            skip: skip,
            samples: samples,
            sample_indecies: sample_indecies,
            features: features,
            feature_indecies: feature_indecies,
            gravity_points: gravity_points,
            sub_gravity_points: sub_gravity_points,
            sample_subsamples: sample_subsamples,
            feature_subsamples: feature_subsamples,
            previous_steps: VecDeque::with_capacity(50),
            rng: rng,
            parameters: parameters,
        }
    }

    fn subsample(&mut self) {
        self.rng.shuffle(&mut self.sample_indecies);
        self.sample_subsamples = self.sample_indecies.iter().cloned().take(self.parameters.sample_subsample.unwrap_or(self.samples)).collect();
        self.rng.shuffle(&mut self.feature_indecies);
        self.feature_subsamples = self.feature_indecies.iter().cloned().take(self.parameters.feature_subsample.unwrap_or(self.features)).collect();
        for (i,ss) in self.sample_subsamples.iter().enumerate() {
            for (j,fs) in self.feature_subsamples.iter().enumerate() {
                self.sub_gravity_points[[i,j]] = self.gravity_points[[*ss,*fs]]
            }
        }

    }

    fn step(&mut self) -> Array<f64,Ix1> {
    // fn step(&mut self) {
        self.subsample();
        for mut sub_point in self.sub_gravity_points.outer_iter_mut() {
            // eprintln!("S1:{:?}",sub_point);
            sub_point.scaled_add(-1.,&self.point);
            // eprintln!("S2:{:?}",sub_point);
            // let distance: f64 = sub_point.iter().map(|x| x.abs()).sum();
            let distance = length(sub_point.view());
            if distance == 0. {
                sub_point.fill(0.);
            }
            else {
                // eprintln!("D:{:?}",distance);
                // eprintln!("D:{:?}",distance.powf(self.parameters.locality.unwrap_or(3.)));
                sub_point /= distance.powf(self.parameters.locality.unwrap_or(3.));
            }
            // eprintln!("S3:{:?}",sub_point);
        }
        let mut sum: Array<f64,Ix1> = self.sub_gravity_points.sum_axis(Axis(0));
        // eprintln!("SD:{:?}", sum);
        let sum_length = length(sum.view());
        let scaling_factor = self.parameters.scaling_factor.unwrap_or(0.1);

        sum /= sum_length / scaling_factor;

        for (feature,shift) in self.feature_subsamples.iter().zip(sum.iter()) {
            self.point[*feature] += shift;
        }

        return self.point.to_owned()
    }

    pub fn descend(&mut self) -> Array<f64,Ix1> {
        for _i in 0..50 {
            self.step();
            self.previous_steps.push_front(self.point.clone());
        }
        let mut step_count = 50;
        loop {
            if step_count % 100 == 0 {
                // eprintln!("S:{:?}",self.point);
                // eprintln!("Step:{:?}",step_count);
            }
            self.step();
            self.previous_steps.push_front(self.point.to_owned());
            self.previous_steps.pop_back();
            let smoothed_displacement = (self.previous_steps.back().unwrap() - &self.point).fold(0.,|acc,x| acc + x.powi(2)).sqrt();
            if smoothed_displacement < self.parameters.convergence_factor.unwrap_or(1.)*self.parameters.scaling_factor.unwrap_or(0.1) {
                break
            }
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
        self.previous_steps.truncate(0);

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
