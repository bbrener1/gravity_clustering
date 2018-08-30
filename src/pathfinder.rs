use ndarray::{Array,Ix1,Ix2,Axis,Zip};
use std::collections::VecDeque;
use std::sync::Arc;
use rand::{Rng,ThreadRng,thread_rng};
use std::cmp::{min,max};
use io::Parameters;
use ndarray_parallel::prelude::*;



#[derive(Debug)]
pub struct Pathfinder {
    point: Array<f64,Ix1>,
    origin: Array<f64,Ix1>,
    skip: usize,
    samples: usize,
    features: usize,
    gravity_points: Arc<Array<f64,Ix2>>,
    sub_gravity_points: Array<f64,Ix2>,
    sample_subsamples: Vec<usize>,
    feature_subsamples: Array<bool,Ix1>,
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

        let mut subsample_size = gravity_points.shape()[0];
        if samples > 1000 {
            subsample_size = max(min(1000, samples/10),2);
        }
        if parameters.sample_subsample.is_some() {
            subsample_size = parameters.sample_subsample.unwrap();
        }

        let sample_subsamples = Vec::new();
        let feature_subsamples = Array::from_elem(features,false);

        let mut sub_gravity_points = Array::zeros((subsample_size,features));

        assert!(parameters.scaling_factor.as_ref().map(|x| if *x == 0. {0.} else {1.}).unwrap_or(1.) != 0.);

        Pathfinder {
            point: point.clone(),
            origin: origin,
            skip: skip,
            samples: samples,
            features: features,
            gravity_points: gravity_points,
            sub_gravity_points: sub_gravity_points,
            sample_subsamples: sample_subsamples,
            feature_subsamples: feature_subsamples,
            previous_steps: VecDeque::new(),
            rng: rng,
            parameters: parameters,
        }
    }

    fn subsample(&mut self) {
        let mut sample_indecies = (0..self.samples).collect::<Vec<usize>>();
        self.rng.shuffle(&mut sample_indecies);
        sample_indecies.truncate(self.parameters.sample_subsample.unwrap_or(self.samples/20));
        self.sample_subsamples = sample_indecies;

        let mut feature_indecies = (0..self.features).collect::<Vec<usize>>();
        self.rng.shuffle(&mut feature_indecies);
        feature_indecies.truncate(self.parameters.feature_subsample.unwrap_or(self.features));
        for feature in feature_indecies {
            self.feature_subsamples[feature] = true;
        }

        for (i,subsample) in self.sample_subsamples.iter().enumerate() {
            if *subsample == self.skip {
                continue
            }
            self.sub_gravity_points.row_mut(i).assign(&self.gravity_points.row(*subsample));
        }

    }

    // fn step(&mut self) -> Array<f64,Ix1> {
    fn step(&mut self) {
        self.subsample();
        // for row_index in 0..self.sub_gravity_points.shape()[0] {
        for mut sub_point in self.sub_gravity_points.outer_iter_mut() {

            // let mut sub_point = self.sub_gravity_points.row_mut(row_index);
            // println!("{:?}",sub_point);
            let locality = self.parameters.locality.unwrap_or(4.);
            // sub_point.zip_mut_with(&self.point, |s,c| {*s -= c; sq_len_acc += s.powf(locality).abs();});
            Zip::from(&mut sub_point).and(self.point.view()).and(self.feature_subsamples.view())
                .apply(|sp,p,fs| {
                    if *fs {
                        *sp -= p;
                    };
                });
            let sq_len_acc = sub_point.iter().zip(self.feature_subsamples.iter()).fold(0.0,|acc,(x,fs)| if *fs {acc + x.powf(locality)} else {acc});
            if sq_len_acc == 0. {
                sub_point.fill(0.);
            }
            else {
                sub_point.mapv_inplace(|x| x/sq_len_acc);
            }
            // println!("{:?}",sub_point);
        }
        // println!("{:?}",self.sub_gravity_points);
        let sum: Array<f64,Ix1> = self.sub_gravity_points.sum_axis(Axis(0));
        let sum_length = sum.iter()
            .zip(self.feature_subsamples.iter())
            .map(|(x,fs)| if *fs {x.powi(2)} else {0.})
            .sum::<f64>().sqrt();

        let scaling_factor = self.parameters.scaling_factor.unwrap_or(0.1);

        Zip::from(&mut self.point).and(sum.view()).and(self.feature_subsamples.view())
            .apply(|pf,sf,fs| {
                if *fs {
                    *pf += sf / (sum_length / scaling_factor);
                }
            });

        // for (feature,shift) in self.point.iter_mut().zip(sum.iter()) {
        //     *feature += shift/(sum_length/self.parameters.scaling_factor.unwrap_or(0.1));
        // }
        // return self.point.to_owned()
        // return sum.iter().map(|x| (x/(sum_length/self.scaling_factor)).powi(2)).sum::<f64>().sqrt()
    }

    pub fn descend(&mut self) -> Array<f64,Ix1> {
        for _i in 0..50 {
            // let step = self.step();
            // println!("Step:{:?}",step);
            self.step();
            self.previous_steps.push_front(self.point.clone());
        }
        // println!("Steps:{:?}",self.previous_steps);
        let mut step_count = 50;
        loop {
            // let step = self.step();
            // println!("Step:{:?}",step);
            self.step();
            self.previous_steps.push_front(self.point.to_owned());
            self.previous_steps.pop_back();
            let smoothed_displacement = (self.previous_steps.back().unwrap() - &self.point).fold(0.,|acc,x| acc + x.powi(2)).sqrt();
            // if step_count%1000 == 0 {
            //     println!("Steps:{:?}",self.previous_steps);
            //     println!("Disp:{:?}",smoothed_displacement);
            // }
            if smoothed_displacement < self.parameters.convergence_factor.unwrap_or(1.)*self.parameters.scaling_factor.unwrap_or(0.1) {
                // eprintln!("Converged after {} steps", step_count);
                break
            }
            step_count += 1;
            if step_count > 20000000 {
                panic!("Failed to converge after 20 million steps, adjust parameters")
            }
        }
        self.point.to_owned()
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
