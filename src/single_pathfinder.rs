use ndarray::{Array,ArrayView,Ix1,Ix2,Axis};
use std::f64;
use std::sync::Arc;
use std::collections::VecDeque;
use rand::{Rng,ThreadRng,thread_rng};
use rand::seq::sample_indices;
use std::cmp::{min,max};
use io::Parameters;
use ndarray_parallel::prelude::*;
use distance;
use length;

#[derive(Debug)]
pub struct Pathfinder {
    pub id: usize,
    samples: usize,
    sample_indecies: Vec<usize>,
    features: usize,
    points: Arc<Array<f64,Ix2>>,
    sample_subsamples: Vec<usize>,
    previous_steps: VecDeque<Array<f64,Ix1>>,
    parameters: Arc<Parameters>,
}

impl Pathfinder {
    // pub fn init(origin: ArrayView<'a,f64,Ix1>, gravity_points: Arc<Array<f64,Ix2>>, skip: usize, scaling_factor:Option<f64>, subsample_arg: Option<usize>, convergence_arg: Option<f64>,locality: Option<f64>) -> Pathfinder {
    pub fn init(id: usize, points: Arc<Array<f64,Ix2>>, parameters: Arc<Parameters>) -> Pathfinder {

        let samples = points.shape()[0];
        let features = points.shape()[1];

        let mut sample_indecies: Vec<usize> = (0..samples).collect();
        sample_indecies.remove(id);

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

        eprintln!("INITIALIZED");

        Pathfinder {
            id: id,
            samples: samples,
            sample_indecies: sample_indecies,
            features: features,
            points: points,
            sample_subsamples: sample_subsamples,
            previous_steps: VecDeque::with_capacity(50),
            parameters: parameters,
        }

    }

    fn point(&self) -> Array<f64,Ix1> {
        self.points.row(self.id).to_owned()
    }

    fn point_view<'a>(&'a self) -> ArrayView<'a,f64,Ix1> {
        self.points.row(self.id)
    }

    pub fn set_points(&mut self,new_points:Arc<Array<f64,Ix2>>) {
        if !self.converged() {
            let current_point = self.point();
            self.previous_steps.push_front(current_point);
            if self.previous_steps.len() > 50 {
                self.previous_steps.pop_back();
            }
            self.points = new_points;
        }
    }

    fn subsampled_nearest(&self) -> Option<(Array<f64,Ix1>,f64)> {

        // eprintln!("SS:{:?}",self.sample_subsamples);

        let mut jump_point = None;

        let mut maximum_jump_distance = f64::MAX;

        for mut sub_point_index in sample_indices(&mut thread_rng(), self.samples, self.parameters.sample_subsample.unwrap_or(self.samples)) {
            // eprintln!("P:{:?}",self.point.view());
            // eprintln!("S1:{:?}",self.points.row(*sub_point_index).view());
            // eprintln!("D:{:?}",distance(self.point.view(),self.points.row(*sub_point_index).view()));
            let point_distance = distance(self.point_view(),self.points.row(sub_point_index).view());
            if (point_distance < maximum_jump_distance) && point_distance > 0. {
                maximum_jump_distance = point_distance;
                jump_point = Some(sub_point_index);
            }
        }

        jump_point.map(|x| (self.points.row(x).to_owned(),maximum_jump_distance))

    }

    fn subsampled_nearest_n(&self,n: usize) -> Vec<(Array<f64,Ix1>,f64)> {

        let mut sub_points: Vec<(Array<f64,Ix1>,f64)> = Vec::with_capacity(n+1);

        let sample_subsamples = sample_indices(&mut thread_rng(), self.samples, self.parameters.sample_subsample.unwrap_or(self.samples));

        if let Some(first_index) = sample_subsamples.get(0) {
            let first_subsample = self.points.row(*first_index);
            sub_points.push((first_subsample.to_owned(),distance(self.point_view(), first_subsample.view())));
        }

        for sub_point_index in sample_subsamples {

            let sub_point = self.points.row(sub_point_index);
            let sub_point_distance = distance(self.point_view(),sub_point.view());

            let mut insert_index = None;

            for (i,(previous_point,previous_distance)) in sub_points.iter().enumerate() {
                if sub_point_distance < *previous_distance {
                    insert_index = Some(i);
                    break
                }
            }

            if let Some(insert) = insert_index {
                sub_points.insert(insert,(sub_point.to_owned(),sub_point_distance));
            }

            sub_points.truncate(n+1);

        }

        sub_points

    }


    pub fn step(&self) -> Option<(Array<f64,Ix1>,f64)> {

        if self.converged() {
            return None
        }

        // eprintln!("STEP");

        let smoothing = self.parameters.smoothing.unwrap_or(5);

        let mut jump_point = Array::zeros(self.features);
        // let mut total_distance = 0.;
        let mut bag_counter = 0;

        for (bagged_point,_jump_distance) in self.subsampled_nearest_n(smoothing) {
            jump_point += &bagged_point;
            bag_counter += 1;
        };


        if bag_counter > 0 {

            jump_point /= bag_counter as f64;
            // let mean_distance = total_distance / bag_counter as f64;

            jump_point = (jump_point * 0.3) + (&self.point() * 0.7);

            // eprintln!("J:{:?}",jump_point);

            let jump_distance = distance(jump_point.view(),self.point_view());

            // if jump_distance == 0. {
            //     // eprintln!("P:{:?}",self.point);
            //     // eprintln!("J:{:?}",jump_point);
            //     // eprintln!("MD:{:?}",mean_distance);
            //     // eprintln!("JD:{:?}",jump_distance);
            //     panic!()
            // }

            return Some((jump_point,jump_distance))
        }

        None
    }

    fn converged(&self) -> bool {
        if self.previous_steps.len() < 50 {
            return false
        }
        let distant_point = self.previous_steps.back().unwrap().view();
        let previous_point = self.previous_steps.front().unwrap().view();
        let current_point = self.point_view();
        let short_displacement = distance(previous_point,current_point);
        let long_displacement = distance(distant_point,current_point);
        long_displacement < (short_displacement * self.parameters.convergence_factor.unwrap_or(3.))
    }

    fn fuzz(&self) -> f64 {
        let distant_point = self.previous_steps.back().unwrap().view();
        let previous_point = self.previous_steps.front().unwrap().view();
        distance(distant_point,previous_point)
    }

}

fn test(){
    fn is_sync<T:Sync>(){}
    is_sync::<Pathfinder>();
}
