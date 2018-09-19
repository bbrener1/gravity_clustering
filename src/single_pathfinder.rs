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
    // points: Arc<Array<f64,Ix2>>,
    sample_subsamples: Vec<usize>,
    previous_steps: VecDeque<Array<f64,Ix1>>,
    parameters: Arc<Parameters>,
}

impl Pathfinder {
    // pub fn init(origin: ArrayView<'a,f64,Ix1>, gravity_points: Arc<Array<f64,Ix2>>, skip: usize, scaling_factor:Option<f64>, subsample_arg: Option<usize>, convergence_arg: Option<f64>,locality: Option<f64>) -> Pathfinder {
    pub fn init(id: usize, samples: usize,features: usize, parameters: Arc<Parameters>) -> Pathfinder {

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

        // eprintln!("INITIALIZED");

        Pathfinder {
            id: id,
            samples: samples,
            sample_indecies: sample_indecies,
            features: features,
            // points: points,
            sample_subsamples: sample_subsamples,
            previous_steps: VecDeque::with_capacity(51),
            parameters: parameters,
        }

    }

    fn point(&self,points:Arc<Array<f64,Ix2>>) -> Array<f64,Ix1> {
        points.row(self.id).to_owned()
    }

    fn point_view<'a>(&'a self,points:&'a Arc<Array<f64,Ix2>>) -> ArrayView<'a,f64,Ix1> {
        points.row(self.id)
    }

    pub fn memorize_step(&mut self,point_option: Option<Array<f64,Ix1>>,points:Arc<Array<f64,Ix2>>) {
        if !self.converged(points.clone()) {
            if let Some(point) = point_option {
                self.previous_steps.push_front(point);
                if self.previous_steps.len() > 50 {
                    self.previous_steps.pop_back();
                }
            }
            else {
                let current_point = self.point(points);
                self.previous_steps.push_front(current_point);
                if self.previous_steps.len() > 50 {
                    self.previous_steps.pop_back();
                }
            }
        }
    }

    fn subsampled_nearest(&self,points:Arc<Array<f64,Ix2>>) -> Option<(Array<f64,Ix1>,f64)> {

        // eprintln!("SS:{:?}",self.sample_subsamples);

        let mut jump_point = None;

        let mut maximum_jump_distance = f64::MAX;

        for mut sub_point_index in sample_indices(&mut thread_rng(), self.samples, self.parameters.sample_subsample.unwrap_or(self.samples)) {
            // eprintln!("P:{:?}",self.point.view());
            // eprintln!("S1:{:?}",self.points.row(*sub_point_index).view());
            // eprintln!("D:{:?}",distance(self.point.view(),self.points.row(*sub_point_index).view()));
            let point_distance = distance(self.point_view(&points),points.row(sub_point_index).view());
            if (point_distance < maximum_jump_distance) && point_distance > 0. {
                maximum_jump_distance = point_distance;
                jump_point = Some(sub_point_index);
            }
        }

        jump_point.map(|x| (points.row(x).to_owned(),maximum_jump_distance))

    }

    fn subsampled_nearest_n(&self,n: usize,points:Arc<Array<f64,Ix2>>) -> Vec<(Array<f64,Ix1>,f64)> {

        self.subsampled_nearest_n_to(self.point_view(&points),n,points)

    }

    fn subsampled_nearest_n_to(&self,center: ArrayView<f64,Ix1>,n: usize,points:Arc<Array<f64,Ix2>>) -> Vec<(Array<f64,Ix1>,f64)> {

        let mut sub_points: Vec<(Array<f64,Ix1>,f64)> = Vec::with_capacity(n+1);

        let sample_subsamples = sample_indices(&mut thread_rng(), self.samples, self.parameters.sample_subsample.unwrap_or(self.samples));

        if let Some(first_index) = sample_subsamples.get(0) {
            let first_subsample = points.row(*first_index);
            sub_points.push((first_subsample.to_owned(),distance(center, first_subsample.view())));
        }

        for sub_point_index in sample_subsamples {

            let sub_point = points.row(sub_point_index);
            let sub_point_distance = distance(center,sub_point.view());

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



    pub fn step(&self,points:Arc<Array<f64,Ix2>>) -> Option<(Array<f64,Ix1>,f64)> {

        self.step_from(self.point_view(&points),points)

    }

    pub fn step_from(&self,point: ArrayView<f64,Ix1>,points:Arc<Array<f64,Ix2>>) -> Option<(Array<f64,Ix1>,f64)> {

        if self.converged(points.clone()) {
            return None
        }

        // eprintln!("STEP");

        let smoothing = self.parameters.smoothing.unwrap_or(5);

        let mut jump_point = Array::zeros(self.features);
        // let mut total_distance = 0.;
        let mut bag_counter = 0;

        for (bagged_point,_jump_distance) in self.subsampled_nearest_n(smoothing,points.clone()) {
            jump_point += &bagged_point;
            bag_counter += 1;
        };


        if bag_counter > 0 {

            jump_point /= bag_counter as f64;

            jump_point = (jump_point * 0.1) + (&self.point(points.clone()) * 0.9);

            let jump_distance = distance(jump_point.view(),self.point_view(&points));

            return Some((jump_point,jump_distance))
        }

        None
    }

    pub fn single_descend(&mut self,points:Arc<Array<f64,Ix2>>) -> Array<f64,Ix1> {

        let mut point = self.point(points);

        while let Some((step,distance)) = self.step_from(point.view(),points) {
            point = step;
            self.memorize_step(Some(point.clone()),points);
        }

        self.previous_steps.clear();

        point

    }

    pub fn fuzzy_descend(&self,fuzz:usize,points:Arc<Array<f64,Ix2>>) -> (Array<f64,Ix1>,(f64,f64)) {

        let mut final_points: Array<f64,Ix2> = Array::zeros((fuzz,self.features));

        for i in 0..fuzz {
            final_points.row_mut(i).assign(&self.single_descend(points));
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

    fn converged(&self,points:Arc<Array<f64,Ix2>>) -> bool {
        if self.previous_steps.len() < 50 {
            return false
        }
        let distant_point = self.previous_steps.back().unwrap().view();
        let previous_point = self.previous_steps.front().unwrap().view();
        let current_point = self.point_view(&points);
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
