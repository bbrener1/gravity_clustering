use ndarray::{Array,ArrayView,Ix1,Ix2,Axis};
use std::f64;
use std::sync::Arc;
use std::collections::VecDeque;
use rand::{Rng,ThreadRng,thread_rng};
use rand::seq::sample_indices;
use std::cmp::{min,max};
use io::Parameters;
// use ndarray_parallel::prelude::*;
use length;
use io::Distance;

#[derive(Debug)]
pub struct Pathfinder {
    pub id: usize,
    samples: usize,
    sample_indecies: Vec<usize>,
    features: usize,
    // points: Arc<Array<f64,Ix2>>,
    sample_subsample:usize,
    sample_subsamples: Vec<usize>,
    previous_steps: VecDeque<Array<f64,Ix1>>,
    distance: Distance,
    smoothing: usize,
    convergence: f64,
    converged: bool,
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
            sample_subsample: parameters.sample_subsample.unwrap_or(samples),
            sample_subsamples: sample_subsamples,
            previous_steps: VecDeque::with_capacity(51),
            smoothing: parameters.smoothing.unwrap_or(5),
            distance: parameters.distance.unwrap_or(Distance::Cosine),
            convergence: parameters.convergence_factor.unwrap_or(1.),
            converged: false,
        }

    }

    fn point(&self,points:&Arc<Array<f64,Ix2>>) -> Array<f64,Ix1> {
        points.row(self.id).to_owned()
    }

    fn point_view<'a>(&'a self,points:&'a Arc<Array<f64,Ix2>>) -> ArrayView<'a,f64,Ix1> {
        points.row(self.id)
    }

    pub fn memorize_step(&mut self,point_option: Option<Array<f64,Ix1>>,points:&Arc<Array<f64,Ix2>>) {
        if !self.converged(points) {
            // eprintln!("PO:{:?}", point_option);
            if let Some(point) = point_option {
                self.previous_steps.push_front(point);
                if self.previous_steps.len() > 50 {
                    self.previous_steps.pop_back();
                }
            }
            else {
                // let current_point = self.point(points);
                // self.previous_steps.push_front(current_point);
                // if self.previous_steps.len() > 50 {
                //     self.previous_steps.pop_back();
                // }
            }
        }
        // eprintln!("ST:{:?}", self.previous_steps);
    }

    fn subsampled_nearest(&self,points:Arc<Array<f64,Ix2>>) -> Option<(Array<f64,Ix1>,f64)> {

        // eprintln!("SS:{:?}",self.sample_subsamples);

        let mut jump_point = None;

        let mut maximum_jump_distance = f64::MAX;

        for mut sub_point_index in sample_indices(&mut thread_rng(), self.samples, self.sample_subsample) {
            // eprintln!("P:{:?}",self.point.view());
            // eprintln!("S1:{:?}",self.points.row(*sub_point_index).view());
            // eprintln!("D:{:?}",distance(self.point.view(),self.points.row(*sub_point_index).view()));
            let point_distance = self.distance.measure(self.point_view(&points),points.row(sub_point_index).view());
            if (point_distance < maximum_jump_distance) && point_distance > 0. {
                maximum_jump_distance = point_distance;
                jump_point = Some(sub_point_index);
            }
        }

        jump_point.map(|x| (points.row(x).to_owned(),maximum_jump_distance))

    }

    fn subsampled_nearest_n(&self,n: usize,points:Arc<Array<f64,Ix2>>) -> Vec<(Array<f64,Ix1>,f64)> {

        self.subsampled_nearest_n_to(self.point_view(&points),n,points.clone())

    }

    fn subsampled_nearest_n_to(&self,center: ArrayView<f64,Ix1>,n: usize,points:Arc<Array<f64,Ix2>>) -> Vec<(Array<f64,Ix1>,f64)> {

        let mut sub_points: Vec<(Array<f64,Ix1>,f64)> = Vec::with_capacity(n+1);

        let sample_subsamples = sample_indices(&mut thread_rng(), self.samples, self.sample_subsample);

        if let Some(first_index) = sample_subsamples.get(0) {
            let first_subsample = points.row(*first_index);
            sub_points.push((first_subsample.to_owned(),self.distance.measure(center, first_subsample.view())));
        }

        for sub_point_index in sample_subsamples {

            let sub_point = points.row(sub_point_index);
            let sub_point_distance = self.distance.measure(center,sub_point.view());

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



    pub fn step(&self,points:&Arc<Array<f64,Ix2>>) -> Option<(Array<f64,Ix1>,f64)> {

        self.step_from(self.point_view(&points),points)

    }

    pub fn step_from(&self,point: ArrayView<f64,Ix1>,points:&Arc<Array<f64,Ix2>>) -> Option<(Array<f64,Ix1>,f64)> {

        if self.converged {
            return None
        }

        // eprintln!("STEP");
        // eprintln!("P:{:?}",point);

        let smoothing = self.smoothing;

        let mut jump_point = Array::zeros(self.features);
        // let mut total_distance = 0.;
        let mut bag_counter = 0;

        for (bagged_point,_jump_distance) in self.subsampled_nearest_n_to(point,smoothing,points.clone()) {
            jump_point += &bagged_point;
            bag_counter += 1;
        };


        if bag_counter > 0 {

            jump_point /= bag_counter as f64;

            // eprintln!("J:{:?}",jump_point);

            jump_point = (jump_point * 0.3) + (&point * 0.7);

            let jump_distance = self.distance.measure(jump_point.view(),self.point_view(&points));

            // eprintln!("J:{:?}",jump_point);

            return Some((jump_point,jump_distance))
        }

        None
    }

    pub fn single_descend(&mut self,points:&Arc<Array<f64,Ix2>>) -> Array<f64,Ix1> {

        let mut point = self.point(points);
        // let mut step_counter = 0;

        while let Some((step,distance)) = self.step_from(point.view(),points) {
            point = step;
            self.memorize_step(Some(point.clone()),points);
            // step_counter += 1;
            // if step_counter%10 == 0 {
                // eprintln!("S:{:?}",step_counter);
            // }
        }

        // eprintln!("Steps:{:?}", self.previous_steps);

        self.previous_steps.clear();
        self.converged = false;

        point

    }

    pub fn fuzzy_descend(&mut self,fuzz:usize,points:Arc<Array<f64,Ix2>>) -> (Array<f64,Ix1>,(f64,f64)) {
    // pub fn fuzzy_descend(&mut self,fuzz:usize,points:&Array<f64,Ix2>) -> (Array<f64,Ix1>,(f64,f64)) {

        let mut final_points: Array<f64,Ix2> = Array::zeros((fuzz,self.features));

        for i in 0..fuzz {
            final_points.row_mut(i).assign(&self.single_descend(&points));
        }

        let average_point = final_points.sum_axis(Axis(0))/fuzz as f64;

        // eprintln!("Average:{:?}",average_point);

        let mut av_deviation = 0.;

        for mut final_point in final_points.outer_iter_mut() {
            av_deviation += self.distance.measure(average_point.view(),final_point.view()) / fuzz as f64;
            // final_point.scaled_add(-1.,&average_point);
            // av_deviation += length(final_point.view())/fuzz as f64;
        }

        eprintln!("Deviation:{:?}",av_deviation);

        let displacement = length((&self.point(&points) - &average_point).view());

        eprintln!("Displacement:{:?}",displacement);

        // eprintln!("Steps:{:?}", self.previous_steps);

        (average_point,(av_deviation,displacement))

    }

    fn converged(&mut self,points:&Arc<Array<f64,Ix2>>) -> bool {
        if self.previous_steps.len() < 50 {
            return false
        }
        if self.converged {
            return true
        }
        let distant_point = self.previous_steps.back().unwrap().view();
        let previous_point = self.previous_steps.front().unwrap().view();
        let current_point = self.point(&points);
        let short_displacement = self.distance.measure(previous_point,current_point.view());
        let long_displacement = self.distance.measure(distant_point,current_point.view());
        // eprintln!("SD:{:?}",short_displacement);
        // eprintln!("LD:{:?}",long_displacement);
        if long_displacement < (short_displacement * self.convergence) {
            self.converged = true;
            return true
        }
        else {
            return false
        }
    }

    pub fn fuzz(&self) -> f64 {
        let distant_point = self.previous_steps.back().unwrap().view();
        let previous_point = self.previous_steps.front().unwrap().view();
        self.distance.measure(distant_point,previous_point)
    }

    pub fn sub_fuzz(&self,fuzz: usize,points:Arc<Array<f64,Ix2>>) -> f64 {
        let mut acc = 0.;
        for (point,_jump_distance) in self.subsampled_nearest_n(fuzz, points.clone()) {
            acc += self.distance.measure(self.point_view(&points), point.view()) / fuzz as f64;
        }
        acc
    }

}

fn test(){
    fn is_sync<T:Sync>(){}
    is_sync::<Pathfinder>();
}
