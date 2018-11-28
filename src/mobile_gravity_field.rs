use std::sync::Arc;
use std::collections::HashSet;
use ndarray::{Array,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,stack};
use std::f64;
use std::cmp::max;
use rayon::prelude::*;
use std::cmp::PartialOrd;
use std::cmp::Ordering;

use io::{Parameters,Distance};

use cluster::Cluster;
use single_pathfinder::Pathfinder;
use distance;
use length;


pub struct GravityField {
    samples: usize,
    features: usize,
    pub initial_positions: Array<f64,Ix2>,
    pub current_positions: Option<Arc<Array<f64,Ix2>>>,
    fuzz: Array<f64,Ix1>,
    pub clusters: Vec<Cluster>,
    parameters: Arc<Parameters>,
    distance: Distance,
}

impl GravityField {

    pub fn init(gravity_points: Array<f64,Ix2>, parameters:Arc<Parameters>) -> GravityField {

        eprintln!("Initializing:{:?}", gravity_points.shape());

        assert!(!gravity_points.iter().any(|x| x.is_nan()));
        let samples = gravity_points.shape()[0];
        let features = gravity_points.shape()[1];
        let current_positions = Some(Arc::new(gravity_points.clone()));
        let fuzz = Array::zeros(samples);

        GravityField {
            samples: samples,
            features: features,
            initial_positions: gravity_points,
            current_positions: current_positions,
            fuzz: fuzz,
            clusters: vec![],
            distance: parameters.distance.unwrap_or(Distance::Cosine),
            parameters: parameters,
        }
    }



    pub fn fuzzy_fit_mobile(&mut self) -> Array<f64,Ix2> {

        let mut pathfinders: Vec<Pathfinder> = (0..self.samples).map(|i| Pathfinder::init(i,self.samples,self.features,self.parameters.clone())).collect();

        let mut moving_points = pathfinders.len();

        let mut step_counter = 0;

        while moving_points > 0 {

            moving_points = pathfinders.len();

            eprintln!("Starting");

            let mut current_positions = self.current_positions.take().unwrap();

            let stepped_positions: Vec<Option<(Array<f64,Ix1>,f64)>> =
                pathfinders
                // .iter()
                .par_iter()
                .map(|pathfinder| {
                    pathfinder.step(&current_positions.clone())
                })
                .collect();

            eprintln!("Stepped");

            for (i,(mut position,new_position_option)) in
                Arc::get_mut(&mut current_positions)
                .unwrap()
                .outer_iter_mut()
                .zip(stepped_positions.iter())
                .enumerate() {
                    if let Some((new_position,step_length)) = new_position_option {
                        position.assign(&new_position);
                    }
                    else {
                        moving_points -= 1;
                    }
            }

            for pathfinder in pathfinders.iter_mut() {
                pathfinder.memorize_step(None,&current_positions.clone());
            }

            // for (pathfinder,step) in pathfinders.iter_mut().zip(stepped_positions) {
            //     pathfinder.memorize_step(step)
            // }

            self.current_positions = Some(current_positions.clone());

            eprintln!("Finished: {}", moving_points);

            step_counter += 1;

            if step_counter > 2000 {
                break
            }

        }

        for (i,pathfinder) in pathfinders.iter().enumerate() {
            self.fuzz[i] = pathfinder.sub_fuzz(10,self.current_positions.as_ref().unwrap().clone());
        }

        Arc::get_mut(self.current_positions.as_mut().unwrap()).unwrap().clone()


    }

    pub fn fit(&mut self) -> Array<f64,Ix2> {

        eprintln!("Starting a fuzzy fit:");

        let shared_positions = Arc::new(self.initial_positions.clone());
        let mut final_positions = Array::zeros((self.samples,self.features));

        let position_vec: Vec<(Array<f64,Ix1>,f64)> = (0..self.samples)
            // .into_iter()
            .into_par_iter()
            .map(|sample| {
                if sample % 10 == 0 {
                    eprintln!("s:{:?}", sample);
                };
                // eprintln!("{:?}",shared_positions.row(sample));
                let mut pathfinder = Pathfinder::init(sample, self.samples,self.features, self.parameters.clone());
                pathfinder.single_descend(&shared_positions)
            }).collect();

        for (i,(position,fuzz)) in position_vec.into_iter().enumerate() {
            final_positions.row_mut(i).assign(&position);
            self.fuzz[i] = fuzz * 10.
        }

        // eprintln!("{:?}",shared_positions.row(0));
        // eprintln!("{:?}",final_positions.row(0));

        self.current_positions = Some(Arc::new(final_positions.clone()));

        final_positions

    }

    pub fn fuzzy_fit_single(&mut self) -> Array<f64,Ix2> {

        eprintln!("Starting a fuzzy fit:");

        let shared_positions = Arc::new(self.initial_positions.clone());
        let mut final_positions = Array::zeros((self.samples,self.features));

        let position_vec: Vec<(Array<f64,Ix1>,(f64,f64))> = (0..self.samples)
            // .into_iter()
            .into_par_iter()
            .map(|sample| {
                if sample % 10 == 0 {
                    eprintln!("s:{:?}", sample);
                };
                // eprintln!("{:?}",shared_positions.row(sample));
                let mut pathfinder = Pathfinder::init(sample, self.samples,self.features, self.parameters.clone());
                pathfinder.fuzzy_descend(self.parameters.fuzz,shared_positions.clone())
            }).collect();

        for (i,(position,(deviation,_displacement))) in position_vec.into_iter().enumerate() {
            final_positions.row_mut(i).assign(&position);
            self.fuzz[i] = deviation
        }

        // eprintln!("{:?}",shared_positions.row(0));
        // eprintln!("{:?}",final_positions.row(0));

        self.current_positions = Some(Arc::new(final_positions.clone()));

        final_positions

    }

    pub fn fuzzy_predict(&mut self) -> Array<usize,Ix1> {

        let mut predictions = Array::zeros(self.samples);

        self.cluster_points();

        // self.merge_clusters();

        for cluster in &self.clusters {
            for point in &cluster.members {
                predictions[*point] = cluster.id;
            }
        }

        predictions
    }

    // pub fn very_fuzzy_predict(&mut self) -> Array<usize,Ix1> {
    //
    // }

    pub fn cluster_points(&mut self) {

        let final_positions = self.current_positions.as_ref().unwrap();

        let mut available_points: HashSet<usize> = (0..final_positions.shape()[0]).into_iter().collect();

        let first_cluster_candidate = self.best_cluster_candidate(Some(&available_points));

        if let Some(first_cluster_ind) = first_cluster_candidate {

            let first_cluster = Cluster::init(1,final_positions.clone(),first_cluster_ind,self.parameters.clone());

            self.clusters.push(first_cluster);

            while available_points.len() > 0 {

                let mut moved_points = vec![];

                for point_index in available_points.iter().cloned() {

                    let point = final_positions.row(point_index);

                    let displacement = self.distance.measure(self.initial_positions.row(point_index),point);

                    let mut distances_to_clusters = vec![];

                    for (i,cluster) in self.clusters.iter().enumerate() {

                        distances_to_clusters.push((i,self.distance.measure(point, cluster.center.view())));
                        // if self.parameters.distance.unwrap_or(Distance::Cosine).measure(point,cluster.center.view()) < (cluster.radius + self.fuzz[point_index]) {
                        // // if distance(point,cluster.center.view()) < self.parameters.scaling_factor.unwrap_or(0.1) * self.parameters.convergence_factor.unwrap_or(5.){
                        //     cluster.merge_point(point,point_index);
                        //     moved_points.push(point_index);
                        //     break
                        // }
                    }

                    let best_cluster_option = distances_to_clusters.iter().min_by(|x,y| x.1.partial_cmp(&y.1).unwrap_or(Ordering::Greater));

                    // eprintln!("PP:{:?}",point);
                    // eprintln!("FF:{:?}",self.fuzz[point_index]);
                    // eprintln!("BCO:{:?}", best_cluster_option);
                    // eprintln!("BCR:{:?}",self.clusters[best_cluster_option.unwrap().0].radius);

                    if let Some((best_cluster_index,distance_to_cluster)) = best_cluster_option {
                        let best_cluster: &mut Cluster = &mut self.clusters[*best_cluster_index];
                        // eprintln!("Try");
                        // eprintln!("ID:{:?}",best_cluster.id);
                        // eprintln!("FF:{:?}",self.fuzz[point_index]);
                        // eprintln!("CM:{:?}",distance_to_cluster);
                        // eprintln!("DS:{:?}",displacement);
                        // eprintln!("CC:{:?}",best_cluster.center);
                        // eprintln!("PC:{:?}",point);

                        // if *distance_to_cluster < (best_cluster.radius + self.fuzz[point_index]) * 2. {
                        if *distance_to_cluster < (best_cluster.radius + self.fuzz[point_index]) {
                            moved_points.push(point_index);
                            best_cluster.merge_point(point,point_index);
                            // eprintln!("ID:{:?}",best_cluster.id);
                            // eprintln!("FF:{:?}",self.fuzz[point_index]);
                            // eprintln!("CR:{:?}",best_cluster.radius);
                            // eprintln!("CM:{:?}",distance_to_cluster);
                            // eprintln!("DS:{:?}",displacement);
                            // eprintln!("CC:{:?}",best_cluster.center);
                            // eprintln!("PC:{:?}",point);
                        }
                    }

                }


                for point in &moved_points {
                    available_points.remove(point);
                }

                if moved_points.len() < 1 {
                    if let Some(new_cluster_point) = self.best_cluster_candidate(Some(&available_points)) {
                        available_points.remove(&new_cluster_point);
                        let new_cluster = Cluster::init(self.clusters.len()+1, final_positions.clone(), new_cluster_point,self.parameters.clone());
                        self.clusters.push(new_cluster);
                    }
                    else {
                        break
                    }
                }

                eprintln!("Unclustered:{:?}",available_points.len());
                eprintln!("Clusters:{:?}",self.clusters.len());

            }


        }

        eprintln!("Coarse clusters: {:?}", self.clusters.len())

    }

    pub fn merge_clusters(&mut self) {

        let mut clusters = self.clusters.clone();
        let final_positions = self.current_positions.as_ref().unwrap();


        loop {

            let mut merge_candidates: Option<(usize,usize)> = None;

            'i_loop: for i in 0..clusters.len() {


                let c1 = &clusters[i];

                'j_loop: for j in 0..clusters.len() {
                    if i != j {

                        let c2 = &clusters[j];

                        // if self.parameters.distance.unwrap_or(Distance::Cosine).measure(c1.center.view(),c2.center.view()) < (c1.radius + c2.radius)*2. {
                        //     eprintln!("Failed");
                        //     eprintln!("C1:{:?}",c1.center);
                        //     eprintln!("C2:{:?}",c2.center);
                        //     eprintln!("R1:{:?}",c1.radius);
                        //     eprintln!("R2:{:?}",c2.radius);
                        //     eprintln!("Merging:{:?}",merge_candidates);
                        //     eprintln!("Distance:{:?}",self.parameters.distance.unwrap_or(Distance::Cosine).measure(c1.center.view(),c2.center.view()));
                        // }

                        if self.distance.measure(c1.center.view(),c2.center.view()) < (c1.radius.sqrt() + c2.radius.sqrt()).powi(2) {
                            merge_candidates = Some((i,j));
                            // eprintln!("C1:{:?}",c1.center);
                            // eprintln!("C2:{:?}",c2.center);
                            // eprintln!("R1:{:?}",c1.radius);
                            // eprintln!("R2:{:?}",c2.radius);
                            // eprintln!("W1:{:?}",c1.weight);
                            // eprintln!("W2:{:?}",c2.weight);
                            // eprintln!("Merging:{:?}",merge_candidates);
                            break 'i_loop;
                        }
                    }
                }

            }

            if let Some((c1i,c2i)) = merge_candidates {
                let new_cluster = clusters[c1i].merge_cluster(&clusters[c2i]);
                // eprintln!("N:{:?}",new_cluster.center);
                // eprintln!("N:{:?}",new_cluster.radius);
                clusters[c1i] = new_cluster;
                clusters.remove(c2i);
            }
            else {
                break
            }

        }

        eprintln!("Merged Clusters: {:?}",clusters.len());

        self.clusters = clusters;
    }

    pub fn best_cluster_candidate(&self, from_points: Option<&HashSet<usize>>) -> Option<usize> {

        let mut best_candidate = (0,f64::MAX);
        let mut any_candidate = false;
        let mut candidate_points = HashSet::with_capacity(0);
        if from_points.is_some() {
            candidate_points = from_points.unwrap().clone();
        }
        else {
            candidate_points = (0..self.samples).into_iter().collect();
        };
        for point in candidate_points.iter() {
            if self.fuzz[*point] < best_candidate.1 {
                best_candidate = (*point,self.fuzz[*point]);
                any_candidate = true;
            }
        }

        if any_candidate {
            Some(best_candidate.0)
        }
        else {
            None
        }
    }


    // pub fn write_clusters(&self) -> Array<f64,Ix2> {
    //     let array = Array::zeros((self.clusters.len(),self.gravity_points.shape()[1]));
    //     for cluster in kk
    //     array
    // }
}
