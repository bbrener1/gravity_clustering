use std::sync::Arc;
use std::collections::HashSet;
use ndarray::{Array,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,stack};
use std::f64;
use std::cmp::max;
use rayon::prelude::*;

use io::Parameters;

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
}

impl GravityField {

    pub fn init(gravity_points: Array<f64,Ix2>, parameters:Arc<Parameters>) -> GravityField {
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
            parameters: parameters,
        }
    }



    pub fn fuzzy_fit(&mut self) -> Array<f64,Ix2> {

        let mut pathfinders: Vec<Pathfinder> = (0..self.samples).map(|i| Pathfinder::init(i,self.current_positions.as_ref().unwrap().clone(),self.parameters.clone())).collect();

        let mut moving_points = pathfinders.len();

        while moving_points > 0 {

            moving_points = pathfinders.len();

            eprintln!("Starting");

            let stepped_positions: Vec<Option<(Array<f64,Ix1>,f64)>> =
                pathfinders
                .iter()
                // .par_iter()
                .map(|pathfinder| {
                    pathfinder.step()
                })
                .collect();

            eprintln!("Stepped");

            let mut final_positions: Arc<Array<f64,Ix2>> = self.current_positions.take().unwrap();

            for (i,(mut position,new_position_option)) in
                Arc::make_mut(&mut final_positions)
                .outer_iter_mut()
                .zip(stepped_positions.iter())
                .enumerate() {
                    if let Some((new_position,fuzz)) = new_position_option {
                        position.assign(new_position);
                        self.fuzz[i] = *fuzz;
                    }
                    else {
                        moving_points -= 1;
                    }
            }

            for pathfinder in pathfinders.iter_mut() {
                pathfinder.set_points(final_positions.clone());
            }

            self.current_positions = Some(final_positions);

            eprintln!("Finished: {}", moving_points);

        }

        Arc::get_mut(self.current_positions.as_mut().unwrap()).unwrap().clone()


    }

    pub fn fuzzy_predict(&mut self) -> Array<usize,Ix1> {

        let mut predictions = Array::zeros(self.current_positions.as_ref().unwrap().shape()[0]);

        self.cluster_points();

        self.merge_clusters();

        for cluster in &self.clusters {
            for point in &cluster.members {
                predictions[*point] = cluster.id;
            }
        }

        predictions
    }

    pub fn cluster_points(&mut self) {

        let final_positions = self.current_positions.as_ref().unwrap();

        let mut available_points: HashSet<usize> = (0..final_positions.shape()[0]).into_iter().collect();

        let first_cluster_candidate = self.best_cluster_candidate(Some(&available_points));

        if let Some(first_cluster_ind) = first_cluster_candidate {

            let first_cluster = Cluster::init(1,final_positions.row(first_cluster_ind),first_cluster_ind);

            self.clusters.push(first_cluster);

            while available_points.len() > 0 {

                let mut moved_points = vec![];

                for point_index in available_points.iter().cloned() {

                    let point = final_positions.row(point_index);

                    for cluster in self.clusters.iter_mut() {
                        if distance(point,cluster.center.view()) < cluster.radius(final_positions,&self.fuzz) * 10. {
                        // if distance(point,cluster.center.view()) < self.parameters.scaling_factor.unwrap_or(0.1) * self.parameters.convergence_factor.unwrap_or(5.){
                            // eprintln!("ID:{:?}",cluster.id);
                            // eprintln!("CM:{:?}",distance(point,cluster.center.view()));
                            // eprintln!("CC:{:?}",cluster.center);
                            // eprintln!("PC:{:?}",point);
                            cluster.merge_point(point,point_index);
                            moved_points.push(point_index);
                            break
                        }
                    }
                }

                for point in &moved_points {
                    available_points.remove(point);
                }

                if moved_points.len() < 1 {
                    if let Some(new_cluster_point) = self.best_cluster_candidate(Some(&available_points)) {
                        available_points.remove(&new_cluster_point);
                        let new_cluster = Cluster::init(self.clusters.len()+1, final_positions.row(new_cluster_point), new_cluster_point);
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

            let mut new_cluster: Option<Cluster> = None;
            let mut removed_clusters: Option<(usize,usize)> = None;

            'i_loop: for i in 0..clusters.len() {


                let c1 = &clusters[i];
                let c1r = c1.radius(&final_positions,&self.fuzz);

                'j_loop: for j in 0..clusters.len() {
                    if i != j {

                        let c2 = &clusters[j];

                        let c2r = c2.radius(&final_positions,&self.fuzz);

                        if distance(c1.center.view(),c2.center.view()) < (c1r + c2r) * 3. {
                            new_cluster = Some(c1.merge_cluster(c2));
                            removed_clusters = Some((i,j));
                            eprintln!("Merged:{:?}",removed_clusters);
                            break 'i_loop;
                        }
                    }
                }

            }

            if let Some((c1r,c2r)) = removed_clusters {
                clusters[c1r] = new_cluster.unwrap();
                clusters.remove(c2r);
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

#[derive(Clone)]
pub struct Cluster {
    pub id: usize,
    weight: usize,
    center: Array<f64,Ix1>,
    members: Vec<usize>,
    // array: Array<f64,Ix2>
}

impl Cluster {
    pub fn init(id: usize, point: ArrayView<f64,Ix1>, point_id: usize) -> Cluster {
        let mut array = Array::zeros((1,point.shape()[0]));
        array.row_mut(0).assign(&point);
        Cluster {
            id: id,
            weight: 1,
            center: point.to_owned(),
            members: vec![point_id],
            // array: array,

        }
    }

    pub fn merge_cluster(&self,cluster: &Cluster) -> Cluster {
        let new_weight = self.weight + cluster.weight;
        let new_center = ((&self.center * self.weight as f64) + (&cluster.center * cluster.weight as f64)) / (new_weight) as f64;
        let new_members = [self.members.iter(),cluster.members.iter()].iter().flat_map(|x| x.clone()).cloned().collect();
        Cluster {
            id : self.id,
            weight: new_weight,
            center: new_center,
            members: new_members,
        }
    }

    pub fn merge_point(&mut self ,point: ArrayView<f64,Ix1>, point_id:usize) -> usize {

        self.center *= self.weight as f64/(self.weight as f64 + 1.);
        self.center.scaled_add(1./(self.weight as f64 + 1.), &point);
        self.weight += 1;
        self.members.push(point_id);
        // self.array = stack!(Axis(0),self.array, point.to_owned().insert_axis(Axis(1)).t());
        self.id
    }

    pub fn radius(&self,array:&Arc<Array<f64,Ix2>>,fuzz:&Array<f64,Ix1>) -> f64 {
        if self.members.len() == 1 {
            return fuzz[self.members[0]];
        }
        let radius = self.members.iter().map(|x| length((&array.row(*x) - &self.center).view())).sum::<f64>() / self.weight as f64;
        // eprintln!("R:{:?}",radius);
        radius
    }

}
