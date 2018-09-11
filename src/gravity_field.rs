use std::sync::Arc;
use std::collections::HashSet;
use ndarray::{Array,Ix1,Ix2,Zip,Axis,ArrayView,stack};
use std::f64;
use std::cmp::max;
use rayon::prelude::*;

use io::Parameters;

use nearest_pathfinder::Pathfinder;
use distance;
use length;

pub struct GravityField {
    pub gravity_points: Arc<Array<f64,Ix2>>,
    pub final_positions: Option<Array<f64,Ix2>>,
    pub fuzz: Vec<(f64,f64)>,
    pub clusters: Vec<Cluster>,
    parameters: Arc<Parameters>,
}

impl GravityField {

    pub fn init(gravity_points: Array<f64,Ix2>, parameters:Arc<Parameters>) -> GravityField {
        assert!(!gravity_points.iter().any(|x| x.is_nan()));
        GravityField {
            gravity_points: Arc::new(gravity_points),
            final_positions: None,
            fuzz: vec![],
            clusters: vec![],
            parameters: parameters,
        }
    }


    pub fn fit(&mut self) -> &Option<Array<f64,Ix2>> {

        let mut final_positions: Array<f64,Ix2> = Array::zeros((self.gravity_points.shape()[0],self.gravity_points.shape()[1]));

        let solutions: Vec<Array<f64,Ix1>> = (0..self.gravity_points.shape()[0])
            .into_par_iter()
            .map(|pi| {
                // eprintln!("Fitting sample {}",pi);
                let mut pathfinder = Pathfinder::init(self.gravity_points.row(pi).to_owned(),self.gravity_points.clone(),pi,self.parameters.clone());
                pathfinder.descend()
            }
        ).collect();

        for (solution, mut final_store) in solutions.iter().zip(final_positions.outer_iter_mut()) {
            final_store.assign(solution);
        }

        self.final_positions = Some(final_positions);

        &self.final_positions
    }

    pub fn predict(&mut self) -> Array<usize,Ix1> {

        let final_positions = self.final_positions.as_ref().unwrap();

        let merge_distance = self.parameters.merge_distance.unwrap_or(self.parameters.scaling_factor.unwrap_or(self.gravity_points.shape()[1] as f64 / 5.) * 10.) ;

        let mut cluster_labels = Array::zeros(final_positions.shape()[0]);

        for (point, mut cluster_label) in final_positions.outer_iter().zip(cluster_labels.iter_mut()) {
            for (key_point,existing_cluster) in self.clusters.iter().map(|cluster| (&cluster.center,cluster.id)) {
                if distance(point, key_point.view()) < merge_distance {
                    *cluster_label = existing_cluster
                }
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

    pub fn fuzzy_fit(&mut self) {

        let fuzzy_positions: Vec<(Array<f64,Ix1>,(f64,f64))> =
            (0..self.gravity_points.shape()[0])
            .into_par_iter()
            // .into_iter()
            .map(|pi| {
                eprintln!("Fitting sample {}",pi);
                let mut pathfinder = Pathfinder::init(self.gravity_points.row(pi).to_owned(),self.gravity_points.clone(),pi,self.parameters.clone());
                pathfinder.fuzzy_descend(10)
            })
            .collect();

        let mut final_positions: Array<f64,Ix2> = Array::zeros((self.gravity_points.shape()[0],self.gravity_points.shape()[1]));
        let mut fuzz_vec = Vec::with_capacity(fuzzy_positions.len());

        for (i,(fuzzy_position,fuzz)) in fuzzy_positions.into_iter().enumerate() {
            final_positions.row_mut(i).assign(&fuzzy_position);
            fuzz_vec.push(fuzz)
        }

        self.final_positions = Some(final_positions);
        self.fuzz = fuzz_vec;
        eprintln!("Convergent Points: {:?}",self.convergent_points().len());

    }

    pub fn fuzzy_predict(&mut self) -> Array<usize,Ix1> {

        let mut predictions = Array::zeros(self.gravity_points.shape()[0]);

        self.cluster_points();

        // self.merge_clusters();

        for cluster in &self.clusters {
            for point in &cluster.members {
                predictions[*point] = cluster.id;
            }
        }

        predictions
    }

    pub fn cluster_points(&mut self) {

        let final_positions = self.final_positions.as_ref().unwrap();

        let mut available_points: HashSet<usize> = self.convergent_points().into_iter().collect();

        let first_cluster_candidate = self.best_cluster_candidate(Some(&available_points));

        if let Some(first_cluster_ind) = first_cluster_candidate {

            let first_cluster = Cluster::init(1,final_positions.row(first_cluster_ind),first_cluster_ind);

            self.clusters.push(first_cluster);

            while available_points.len() > 0 {

                let mut moved_points = vec![];

                for point_index in available_points.iter().cloned() {

                    let point = final_positions.row(point_index);

                    for cluster in self.clusters.iter_mut() {
                        if distance(point,cluster.center.view()) < self.fuzz[point_index].0 * 3. {
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

    pub fn convergent_points(&self) -> Vec<usize> {
        let mut convergent_points = Vec::new();
        for (i,fuzz) in self.fuzz.iter().enumerate() {
            if fuzz.0 < self.parameters.scaling_factor.unwrap_or(0.1) * self.parameters.convergence_factor.unwrap_or(5.) * 2. {
                convergent_points.push(i);
            }
            // else { eprintln!("conv:{},{},{}", fuzz.0,self.parameters.scaling_factor.unwrap_or(0.1),self.parameters.convergence_factor.unwrap_or(5.)) }
        }
        convergent_points
    }

    pub fn best_cluster_candidate(&self, from_points: Option<&HashSet<usize>>) -> Option<usize> {

        let mut best_candidate = (0,0.);
        let mut any_candidate = false;
        let mut candidate_points = HashSet::with_capacity(0);
        if from_points.is_some() {
            candidate_points = from_points.unwrap().clone();
        }
        else {
            candidate_points = self.convergent_points().into_iter().collect();
        };
        for point in candidate_points.iter() {
            if self.point_factor(*point) > best_candidate.1 {
                best_candidate = (*point,self.point_factor(*point));
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

    pub fn merge_clusters(&mut self) {

        let mut clusters = self.clusters.clone();

        loop {

            let mut new_cluster: Option<Cluster> = None;
            let mut removed_clusters: Option<(usize,usize)> = None;

            'i_loop: for i in 0..clusters.len() {


                let c1 = &clusters[i];

                let mut c1_deviations = Vec::with_capacity(c1.weight);
                for member in &c1.members {
                    let member_coordinates = self.gravity_points.row(*member);
                    c1_deviations.push(distance(member_coordinates, c1.center.view()));
                }
                let c1d: f64 = 2. * (c1_deviations.iter().sum::<f64>() / c1_deviations.len() as f64);

                'j_loop: for j in 0..clusters.len() {
                    if i != j {
                        let c2 = &clusters[j];

                        let mut c2_deviations = Vec::with_capacity(c2.weight);
                        for member in &c2.members {
                            let member_coordinates = self.gravity_points.row(*member);
                            c2_deviations.push(distance(member_coordinates, c2.center.view()));
                        }
                        let c2d: f64 = 2. * (c2_deviations.iter().sum::<f64>() / c2_deviations.len() as f64);

                        if distance(c1.center.view(),c2.center.view()) < c1d.max(c2d) {
                            new_cluster = Some(c1.merge_cluster(c2));
                            removed_clusters = Some((i,j));
                            // eprintln!("Merged:{:?}",removed_clusters);
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


    pub fn point_factor(&self,point_index: usize) -> f64 {
        let point = self.fuzz[point_index];
        if point.0 != 0. {
            point.1/point.0
        }
        else {
            point.1/self.parameters.scaling_factor.unwrap_or(0.1)
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
            members: Vec::new(),
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

    // pub fn radius(&self) -> f64 {
    //     self.array.outer_iter().map(|x| length((&x - &self.center).view())).sum::<f64>() / self.weight as f64
    // }

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
        GravityField::init(raw, Arc::new(parameters));
    }

    #[test]
    #[should_panic]
    pub fn initization_nan() {
        let parameters = Parameters::empty();
        let mut raw = basic_field();
        raw[[1,2]] = f64::NAN;
        GravityField::init(raw, Arc::new(parameters));
    }

    #[test]
    pub fn fit() {
        let parameters = Parameters::empty();
        let raw = basic_field();
        let mut field = GravityField::init(raw, Arc::new(parameters));
        field.fit();
    }

    #[test]
    pub fn predict() {
        let parameters = Parameters::empty();
        let raw = basic_field();
        let mut field = GravityField::init(raw, Arc::new(parameters));
        field.fit();
        field.predict();
    }
}
