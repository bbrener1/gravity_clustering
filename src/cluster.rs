use std::sync::Arc;
use std::collections::HashSet;
use ndarray::{Array,Ix1,Ix2,Ix3,Zip,Axis,ArrayView,stack};
use std::f64;
use std::cmp::max;
use rayon::prelude::*;

use io::Parameters;

use single_pathfinder::Pathfinder;
use io::Distance;
use length;

#[derive(Clone)]
pub struct Cluster {
    pub id: usize,
    pub weight: usize,
    pub radius: f64,
    pub center: Array<f64,Ix1>,
    pub members: Vec<usize>,
    pub array: Arc<Array<f64,Ix2>>,
    distance: Distance
}

impl Cluster {
    pub fn init(id: usize, points: Arc<Array<f64,Ix2>>, point_id: usize, parameters: Arc<Parameters>) -> Cluster {
        let point = points.row(point_id);
        let mut array = Array::zeros((1,point.shape()[0]));
        array.row_mut(0).assign(&point);
        Cluster {
            id: id,
            weight: 1,
            radius: 0.0,
            center: point.to_owned(),
            members: vec![point_id],
            array: points.clone(),
            distance: parameters.distance.unwrap_or(Distance::Cosine)
        }
    }

    pub fn merge_cluster(&self,cluster: &Cluster) -> Cluster {
        let new_weight = self.weight + cluster.weight;
        // let new_center = ((&self.center * self.weight as f64) + (&cluster.center * cluster.weight as f64)) / (new_weight) as f64;
        let new_members = [self.members.iter(),cluster.members.iter()].iter().flat_map(|x| x.clone()).cloned().collect();
        let mut new_cluster = Cluster {
            id : self.id,
            weight: new_weight,
            radius: 0.0,
            center: self.center.clone(),
            members: new_members,
            array: self.array.clone(),
            distance: self.distance
        };
        new_cluster.center = new_cluster.center();
        new_cluster.radius = new_cluster.radius();
        new_cluster
    }

    pub fn merge_point(&mut self ,point: ArrayView<f64,Ix1>, point_id:usize) -> usize {

        self.center *= self.weight as f64/(self.weight as f64 + 1.);
        self.center.scaled_add(1./(self.weight as f64 + 1.), &point);
        self.weight += 1;
        self.members.push(point_id);
        self.radius = self.radius();
        self.id
    }

    pub fn radius(&self) -> f64 {
        let radius = self.members.iter().map(|x| self.distance.measure(self.array.row(*x).view(), self.center.view())).sum::<f64>() / self.weight as f64;
        // eprintln!("R:{:?}",radius);
        radius
    }

    pub fn center(&self) -> Array<f64,Ix1> {
        let mut center = Array::zeros(self.array.shape()[1]);
        for i in self.members.iter() {
            center += &(&self.array.row(*i) / self.weight as f64);
        }
        center
    }

}
