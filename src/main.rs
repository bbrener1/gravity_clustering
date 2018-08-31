
use std::env;

extern crate rand;
extern crate num_cpus;

#[macro_use]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate rayon;

mod pathfinder;
mod io;
mod gravity_field;
use io::{write_array,write_vector};
use gravity_field::GravityField;
use io::{Parameters,Command};
use ndarray::{Array,Axis,Ix1};
use std::sync::Arc;

use std::io::Error;


fn main() -> Result<(),Error> {

    // GravityField::manual_test();
    //
    // Ok(())

    let mut arg_iter = env::args();

    let mut parameters_raw = Parameters::read(&mut arg_iter);

    let gravity_points = parameters_raw.counts.take().unwrap();

    let mut parameters = Arc::new(parameters_raw);

    let mut field = GravityField::init(gravity_points, parameters.clone());

    match parameters.command {
        Command::Fit => {
            field.fit();
            Ok(())
        },
        Command::Predict => {
            field.predict();
            Ok(())
        }
        Command::FitPredict => {
            field.fit();
            let mut predictions = field.predict();

            if parameters.refining {
                let mut refining_parameters = Arc::make_mut(&mut parameters).clone();

                let mut refining_field = GravityField::init(field.final_positions.unwrap(),Arc::new(refining_parameters));

                refining_field.fit();
                predictions = refining_field.predict();

                field = refining_field;
            }

            write_vector(predictions, &parameters.report_address)?;

            if parameters.dump_error.is_some() {
                write_array(field.final_positions.unwrap(), &parameters.dump_error.clone().map(|x| [x,"final_pos.tsv".to_string()].join("")))?;
                let clusters: Vec<Array<f64,Ix1>> = field.clusters.iter().map(|(cluster,_label)| cluster.clone()).collect();
                let mut cluster_acc = Array::zeros((0,field.gravity_points.shape()[1]));
                for cluster in clusters {
                    cluster_acc = stack!(Axis(0),cluster_acc,cluster.insert_axis(Axis(1)).t());
                }
                // eprintln!("{:?}",cluster_acc.shape());
                write_array(cluster_acc, &parameters.dump_error.clone().map(|x| [x,"clusters.tsv".to_string()].join("")))?;
            }
            Ok(())
        }
    }

}
