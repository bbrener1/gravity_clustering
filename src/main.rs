
use std::env;

extern crate rand;
extern crate num_cpus;

#[macro_use]
extern crate ndarray;

mod pathfinder;
mod io;
mod gravity_field;
use io::{write_array,write_vector};
use gravity_field::GravityField;
use io::{Parameters,Command};
use ndarray::{Array,Axis,Ix1};

use std::io::Error;


fn main() -> Result<(),Error> {

    // GravityField::manual_test();
    //
    // Ok(())

    let mut arg_iter = env::args();

    let mut parameters = Parameters::read(&mut arg_iter);

    let mut field = GravityField::init(parameters.counts.take().unwrap(), &parameters);

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
            field.predict();

            let mut refining_field = GravityField::init(field.final_positions.unwrap(),&parameters);

            refining_field.fit();
            let predictions = refining_field.predict();

            write_vector(predictions, &parameters.report_address)?;

            if parameters.dump_error.is_some() {
                write_array(refining_field.final_positions.unwrap(), &parameters.dump_error.clone().map(|x| [x,"final_pos.tsv".to_string()].join("")))?;
                let clusters: Vec<Array<f64,Ix1>> = refining_field.clusters.iter().map(|(cluster,_label)| cluster.clone()).collect();
                let mut cluster_acc = Array::zeros((0,2));
                for cluster in clusters {
                    cluster_acc = stack!(Axis(0),cluster_acc,cluster.insert_axis(Axis(1)).t());
                }
                write_array(cluster_acc, &parameters.dump_error.clone().map(|x| [x,"clusters.tsv".to_string()].join("")))?;
            }
            Ok(())
        }
    }

}
