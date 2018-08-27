
use std::env;

extern crate rand;
extern crate num_cpus;

#[macro_use]
extern crate rulinalg;
#[macro_use]
extern crate ndarray;

mod pathfinder;
mod io;
mod gravity_field;
use io::{write_array,write_vector};
use gravity_field::GravityField;
use pathfinder::Pathfinder;
use io::{Parameters,Command};

use std::io::Error;


fn main() -> Result<(),Error> {

    // GravityField::manual_test();
    //
    // Ok(())

    let mut arg_iter = env::args();

    let mut parameters = Parameters::read(&mut arg_iter);

    let mut field = GravityField::init(parameters.counts.unwrap(), parameters.scaling_factor, parameters.sample_subsample);

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
            let predictions = field.predict();
            write_vector(predictions, parameters.report_address)?;
            Ok(())
        }
        _ => panic!("Invalid command? Error in parameter module")
    }

}
