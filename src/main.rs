use std::env;
use std::io::Write;
use std::fs::File;
use std::fs::OpenOptions;


extern crate rand;
extern crate num_cpus;

#[macro_use]
extern crate ndarray;
// extern crate ndarray_linalg;
// extern crate ndarray_parallel;
extern crate rayon;

mod io;
mod mobile_gravity_field;
mod single_pathfinder;
mod cluster;
use io::{write_array,write_vector};
use mobile_gravity_field::GravityField;
use io::{Parameters,Command,Distance};
use ndarray::{Array,Axis,Ix1,Ix2,Zip,ArrayView};
use std::sync::Arc;
use std::io::Error;
use io::borrow;
use io::standardize;

fn main() -> Result<(),Error> {

    // GravityField::manual_test();
    //
    // Ok(())

    let mut arg_iter = env::args();

    let mut parameters_raw = Parameters::read(&mut arg_iter);

    let mut gravity_points = parameters_raw.counts.take().unwrap();

    if parameters_raw.standardize {
        gravity_points = standardize(&gravity_points);
    }

    for _ in 0..*parameters_raw.borrow.as_ref().unwrap_or(&0)  {
        gravity_points = borrow(gravity_points,parameters_raw.distance.as_ref().unwrap_or(&Distance::Cosine),parameters_raw.verbose);
    }

    let mut parameters = Arc::new(parameters_raw);

    eprintln!("Clustering with parameters:{:?}",parameters);

    let mut field = GravityField::init(gravity_points, parameters.clone());

    match parameters.command {
        Command::Fit => {
            // field.fit();
            Ok(())
        },
        Command::Predict => {
            // field.predict();
            Ok(())
        }
        Command::FitPredict => {
            let mut final_positions = field.fit();
            let mut predictions = field.fuzzy_predict();

            if parameters.refining {
                let mut refining_parameters = Arc::make_mut(&mut parameters).clone();

                let mut refining_field = GravityField::init(final_positions.clone(),Arc::new(refining_parameters));

                final_positions = refining_field.fit();
                predictions = refining_field.fuzzy_predict();

                field = refining_field;
            }

            write_vector(predictions, &parameters.report_address)?;

            if parameters.dump_error.is_some() {
                write_array(final_positions, &parameters.dump_error.clone().map(|x| [x,"final_pos.tsv".to_string()].join("")))?;
                // let clusters: Vec<Array<f64,Ix1>> = field.clusters.iter().map(|x| x.id).collect();
                // let mut cluster_acc = Array::zeros((0,field.gravity_points.shape()[1]));
                // for cluster in clusters {
                //     cluster_acc = stack!(Axis(0),cluster_acc,cluster.insert_axis(Axis(1)).t());
                // }
                // // eprintln!("{:?}",cluster_acc.shape());
                // write_array(cluster_acc, &parameters.dump_error.clone().map(|x| [x,"clusters.tsv".to_string()].join("")))?;
            }
            Ok(())
        },
        Command::Fuzzy => {
            let mut final_positions = field.fuzzy_fit_single();
            let mut predictions = field.fuzzy_predict();

            let initial_fuzz = field.fuzz;

            if parameters.refining {

                let mut refining_parameters = Arc::make_mut(&mut parameters).clone();

                refining_parameters.scaling_factor = refining_parameters.scaling_factor.map(|x| x/5.);

                let mut refining_field = GravityField::init(final_positions,Arc::new(refining_parameters));

                final_positions = refining_field.fuzzy_fit_single();

                refining_field.fuzz = initial_fuzz;

                predictions = refining_field.fuzzy_predict();

                field = refining_field;
            }

            write_vector(predictions, &parameters.report_address)?;
            if parameters.dump_error.is_some() {
                write_array(final_positions, &parameters.dump_error.clone().map(|x| [x,"final_pos.tsv".to_string()].join("")))?;
            }
            Ok(())
        },
        Command::Mobile => {
            let mut final_positions = field.fuzzy_fit_mobile();
            let mut predictions = field.fuzzy_predict();

            write_vector(predictions, &parameters.report_address)?;
            if parameters.dump_error.is_some() {
                write_array(final_positions, &parameters.dump_error.clone().map(|x| [x,"final_pos.tsv".to_string()].join("")))?;
                let mut cluster_file = OpenOptions::new().create(true).append(true).open([parameters.dump_error.as_ref().unwrap(),"clusters.tsv"].join("")).unwrap();
                for cluster in field.clusters {
                    cluster_file.write(format!("{:?}",cluster.center()).as_bytes())?;
                }
            }

            Ok(())
        }

    }

}

pub fn length(v: ArrayView<f64,Ix1>) -> f64 {
    v.fold(0.,|acc,x| acc+x.powi(2)).sqrt()
}

pub fn distance(pa1:ArrayView<f64,Ix1>,pa2:ArrayView<f64,Ix1>) -> f64 {
    let mut acc = 0.;
    Zip::from(pa1).and(pa2).apply(|p1,p2| acc += (*p1 - *p2).powi(2));
    acc = acc.sqrt();
    acc
}

pub fn cos_similarity(pa1:ArrayView<f64,Ix1>,pa2:ArrayView<f64,Ix1>) -> f64 {
    let product_sum = (&pa1 * &pa2).sum();
    let p1ss = pa1.map(|x| x.powi(2)).sum().sqrt();
    let p2ss = pa2.map(|x| x.powi(2)).sum().sqrt();
    product_sum / (p1ss * p2ss)
}

pub fn sq_distance(pa1:ArrayView<f64,Ix1>,pa2:ArrayView<f64,Ix1>) -> f64 {
    let mut acc = 0.;
    Zip::from(pa1).and(pa2).apply(|p1,p2| acc += (*p1 - *p2).powi(2));
    acc
}
