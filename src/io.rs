use std::fs::File;
use std::fs::OpenOptions;
use std::io::Error;

use std::io;
use std::io::prelude::*;
use std::collections::HashMap;
use num_cpus;
use std::f64;
use std::fmt::Debug;
use rayon::iter::IntoParallelIterator;
use std::cmp::Ordering;

use ndarray::{Array,ArrayView,Ix1,Ix2};
use ndarray_linalg::*;

// use rulinalg::matrix::{Matrix,BaseMatrix,BaseMatrixMut,Axes};

#[derive(Debug,Clone)]
pub struct Parameters {
    auto: bool,
    pub command: Command,
    pub counts: Option<Array<f64,Ix2>>,
    pub feature_names: Option<Vec<String>>,
    pub sample_names: Option<Vec<String>>,
    pub report_address: Option<String>,
    pub dump_error: Option<String>,

    pub feature_subsample: Option<usize>,
    pub sample_subsample: Option<usize>,
    pub scaling_factor: Option<f64>,
    pub merge_distance: Option<f64>,
    pub convergence_factor: Option<f64>,
    pub locality: Option<f64>,
    pub refining: bool,
    pub smoothing: Option<usize>,
    pub distance: Option<Distance>,

    count_array_file: String,
    feature_header_file: Option<String>,
    sample_header_file: Option<String>,

    processor_limit: Option<usize>,

}

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {
            auto: false,
            command: Command::FitPredict,
            count_array_file: "".to_string(),
            counts: None,
            feature_header_file: None,
            feature_names: None,
            sample_header_file: None,
            sample_names: None,
            report_address: None,
            dump_error: None,
            distance: None,

            processor_limit: None,

            feature_subsample: None,
            sample_subsample: None,
            merge_distance: None,
            convergence_factor: None,
            locality: None,
            refining: false,
            smoothing: None,

            scaling_factor: None,

        };
        arg_struct
    }

    pub fn read<T: Iterator<Item = String>>(args: &mut T) -> Parameters {

        let mut arg_struct = Parameters::empty();

        let _raw_command = args.next();

        arg_struct.command = Command::parse(&args.next().expect("Please enter a command"));

        let mut _supress_warnings = false;

        while let Some((i,arg)) = args.enumerate().next() {

                match &arg[..] {
                "-sw" | "-suppress_warnings" => {
                    if i!=1 {
                        println!("If the supress warnings flag is not given first it may not function correctly.");
                    }
                _supress_warnings = true;
                },
                "-auto" | "-a"=> {
                    arg_struct.auto = true;
                    arg_struct.auto()
                },
                "-c" | "-counts" => {
                    arg_struct.count_array_file = args.next().expect("Error parsing count location!");
                    arg_struct.counts = Some(read_counts(&arg_struct.count_array_file))
                },
                "-stdin" => {
                    arg_struct.counts = Some(read_standard_in());
                }
                "-stdout" => {
                    arg_struct.report_address = None;
                }
                "-p" | "-processors" | "-threads" => {
                    arg_struct.processor_limit = Some(args.next().expect("Error processing processor limit").parse::<usize>().expect("Error parsing processor limit"));
                },
                "-o" | "-output" => {
                    arg_struct.report_address = Some(args.next().expect("Error processing output destination"))
                },
                "-f" | "-h" | "-features" | "-header" => {
                    arg_struct.feature_header_file = Some(args.next().expect("Error processing feature file"));
                    arg_struct.feature_names = Some(read_header(arg_struct.feature_header_file.as_ref().unwrap()));
                },
                "-s" | "-samples" => {
                    arg_struct.sample_header_file = Some(args.next().expect("Error processing feature file"));
                    arg_struct.sample_names = Some(read_sample_names(arg_struct.sample_header_file.as_ref().unwrap()));
                }
                "-fs" | "-feature_sub" => {
                    arg_struct.feature_subsample = Some(args.next().expect("Error processing feature subsample arg").parse::<usize>().expect("Error feature subsample arg"));
                },
                "-ss" | "-sample_sub" => {
                    arg_struct.sample_subsample = Some(args.next().expect("Error processing sample subsample arg").parse::<usize>().expect("Error sample subsample arg"));
                },
                "-scaling" | "-step" | "-sf" | "-scaling_factor" => {
                    arg_struct.scaling_factor = Some(args.next().map(|x| x.parse::<f64>()).expect("Scaling factor parse error. Not a number?").expect("Iteration error"));
                },
                "-m" | "-merge" | "-merge_distance" => {
                    arg_struct.merge_distance = Some(args.next().map(|x| x.parse::<f64>()).expect("Merge distance parse error. Not a number?").expect("Iteration error"));
                },
                "-error" => {
                    arg_struct.dump_error = Some(args.next().expect("Error processing error destination"))
                },
                "-convergence" => {
                    arg_struct.convergence_factor = Some(args.next().map(|x| x.parse::<f64>()).expect("Convergence distance parse error. Not a number?").expect("Iteration error"));
                },
                "-smoothing" => {
                    arg_struct.smoothing = Some(args.next().map(|x| x.parse::<usize>()).expect("Smoothing distance parse error. Not a number?").expect("Iteration error"));
                },

                "-l" | "-locality" => {
                    arg_struct.locality = Some(args.next().map(|x| x.parse::<f64>()).expect("Locality parse error. Not a number?").expect("Iteration error"))
                },
                "-r" | "-refining" => {
                    arg_struct.refining = true;
                },
                "-d" | "-distance" => {
                    arg_struct.distance = Some(args.next().map(|x| Distance::parse(&x)).expect("Distance parse error"))
                }

                &_ => {
                    panic!("Not a valid argument: {}", arg);
                }

            }
        }

        arg_struct

    }



    fn auto(&mut self) {

        let counts = self.counts.as_ref().expect("Please specify counts file before the \"-auto\" argument.");

        let features = counts.shape()[1];
        let samples = counts.shape()[0];

        let mut output_features = ((features as f64 / (features as f64).log10()) as usize).min(features);

        let input_features: usize;

        if features < 3 {
            input_features = features;
            output_features = features;
        }
        else if features < 100 {
            input_features = ((features as f64 * ((125 - features) as f64) / 125.) as usize).max(1);
        }

        else {
            input_features = ((features as f64 * (((1500 - features as i32) as f64) / 7000.).max(0.1)) as usize).max(1);
        }

        let feature_subsample = output_features;

        let sample_subsample: usize;

        if samples < 10 {
            eprintln!("Warning, you seem to be using suspiciously few samples, are you sure you specified the right file? If so, trees may not be the right solution to your problem.");
            sample_subsample = samples;
        }
        else if samples < 1000 {
            sample_subsample = (samples/3)*2;
        }
        else if samples < 5000 {
            sample_subsample = samples/2;
        }
        else {
            sample_subsample = samples/4;
        }

        let processors = num_cpus::get();


        println!("Automatic parameters:");
        println!("{:?}",feature_subsample);
        println!("{:?}",sample_subsample);
        println!("{:?}",input_features);
        println!("{:?}",output_features);
        println!("{:?}",processors);

        self.auto = true;

        self.feature_subsample.get_or_insert( feature_subsample );
        self.sample_subsample.get_or_insert( sample_subsample );


        self.processor_limit.get_or_insert( processors );

    }

    pub fn distance(&self, p1:ArrayView<f64,Ix1>,p2:ArrayView<f64,Ix1>) -> f64 {
        self.distance.as_ref().unwrap_or(&Distance::Cosine).measure(p1,p2)
    }

}


fn read_header(location: &str) -> Vec<String> {

    println!("Reading header: {}", location);

    let mut header_map = HashMap::new();

    let header_file = File::open(location).expect("Header file error!");
    let mut header_file_iterator = io::BufReader::new(&header_file).lines();

    for (i,line) in header_file_iterator.by_ref().enumerate() {
        let feature = line.unwrap_or("error".to_string());
        let mut renamed = feature.clone();
        let mut j = 1;
        while header_map.contains_key(&renamed) {
            renamed = [feature.clone(),j.to_string()].join("");
            eprintln!("WARNING: Two individual features were named the same thing: {}",feature);
            j += 1;
        }
        header_map.insert(renamed,i);
    };

    let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x| (x.0.clone().clone(),x.1.clone())).collect();
    header_inter.sort_unstable_by_key(|x| x.1);
    let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();

    println!("Read {} lines", header_vector.len());

    header_vector
}

fn read_sample_names(location: &str) -> Vec<String> {

    let mut header_vector = Vec::new();

    let sample_name_file = File::open(location).expect("Sample name file error!");
    let mut sample_name_lines = io::BufReader::new(&sample_name_file).lines();

    for line in sample_name_lines.by_ref() {
        header_vector.push(line.expect("Error reading header line!").trim().to_string())
    }

    header_vector
}



fn read_counts(location:&str) -> Array<f64,Ix2> {


    let count_array_file = File::open(location).expect("Count file error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let mut counts: Vec<f64> = Vec::new();
    let mut samples = 0;

    for (i,line) in count_array_lines.by_ref().enumerate() {

        samples += 1;
        let mut gene_vector = Vec::new();

        let gene_line = line.expect("Readline error");

        for (j,gene) in gene_line.split_whitespace().enumerate() {

            if j == 0 && i%200==0{
                print!("\n");
            }

            if i%200==0 && j%200 == 0 {
                print!("{} ", gene.parse::<f64>().unwrap_or(-1.) );
            }

            // if !((gene.0 == 1686) || (gene.0 == 4660)) {
            //     continue
            // }

            match gene.parse::<f64>() {
                Ok(exp_val) => {

                    gene_vector.push(exp_val);

                },
                Err(msg) => {

                    if gene != "nan" && gene != "NAN" {
                        println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        println!("Cell content: {:?}", gene);
                    }
                    gene_vector.push(f64::NAN);
                }
            }

        }

        counts.append(&mut gene_vector);

        if i % 100 == 0 {
            println!("{}", i);
        }


    };

    let array = Array::from_shape_vec((samples,counts.len()/samples),counts).unwrap_or(Array::zeros((0,0)));

    println!("===========");
    println!("{},{}", array.shape()[0], array.shape()[1]);

    pca(array,50).0

}

fn read_standard_in() -> Array<f64,Ix2> {

    let stdin = io::stdin();
    let count_array_pipe_guard = stdin.lock();

    let mut counts: Vec<f64> = Vec::new();
    let mut samples = 0;

    for (_i,line) in count_array_pipe_guard.lines().enumerate() {

        samples += 1;
        let mut gene_vector = Vec::new();

        for (_j,gene) in line.as_ref().expect("readline error").split_whitespace().enumerate() {

            match gene.parse::<f64>() {
                Ok(exp_val) => {

                    gene_vector.push(exp_val);

                },
                Err(msg) => {

                    if gene != "nan" && gene != "NAN" {
                        println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        println!("Cell content: {:?}", gene);
                    }
                    gene_vector.push(f64::NAN);
                }
            }

        }

        counts.append(&mut gene_vector);

    };

    // eprintln!("Counts read:");
    // eprintln!("{:?}", counts);

    let array = Array::from_shape_vec((samples,counts.len()/samples),counts).unwrap_or(Array::zeros((0,0)));

    pca(array,50).0
}

pub fn pca(input: Array<f64,Ix2>,pc_lim:usize) -> (Array<f64,Ix2>,Array<f64,Ix2>) {


    if let Ok((Some(u),sig,Some(v))) = input.svd(true,true) {
        // let order = argsort(&(0..input.shape()[0]).map(|i| sig[i]).collect());
        let eigenvectors: Array<f64,Ix2> = v.slice(s![.. , 0..pc_lim]).to_owned();
        let scores: Array<f64,Ix2> = (u * sig).slice(s![ .. , 0..pc_lim]).to_owned();
        (scores,eigenvectors)
    }

    else {
        let shape = (input.shape()[0],input.shape()[1]);
        eprintln!("WARNING, SVD FAILED, ATTEMPTING CLUSTERING ON WHOLE MATRIX");
        (input,Array::zeros(shape))
    }

    // let mut rla_mtx: Matrix<f64> = Matrix::new(input.shape()[0],input.shape()[1],input.iter().cloned().collect::<Vec<f64>>());
    // let means: Matrix<f64> = Matrix::from(rla_mtx.mean(Axes::Row)).transpose();
    // let variances: Matrix<f64> = Matrix::from(rla_mtx.variance(Axes::Row).unwrap()).transpose();
    // // eprintln!("{:?}",(rla_mtx.rows(),rla_mtx.cols()));
    // // eprintln!("{:?}",(variances.rows(),variances.cols()));
    // // eprintln!("{:?}",(means.rows(),means.cols()));
    //
    // for row in rla_mtx.row_iter_mut() {
    //     // eprintln!("{:?}",(row.rows(),row.cols()));
    //     let centered = &*row - &means;
    //     // eprintln!("{:?}",(centered.rows(),centered.cols()));
    //     let standardized = centered.elediv(&variances);
    //     // eprintln!("{:?}",(standardized.rows(),standardized.cols()));
    //     row.set_to(standardized);
    // }
    //
    // eprintln!("Covariance?");
    //
    // let cov =  (&rla_mtx.transpose() * &rla_mtx) * (1./ (rla_mtx.rows() - 1) as f64);
    //
    // eprintln!("Covariance established");
    //
    // if let Ok((eigenvalues,eigenvectors_col)) = cov.eigendecomp() {
    //     let mut order = argsort(&eigenvalues);
    //     order.truncate(pc_lim);
    //     let pcs: Matrix<f64> = Matrix::new(pc_lim,eigenvectors_col.rows(), order.into_iter().flat_map(|x| eigenvectors_col.col(x).iter().cloned()).collect::<Vec<f64>>());
    //     let pcs_col = pcs.transpose();
    //     let transformed = rla_mtx * pcs_col;
    //
    //     let pcs_array: Array<f64,Ix2> = Array::from_shape_vec((pcs.rows(),pcs.cols()),pcs.into_vec()).unwrap();
    //     let transformed_array: Array<f64,Ix2> = Array::from_shape_vec((transformed.rows(),transformed.cols()),transformed.into_vec()).unwrap();
    //
    //     (transformed_array,pcs_array)
    // }
    // else {
    //     let shape = (input.shape()[0],input.shape()[1]);
    //     (input,Array::zeros((shape.0,shape.1)))
    // }


}

fn argsort(input: &Vec<f64>) -> Vec<usize> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| x.0).collect();
    out
}

#[derive(Debug,Clone,Copy)]
pub enum Distance {
    Manhattan,
    Euclidean,
    Cosine,
}

impl Distance {
    pub fn parse(argument: &str) -> Distance {
        match &argument[..] {
            "manhattan" | "m" | "cityblock" => Distance::Manhattan,
            "euclidean" | "e" => Distance::Euclidean,
            "cosine" | "c" | "cos" => Distance::Cosine,
            _ => {
                eprintln!("Not a valid distance option, defaulting to cosine");
                Distance::Cosine
            }
        }
    }

    pub fn measure(&self,p1:ArrayView<f64,Ix1>,p2:ArrayView<f64,Ix1>) -> f64 {
        match self {
            Distance::Manhattan => {
                (&p1 - &p2).scalar_sum()
            },
            Distance::Euclidean => {
                (&p1 - &p2).map(|x| x.powi(2)).scalar_sum().sqrt()
            },
            Distance::Cosine => {
                let dot_product = p1.dot(&p2);
                let p1ss = p1.map(|x| x.powi(2)).scalar_sum().sqrt();
                let p2ss = p2.map(|x| x.powi(2)).scalar_sum().sqrt();
                1.0 - (dot_product / (p1ss * p2ss))
            }
        }
    }
}


#[derive(Debug,Clone)]
pub enum Command {
    Fit,
    Predict,
    FitPredict,
    Fuzzy,
    Mobile,
}

impl Command {

    pub fn parse(command: &str) -> Command {

        match &command[..] {
            "fit" => Command::Fit,
            "predict" => Command::Predict,
            "fitpredict" | "fit_predict" | "combined" => Command::FitPredict,
            "fuzzy_predict" | "fuzzy" => Command::Fuzzy,
            "mobile" => Command::Mobile,
            _ =>{
                eprintln!("Not a valid top-level command, please choose from \"fit\",\"predict\", or \"fitpredict\". Exiting");
                panic!()
            }
        }
    }
}


pub fn write_array<T: Debug>(input: Array<T,Ix2>,target:&Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .outer_iter()
        .map(|x| x.iter()
            .map(|y| format!("{:?}",y))
            .collect::<Vec<String>>()
            .join("\t")
        )
        .collect::<Vec<String>>()
        .join("\n");

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }
}

pub fn write_vector<T: Debug>(input: Array<T,Ix1>,target: &Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .iter()
        .map(|x| format!("{:?}",x))
        .collect::<Vec<String>>()
        .join("\n");

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }
}


//
// fn tsv_format<T:Debug>(input:&Vec<Vec<T>>) -> String {
//
//     input.iter().map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")
//
// }










//
