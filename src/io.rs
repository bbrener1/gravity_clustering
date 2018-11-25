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

use ndarray::{Array,ArrayView,Ix1,Ix2,Axis};
// use ndarray_linalg::*;


#[derive(Debug,Clone)]
pub struct Parameters {
    auto: bool,
    pub verbose: bool,
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
    pub borrow: Option<usize>,
    pub standardize: bool,
    pub fuzz: Option<usize>,
    pub step_fraction: Option<f64>,

    count_array_file: String,
    feature_header_file: Option<String>,
    sample_header_file: Option<String>,

    processor_limit: Option<usize>,

}

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {
            auto: false,
            verbose: false,
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
            borrow: None,
            standardize: false,
            fuzz: None,
            step_fraction: None,

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
                        eprintln!("If the supress warnings flag is not given first it may not function correctly.");
                    }
                _supress_warnings = true;
                },
                "-auto" | "-a"=> {
                    arg_struct.auto = true;
                    arg_struct.auto()
                },
                "-c" | "-counts" => {
                    arg_struct.count_array_file = args.next().expect("Error parsing count location!");
                    arg_struct.counts = Some(read_counts(&arg_struct.count_array_file,arg_struct.verbose))
                },
                "-verbose" | "-v"=> {
                    arg_struct.verbose = true;
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
                    arg_struct.feature_names = Some(read_header(arg_struct.feature_header_file.as_ref().unwrap(),arg_struct.verbose));
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
                    if arg_struct.sample_subsample.as_ref().unwrap_or(&0) > &(arg_struct.counts.as_ref().unwrap_or(&Array::zeros((0,0))).shape()[0]) {
                        panic!("Subsamples cannot be greater than samples")
                    }
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
                    arg_struct.smoothing = Some(args.next().map(|x| x.parse::<usize>()).expect("Smoothing parse error. Not a number?").expect("Iteration error"));
                },
                "-fuzz" => {
                    arg_struct.smoothing = Some(args.next().map(|x| x.parse::<usize>()).expect("Fuzz parse error. Not a number?").expect("Iteration error"));
                },
                "-step_fraction" => {
                    arg_struct.step_fraction = Some(args.next().map(|x| x.parse::<f64>()).expect("Step fraction parse error. Not a number?").expect("Iteration error"))
                }
                "-borrow" => {
                    arg_struct.borrow = Some(args.next().map(|x| x.parse::<usize>()).expect("Borrowing parse error. Not a number?").expect("Iteration error"));
                },
                "-standardize" => {
                    arg_struct.standardize = true;
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


        eprintln!("Automatic parameters:");
        eprintln!("{:?}",feature_subsample);
        eprintln!("{:?}",sample_subsample);
        eprintln!("{:?}",input_features);
        eprintln!("{:?}",output_features);
        eprintln!("{:?}",processors);

        self.auto = true;

        self.feature_subsample.get_or_insert( feature_subsample );
        self.sample_subsample.get_or_insert( sample_subsample );


        self.processor_limit.get_or_insert( processors );

    }

    pub fn distance(&self, p1:ArrayView<f64,Ix1>,p2:ArrayView<f64,Ix1>) -> f64 {
        self.distance.as_ref().unwrap_or(&Distance::Cosine).measure(p1,p2)
    }

}


fn read_header(location: &str,verbose:bool) -> Vec<String> {

    if verbose {eprintln!("Reading header: {}", location);}

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

    if verbose {eprintln!("Read {} lines", header_vector.len());}

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



fn read_counts(location:&str,verbose:bool) -> Array<f64,Ix2> {


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
                eprint!("\n");
            }

            if i%200==0 && j%200 == 0 {
                eprint!("{} ", gene.parse::<f64>().unwrap_or(-1.) );
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

        if i % 100 == 0 && verbose {
            eprintln!("{}", i);
        }


    };

    let array = Array::from_shape_vec((samples,counts.len()/samples),counts).unwrap_or(Array::zeros((0,0)));

    if verbose {
        eprintln!("===========");
        eprintln!("{},{}", array.shape()[0], array.shape()[1]);
    }

    array
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

                    if exp_val.is_nan() {
                        eprintln!("Read a nan: {:?},{:?}", gene,exp_val);
                        panic!("Read a nan!")
                    }
                    gene_vector.push(exp_val);

                },
                Err(msg) => {

                    if gene != "nan" && gene != "NAN" {
                        eprintln!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        eprintln!("Cell content: {:?}", gene);
                        panic!("Parsing error");
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

    assert!(!array.iter().any(|x| x.is_nan()));

    array
}

pub fn borrow(input: Array<f64,Ix2>, distance:&Distance,verbose:bool) -> Array<f64,Ix2> {


    // let mut means = input.mean_axis(Axis(0));
    // let mut variances = input.var_axis(Axis(0),0.);
    //
    // for i in 0..input.shape()[0] {
    //     let mut row = input.slice_mut(s![i,..]);
    //     // eprintln!("{:?}",row);
    //     let centered = &row - &means;
    //     // eprintln!("{:?}",centered);
    //     let mut standardized = centered * &variances;
    //     for v in standardized.iter_mut() {
    //         if !v.is_finite() {
    //             *v = 0.;
    //         }
    //     }
    //     // eprintln!("{:?}",standardized);
    //     row.assign(&standardized);
    // }

    // eprintln!("{:?}", rla_mtx);

    if verbose {eprintln!("Covariance?");}

    let standardized = standardize(&input);
    // let similarity = correlation_matrix(input.view().t());
    let similarity = standardized.t().dot(&standardized);
    let similarity = match distance {
        Distance::Euclidean => {
            let standardized = standardize(&input);
            standardized.t().dot(&standardized)
        },
        Distance::Cosine => cosine_similarity_matrix(input.t()),
        _ => euclidean_similarity_matrix(input.view().t()),
    } ;

    if verbose {
        eprintln!("{:?}",(similarity.rows(),similarity.cols()));
        eprintln!("Covariance established");
    }
    // let smoothed:Array<f64,Ix2> = cov.dot(&input.t()).reversed_axes();

    let borrowed = Array::from_shape_vec((input.rows(),input.cols()), similarity.dot(&standardized.t()).t().iter().cloned().collect()).unwrap();

    // let borrowed = Array::from_shape_vec((input.rows(),input.cols()), similairty.dot(&input.t()).t().iter().cloned().collect()).unwrap();

    if verbose {eprintln!("Smoothed:{:?}",borrowed.shape());}

    borrowed

}

//
// pub fn pca(mut input: Array<f64,Ix2>,pc_lim:usize) -> (Array<f64,Ix2>,Array<f64,Ix2>) {
//
//     eprintln!("PCA by means of SVD");
//     eprintln!("{:?}",input.shape());
//     eprintln!("Keeping {:?}",pc_lim);
//
    // eprintln!("{:?}", input);
    //
    // if let Ok((Some(u),sig,Some(v))) = input.clone().svd(true,true) {
    //     // let order = argsort(&(0..input.shape()[0]).map(|i| sig[i]).collect());
    //     let mut diagonal: Array<f64,Ix2> = Array::zeros((u.shape()[1],u.shape()[1]));
    //     for (i,sv) in sig.iter().enumerate() {
    //         eprintln!("{:?}",sv);
    //         diagonal[[i,i]] = *sv;
    //     }
    //     let eigenvectors: Array<f64,Ix2> = v.slice(s![.. , 0..pc_lim]).t().to_owned();
    //     let scores: Array<f64,Ix2> = (u * diagonal).slice(s![ .. , 0..pc_lim]).to_owned();
    //
    //     eprintln!("PCA Was Performed, retaining {} PCs", pc_lim);
    //     eprintln!("{:?},{:?}",scores.shape(),eigenvectors.shape());
    //     eprintln!("{:?}",scores.slice(s![..10,..]));
    //
    //     (scores,eigenvectors)
    //
    // }
    //
    // else {
    //     let shape = (input.shape()[0],input.shape()[1]);
    //     eprintln!("WARNING, SVD FAILED, ATTEMPTING CLUSTERING ON WHOLE MATRIX");
    //     (input,Array::zeros(shape))
    // // }

//     let mut rla_mtx: Matrix<f64> = Matrix::new(input.shape()[0],input.shape()[1],input.iter().cloned().collect::<Vec<f64>>());
//
//     let mut means: Matrix<f64> = Matrix::from(rla_mtx.mean(Axes::Row)).transpose();
//     let mut variances: Matrix<f64> = Matrix::from(rla_mtx.variance(Axes::Row).unwrap()).transpose();
//
//     let mut masked: Vec<usize> = variances.iter().enumerate().filter(|x| *x.1 == 0.).map(|x| x.0).collect();
//
//     for column in &masked {
//         rla_mtx = rla_mtx.sub_slice([0,0],rla_mtx.rows(),*column).into_matrix().hcat(&rla_mtx.sub_slice([0,column+1],rla_mtx.rows(),rla_mtx.cols()-column-1).into_matrix());
//         means = means.sub_slice([0,0],means.rows(),*column).into_matrix().hcat(&means.sub_slice([0,column+1],means.rows(),means.cols()-column-1).into_matrix());
//         variances = variances.sub_slice([0,0],variances.rows(),*column).into_matrix().hcat(&variances.sub_slice([0,column+1],variances.rows(),variances.cols()-column-1).into_matrix());
//     }
//
//     eprintln!("Masked {}",masked.len());
//     eprintln!("{:?}",(rla_mtx.rows(),rla_mtx.cols()));
//
//
//     eprintln!("{:?}",variances);
//     eprintln!("{:?}",means);
//
//     for row in rla_mtx.row_iter_mut() {
//         // eprintln!("{:?}",row);
//         let centered = &*row - &means;
//         // eprintln!("{:?}",centered);
//         let mut standardized = centered.elediv(&variances);
//         for v in standardized.iter_mut() {
//             if !v.is_finite() {
//                 *v = 0.;
//             }
//         }
//         // eprintln!("{:?}",standardized);
//         row.set_to(standardized);
//     }
//
//     // eprintln!("{:?}", rla_mtx);
//
//     eprintln!("Covariance?");
//
//     // let cov =  (&rla_mtx.transpose() * &rla_mtx) * (1./ (rla_mtx.rows() - 1) as f64);
//     let cov =  &rla_mtx.transpose() * &rla_mtx;
//     //
//     eprintln!("{:?}",(cov.rows(),cov.cols()));
//
//     eprintln!("Covariance established");
//
//     // cov.clone().eigendecomp().unwrap();
//
//     if let Ok((eigenvalues,eigenvectors_col)) = cov.eigendecomp() {
//         eprintln!("{:?}",eigenvalues);
//         // eprintln!("{:?}", (eigenvectors_col.rows(),eigenvectors_col.cols()));
//         let mut order = argsort(&eigenvalues);
//         order.truncate(pc_lim);
//         // eprintln!("Attempting {},{}",pc_lim, eigenvectors_col.rows());
//         // eprintln!("{}",order.len());
//         let pcs: Matrix<f64> = Matrix::new(order.len(),eigenvectors_col.rows(), order.into_iter().flat_map(|x| eigenvectors_col.col(x).iter().cloned()).collect::<Vec<f64>>());
//         let pcs_col = pcs.transpose();
//         let transformed = rla_mtx * pcs_col;
//
//         let pcs_array: Array<f64,Ix2> = Array::from_shape_vec((pcs.rows(),pcs.cols()),pcs.into_vec()).unwrap();
//         let transformed_array: Array<f64,Ix2> = Array::from_shape_vec((transformed.rows(),transformed.cols()),transformed.into_vec()).unwrap();
//
//         eprintln!("Returning PCA");
//
//         eprintln!("{:?}",transformed_array.shape());
//         eprintln!("{:?}",pcs_array.shape());
//
//         (transformed_array,pcs_array)
//     }
//     else {
//
//         eprintln!("FAILED EIGENDECOMPOSITION");
//         panic!();
//
//         let shape = (input.shape()[0],input.shape()[1]);
//         (input,Array::zeros((shape.0,shape.1)))
//     }
//
//
// }

pub fn standardize(input: &Array<f64,Ix2>) -> Array<f64,Ix2> {
    let mut means = input.mean_axis(Axis(0));
    let mut variances = input.var_axis(Axis(0),0.);

    let mut standardized = input.clone();

    for i in 0..standardized.shape()[0] {
        let mut row = standardized.slice_mut(s![i,..]);
        // eprintln!("{:?}",row);
        let centered_row = &row - &means;
        // eprintln!("{:?}",centered);
        let mut standardized_row = centered_row * &variances;
        for v in standardized_row.iter_mut() {
            if !v.is_finite() {
                *v = 0.;
            }
        }
        // eprintln!("{:?}",standardized);
        row.assign(&standardized_row);
    }

    standardized

}

pub fn sanitize(mut input: Array<f64,Ix2>) -> Array<f64,Ix2> {
    for ref mut feature in input.axis_iter_mut(Axis(1)) {
        if feature.iter().sum::<f64>() == 0. {
            feature.fill(1.)
        }
    };
    input
}


pub fn cosine_similarity_matrix(slice: ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let mut products = slice.dot(&slice.t());
    // eprintln!("Products");
    let mut geo = (&slice * &slice).sum_axis(Axis(1));
    if geo.iter().any(|x| *x == 0.) {
        panic!("Unsanitized input, detected an all-0 feature (column), please use a different distance metric, or sanitize your input");
    }
    // eprintln!("geo");
    geo.mapv_inplace(f64::sqrt);
    for i in 0..slice.rows() {
        for j in 0..slice.rows() {
            products[[i,j]] /= (&geo[i] * &geo[j])
        }
    }
    for i in 0..slice.rows() {
        products[[i,i]] = 1.;
    }
    products
}


pub fn euclidean_similarity_matrix(slice: ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let mut products = slice.dot(&slice.t());
    // eprintln!("Products");
    let mut geo = (&slice * &slice).sum_axis(Axis(1));
    // eprintln!("geo");

    for i in 0..slice.rows() {
        for j in 0..slice.rows() {
            products[[i,j]] = 1.0 / (&geo[i] + &geo[j] - 2.0 * products[[i,j]]).sqrt();
            if !products[[i,j]].is_finite() {
                products[[i,j]] = 1.0;
            }
        }
    }

    for i in 0..slice.rows() {
        products[[i,i]] = 1.0;
    }

    products
}

pub fn correlation_matrix(slice: ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let mut output = Array::zeros((slice.rows(),slice.cols()));
    for i in 0..slice.rows() {
        for j in 0..slice.cols() {
            output[[i,j]] = correlation(slice.row(i),slice.row(j));
        }
    }
    output
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
    Correlation,
}

impl Distance {
    pub fn parse(argument: &str) -> Distance {
        match &argument[..] {
            "manhattan" | "m" | "cityblock" => Distance::Manhattan,
            "euclidean" | "e" => Distance::Euclidean,
            "cosine" | "c" | "cos" => Distance::Cosine,
            "correlation" => Distance::Correlation,
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
            Distance::Correlation => {
                correlation(p1,p2)
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

fn mean(input: &ArrayView<f64,Ix1>) -> f64 {
    input.iter().sum::<f64>() / (input.len() as f64)
}

fn correlation(p1: ArrayView<f64,Ix1>,p2: ArrayView<f64,Ix1>) -> f64 {

    if p1.len() != p2.len() {
        panic!("Tried to compute correlation for unequal length vectors: {}, {}",p1.len(),p2.len());
    }

    let mean1: f64 = mean(&p1);
    let mean2: f64 = mean(&p2);

    let dev1: Vec<f64> = p1.iter().map(|x| (x - mean1)).collect();
    let dev2: Vec<f64> = p2.iter().map(|x| (x - mean2)).collect();

    let covariance = dev1.iter().zip(dev2.iter()).map(|(x,y)| x * y).sum::<f64>() / (p1.len() as f64 - 1.);

    let std_dev1 = (dev1.iter().map(|x| x.powi(2)).sum::<f64>() / (p1.len() as f64 - 1.).max(1.)).sqrt();
    let std_dev2 = (dev2.iter().map(|x| x.powi(2)).sum::<f64>() / (p2.len() as f64 - 1.).max(1.)).sqrt();

    // println!("{},{}", std_dev1,std_dev2);

    let r = covariance / (std_dev1*std_dev2);

    if r.is_nan() {0.} else {r}

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
