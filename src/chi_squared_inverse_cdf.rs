use lazy_static::*;
use crate::params::*;
use statrs::distribution::*;
use argmin::prelude::*;
use argmin::solver::brent::*;
use crate::chi_squared_inverse_cdf_solver::*;

lazy_static! {
    static ref CHI_SQUARED_INVERSE_CDF_TABLE : Vec<f32> = {
        info!("Initializing chi-squared table");
        let mut result = Vec::new();
        result.push(0f32);
        for i in 1..HEURISTIC_ELLIPSE_MAX_DIMENSION {
            let val = chi_squared_inverse_cdf(i, HEURISTIC_ELLIPSE_CI);
            result.push(val);
        }
        info!("Chi-squared table initialized");

        result
    };
}

pub fn init_chi_squared_inverse_cdf_table() {
    CHI_SQUARED_INVERSE_CDF_TABLE[0];
}

pub fn chi_squared_inverse_cdf_for_heuristic_ci(dof : usize) -> f32 {
    CHI_SQUARED_INVERSE_CDF_TABLE[dof]
}

fn chi_squared_inverse_cdf(dof : usize, quantile : f32) -> f32 {
    let distr = ChiSquared::new(dof as f64).unwrap();
    //Find a bracket for the position of the quantile
    let mut x = 0.001f64;
    loop {
        if (distr.cdf(x) > (quantile as f64)) {
            let solver = Brent::new(x / 2.0f64, x, HEURISTIC_ELLIPSE_BRENT_REL_ERROR);

            let problem = ChiSquaredInverseCdfSolver {
                distr : distr,
                quantile : quantile as f64
            };
            let opt_result = Executor::new(problem, solver, x)
                                      .run();
            if let Result::Ok(opt_result) = opt_result {
                let result = opt_result.state.param;
                return result as f32;
            }
            panic!();
        }
        x *= 2.0f64;
    }
}
