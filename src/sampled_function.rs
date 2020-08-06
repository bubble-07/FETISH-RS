extern crate argmin;
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use std::rc::*;
use crate::model::*;
use crate::params::*;
use crate::test_utils::*;
use crate::inverse_schmear::*;
use crate::enum_feature_collection::*;

extern crate pretty_env_logger;


#[derive(Clone)]
pub struct SampledFunction {
    pub in_dimensions : usize,
    pub mat : Array2<f32>,
    pub feature_collections : Rc<[EnumFeatureCollection; 3]>
}

impl SampledFunction {
    pub fn apply(&self, input : &Array1<f32>) -> Array1<f32> {
        let features : Array1<f32> = to_features(&self.feature_collections, input);
        let result : Array1<f32> = self.mat.dot(&features);
        result
    }

    pub fn get_closest_arg_to_target(self, target : InverseSchmear) -> (Array1<f32>, f32) {
        let linesearch = MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();
        let solver = LBFGS::new(linesearch, LBFGS_HISTORY);

        //Set the initial parameter vector to all-zeroes
        let init_param : Array1<f32> = Array::zeros((self.in_dimensions,));

        let function_target = SampledFunctionTarget::new(self, target);

        let maybe_result = Executor::new(function_target, solver, init_param.clone())
                     .max_iters(NUM_OPT_ITERS)
                     .run();
        match (maybe_result) {
            Result::Ok(result) => (result.state.param, result.state.cost),
            Result::Err(err) => {
                error!("Optimizer failed: {}", err);    
                (init_param, f32::INFINITY)
            }
        }
    }
}

struct SampledFunctionTarget {
    func : SampledFunction,
    target : InverseSchmear,
    y_t_s_m : Array1<f32>,
    m_t_s_m : Array2<f32>
}

impl ArgminOp for SampledFunctionTarget {
    type Param = Array1<f32>;
    type Output = f32;
    type Hessian = Array2<f32>;
    type Jacobian = ();
    type Float = f32;

    fn apply(&self, p : &Self::Param) -> Result<Self::Output, Error> {
        let output : Array1<f32> = self.func.apply(p);
        let dist : f32 = self.target.mahalanobis_dist(&output);
        Ok(dist)
    }
    fn gradient(&self, p : &Self::Param) -> Result<Self::Param, Error> {
        let features = to_features(&self.func.feature_collections, p);
        let jacobian = to_jacobian(&self.func.feature_collections, p);
        let const_term = self.y_t_s_m.dot(&jacobian);
        let mut result = features.dot(&self.m_t_s_m).dot(&jacobian);

        result -= &const_term;
        result *= 2.0f32;
        Ok(result)
    }
}


impl SampledFunctionTarget {
    fn new(func : SampledFunction, target : InverseSchmear) -> SampledFunctionTarget {
        let s = &target.precision;
        let m = &func.mat;
        let y = &target.mean;

        let s_m : Array2<f32> = s.dot(m);
        let y_t_s_m = y.dot(&s_m);
        let m_t_s_m = m.t().dot(&s_m);

        SampledFunctionTarget {
            func : func,
            target : target,
            y_t_s_m : y_t_s_m,
            m_t_s_m : m_t_s_m
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_gradient_is_gradient() {
        let in_dimension = 6;
        let out_dimension = 7;
        let mut successes : usize = 0;
        for i in 0..10 {
            let sampled_function = random_sampled_function(in_dimension, out_dimension);
            let target = random_inv_schmear(out_dimension);
            let sampled_function_target = SampledFunctionTarget::new(sampled_function, target);
            let in_vec = random_vector(in_dimension);

            let gradient = sampled_function_target.gradient(&in_vec).unwrap();
            let empirical_gradient = empirical_gradient(|x| sampled_function_target.apply(x).unwrap(), &in_vec);

            let test = are_equal_vectors_to_within(&gradient, &empirical_gradient, 10.0f32, false);
            if (test) {
                successes += 1;
            }
        }
        if (successes < 5) {
            panic!();
        }
    }
}
