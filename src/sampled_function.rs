extern crate argmin;
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use std::rc::*;
use crate::model::*;
use crate::inverse_schmear::*;
use crate::enum_feature_collection::*;

const NUM_OPT_ITERS : u64 = 100;
const LBFGS_HISTORY : usize = 10;

#[derive(Clone)]
pub struct SampledFunction {
    pub in_dimensions : usize,
    pub mat : Array2<f32>,
    pub feature_collections : Rc<[EnumFeatureCollection; 3]>
}

impl SampledFunction {
    pub fn apply(&self, input : &Array1<f32>) -> Array1<f32> {
        let features : Array1<f32> = to_features(&self.feature_collections, input);
        let result : Array1<f32> = einsum("ab,b->a", &[&self.mat, &features])
                                         .unwrap().into_dimensionality::<Ix1>().unwrap();
        result
    }

    pub fn get_closest_arg_to_target(self, target : InverseSchmear) -> (Array1<f32>, f32) {
        let linesearch = MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();
        let solver = LBFGS::new(linesearch, LBFGS_HISTORY);

        //Set the initial parameter vector to all-zeroes
        let init_param : Array1<f32> = Array::zeros((self.in_dimensions,));

        let function_target = SampledFunctionTarget::new(self, target);

        let result = Executor::new(function_target, solver, init_param)
                     .max_iters(NUM_OPT_ITERS)
                     .run().unwrap();
        (result.state.param, result.state.cost)
        
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
        let const_term : Array1<f32> = einsum("a,ab->a", &[&self.y_t_s_m, &jacobian]).unwrap()
                                       .into_dimensionality::<Ix1>().unwrap();
        let mut result : Array1<f32> = einsum("a,ab,bc->c", &[&features, &self.m_t_s_m, &jacobian]).unwrap()
                                       .into_dimensionality::<Ix1>().unwrap();
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

        let y_t_s_m : Array1<f32> = einsum("a,ab,bc->c", &[y, s, m]).unwrap()
                                    .into_dimensionality::<Ix1>().unwrap();
        let m_t_s_m : Array2<f32> = einsum("ba,bc,cd->ad", &[m, s, m]).unwrap()
                                    .into_dimensionality::<Ix2>().unwrap();
        SampledFunctionTarget {
            func : func,
            target : target,
            y_t_s_m : y_t_s_m,
            m_t_s_m : m_t_s_m
        }
    }
}

