extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use ndarray_einsum_beta::*;

use std::rc::*;
use crate::type_id::*;
use crate::application_table::*;
use crate::type_space::*;
use crate::term::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::term_application::*;
use crate::term_application_result::*;
use crate::func_impl::*;
use crate::bayes_utils::*;
use crate::model::*;
use crate::model_space::*;
use crate::schmear::*;
use crate::inverse_schmear::*;
use crate::embedder_state::*;
use crate::interpreter_state::*;

use crate::feature_collection::*;
use crate::linear_feature_collection::*;
use crate::quadratic_feature_collection::*;
use crate::fourier_feature_collection::*;
use crate::cauchy_fourier_features::*;
use crate::enum_feature_collection::*;

pub struct OptimizerStateWithTarget {
    optimizer_state : OptimizerState,
    target_inv_schmear : InverseSchmear,
    target_type_id : TypeId
}

pub struct OptimizerState {
    interpreter_state : InterpreterState,
    embedder_state : EmbedderState,
}

impl OptimizerStateWithTarget {
    pub fn optimize_evaluate_step(&mut self) -> TermPointer {
        self.optimizer_state.optimize_evaluate_step(self.target_type_id, &self.target_inv_schmear)
    }

    fn new(data_points : Vec::<(Array1<f32>, Array1<f32>)>) -> OptimizerStateWithTarget {

        //Step 1: find the embedding of the target term

        if (data_points.is_empty()) {
            panic!(); 
        }
        let in_dimensions : usize = data_points[0].0.shape()[0];
        let out_dimensions : usize = data_points[0].1.shape()[0];

        let in_type_id : TypeId = get_type_id(&Type::VecType(in_dimensions));
        let out_type_id : TypeId = get_type_id(&Type::VecType(out_dimensions));
        let target_type_id : TypeId = get_type_id(&Type::FuncType(in_type_id, out_type_id));

        let feature_collections = get_feature_collections(in_dimensions);
        let rc_feature_collections = Rc::new(feature_collections);
        
        let mut target_model : Model = Model::new(rc_feature_collections, in_dimensions, out_dimensions);

        let out_precision = Array::eye(out_dimensions);
        
        for (in_vec, out_vec) in data_points.iter() {
            let data_point = DataPoint {
                in_vec : in_vec.clone(),
                out_inv_schmear : InverseSchmear {
                    mean : out_vec.clone(),
                    precision : out_precision.clone()
                }
            };
            target_model += data_point;
        }

        let target_inv_schmear : InverseSchmear = target_model.get_inverse_schmear();

        let optimizer_state = OptimizerState::new();
        OptimizerStateWithTarget {
            optimizer_state,
            target_inv_schmear,
            target_type_id
        }
    }
}

impl OptimizerState {
    fn optimize_evaluate_step(&mut self, target_type_id : TypeId, target_inv_schmear : &InverseSchmear)
                                     -> TermPointer {
        //Sample for the best term in the space of the target 
        let (term_pointer, term_dist) = self.embedder_state.thompson_sample_term(target_type_id, target_inv_schmear);

        //Now, iterate through all term applications yielding the type of the target
        let mut application_type_ids : Vec::<(TypeId, TypeId)> = get_application_type_ids(target_type_id);
        let mut best_dist : f32 = f32::INFINITY;
        let mut best_application_and_types : Option<(TermApplication, TypeId, TypeId)> = Option::None;
        
        for (func_type_id, arg_type_id) in application_type_ids.drain(..) {
            let (application, dist) = self.embedder_state.thompson_sample_app(func_type_id, arg_type_id, 
                                                                              target_inv_schmear);
            if (dist < best_dist) {
                best_application_and_types = Option::Some((application, func_type_id, arg_type_id));
                best_dist = dist;
            }
        }
        match best_application_and_types {
            Option::None => term_pointer,
            Option::Some(application_and_types) => {
                let (application, func_type_id, arg_type_id) = application_and_types;
                if (best_dist < term_dist) {
                    let target_mean = &target_inv_schmear.mean;
                    let (func_target, maybe_arg_target) = 
                        self.embedder_state.find_better_app(&application, target_mean);

                    let better_func = self.optimize_evaluate_step(func_type_id, &func_target);

                    let better_arg : TermReference = match (maybe_arg_target) {
                        Some(arg_target) => {
                            let better_arg = self.optimize_evaluate_step(arg_type_id, &arg_target);
                            TermReference::FuncRef(better_arg)
                        },
                        None => {
                            //In this case, we just optimized the function choice
                            //since the argument choice was already the best possible
                            application.arg_ref 
                        }
                    };
                    let better_application = TermApplication {
                        func_ptr : better_func,
                        arg_ref : better_arg
                    };
                    let result_ref = self.interpreter_state.evaluate(&better_application);
                    if let TermReference::FuncRef(ret_ptr) = result_ref {
                        ret_ptr
                    } else {
                        //This should not happen, because we should never be using the optimizer
                        //to attempt to find applications yielding bare vectors
                        panic!();
                    }
                } else {
                    term_pointer
                }
            }
        }

    }
    fn new() -> OptimizerState {
        OptimizerState {
            interpreter_state : InterpreterState::new(),
            embedder_state : EmbedderState::new()
        }
    }
}