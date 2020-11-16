extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::*;
use ndarray_linalg::*;

use crate::holed_application::*;
use crate::holed_linear_expression::*;
use crate::bounded_holed_linear_expression::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::ellipsoid::*;
use crate::type_id::*;
use crate::featurized_points_directory::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::params::*;
use crate::featurization_inverse_directory::*;
use crate::sampled_embedder_state::*;

struct PritoritizedLinearExpression {
    expr : BoundedHoledLinearExpression,
    dist : f32
}

impl PritoritizedLinearExpression {
    pub fn new(expr : HoledLinearExpression, bound : Ellipsoid, dist : f32) -> PritoritizedLinearExpression {
        let bounded_holed = BoundedHoledLinearExpression {
            expr : expr,
            bound : bound
        };
        PritoritizedLinearExpression {
            expr : bounded_holed,
            dist : dist
        }
    }
}

impl Ord for PritoritizedLinearExpression {
    fn cmp(&self, other : &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for PritoritizedLinearExpression {
    fn partial_cmp(&self, other : &Self) -> Option<Ordering> {
        other.dist.partial_cmp(&self.dist)
    }
}

impl PartialEq for PritoritizedLinearExpression {
    fn eq(&self, other : &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for PritoritizedLinearExpression { }


pub struct LinearExpressionQueue {
    queue : BinaryHeap<PritoritizedLinearExpression>
}


fn get_ellipsoid_cost(type_id : TypeId, embedder_state : &SampledEmbedderState, 
                                        ellipsoid : &Ellipsoid) -> f32 {
    if (is_vector_type(type_id)) {
        0.0f32
    } else {
        let d = ellipsoid.dims();
        let zeros = Array::zeros([d,]);
        ellipsoid.mahalanobis_dist(&zeros)
    }
}


impl LinearExpressionQueue {

    /*
    pub fn find_within_bound(&mut self, type_id : TypeId, bound : &Ellipsoid,
                                        embedder_state : &SampledEmbedderState,
                                        feat_inverse_directory : &FeaturizationInverseDirectory) 
                                                                            -> LinearExpression {
    }*/

    pub fn add_neighbors(&mut self, bound_expr : &BoundedHoledLinearExpression, 
                         embedder_state : &SampledEmbedderState,
                         feat_inverse_directory : &FeaturizationInverseDirectory) 
                                                                                 -> FeaturizedPointsDirectory {
        let mut feat_points_directory = FeaturizedPointsDirectory::new(embedder_state);

        let expr = &bound_expr.expr;
        let bound = &bound_expr.bound;
        let ret_type = expr.get_type();
        let application_type_ids = get_application_type_ids(ret_type);
        for (func_type, arg_type) in application_type_ids.iter() {
            let func_embedding_space = embedder_state.embedding_spaces.get(func_type).unwrap();

            //First, handle the function hole case, which is irrefutable.
            //We only do this if the argument type is not a vector
            if !is_vector_type(*arg_type) {
                let arg_embedding_space = embedder_state.embedding_spaces.get(arg_type).unwrap();
                for arg_index in arg_embedding_space.models.keys() {
                    let arg_embedding = arg_embedding_space.get_embedding(*arg_index);
                    let arg_vec = arg_embedding.get_compressed();
                    //Now we featurize the argument vector according to the function type
                    let feat_points = feat_points_directory.get_space(func_type);
                    let arg_feat_vec = feat_points.get_features(&arg_vec);
                    //Derive the ellipsoid on the vectorized transform
                    let func_ellipsoid = bound.backpropagate_to_vectorized_transform(&arg_feat_vec);

                    //Now, package this information up into a new holed expression
                    let ellipsoid_cost = get_ellipsoid_cost(*func_type, embedder_state, &func_ellipsoid);
                    let arg_ptr = TermPointer {
                        type_id : *arg_type,
                        index : *arg_index
                    };
                    let arg_ref = TermReference::FuncRef(arg_ptr);
                    let holed_app = HoledApplication::FunctionHoled(arg_ref, ret_type);
                    let extended_expr = expr.extend_with_holed(holed_app);

                    let wrapper = PritoritizedLinearExpression::new(extended_expr, func_ellipsoid, ellipsoid_cost);
                    self.queue.push(wrapper);
                }
            }
            
            //Then, handle the argument hole case, which may fail
            for func_index in func_embedding_space.models.keys() {
                let func_embedding = func_embedding_space.models.get(func_index).unwrap();
                //Now, get the bound on the featurized argument
                let feat_bound = bound.backpropagate_through_transform(func_embedding);
                
                //Now, try to propagate the feature-space bound through to the input space
                let mut feat_points = feat_points_directory.get_space(func_type);
                let inverse_model = feat_inverse_directory.get(func_type);
                
                let mut rng = rand::thread_rng();

                //Get a sampling of inputs for an initial featurized point
                let sampled_inputs = inverse_model.sample(&mut rng, &feat_bound, 
                                                          NUM_FUNCTION_SAMPLES, NUM_ELLIPSOID_SAMPLES);

                let maybe_input_bound = feat_bound.approx_backpropagate_through_featurization(&mut feat_points,
                                                                    sampled_inputs);
                if let Option::Some(input_bound) = maybe_input_bound {
                    //We have a concrete bound on what the input needs to be.
                    //If the input type is a vector type, then we're completely done
                    //because we can just output the center.
                    //Otherwise, we need to package this information into a new holed expression
                    let func_ptr = TermPointer {
                        type_id : *func_type,
                        index : *func_index
                    };
                    let holed = HoledApplication::ArgumentHoled(func_ptr);
                    let extended_expr = expr.extend_with_holed(holed);
                    let ellipsoid_cost = get_ellipsoid_cost(*arg_type, embedder_state, &input_bound);

                    let wrapper = PritoritizedLinearExpression::new(extended_expr, input_bound, ellipsoid_cost);
                    self.queue.push(wrapper);
                }
            }
        }
        feat_points_directory
    }
}
