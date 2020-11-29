extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::*;
use ndarray_linalg::*;

use crate::array_utils::*;
use crate::holed_application::*;
use crate::bounded_holed_application::*;
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
use crate::interpreter_state::*;
use crate::displayable_with_state::*;

pub struct BoundedHole {
    pub type_id : TypeId,
    pub bound : Ellipsoid
}

impl BoundedHole {
    pub fn get_term_fillers(&self, embedder_state : &SampledEmbedderState) -> Vec<TermReference> {
        let mut result = Vec::new();

        if (!is_vector_type(self.type_id)) {
            let embedding_space = embedder_state.embedding_spaces.get(&self.type_id).unwrap();
            for term_id in embedding_space.models.keys() {
                let embedding = embedding_space.get_embedding(*term_id);
                let embedding_vec = embedding.get_flattened();
                if self.bound.contains(&embedding_vec) {
                    let term_ptr = TermPointer {
                        type_id : self.type_id,
                        index : *term_id
                    };
                    let term_ref = TermReference::FuncRef(term_ptr);
                    result.push(term_ref)
                }
            }
        } else {
            let center = self.bound.center();
            let noisy = to_noisy(center);
            let term_ref = TermReference::VecRef(noisy);
            result.push(term_ref);
        }
        result
    }

    pub fn get_single_holed_fillers(&self,
                            interpreter_state : &InterpreterState,
                            embedder_state : &SampledEmbedderState,
                            feat_inverse_directory : &FeaturizationInverseDirectory)
                                                     
                        -> (FeaturizedPointsDirectory, Vec<BoundedHoledApplication>) {

        let mut bounded_holes = Vec::new();
        let mut feat_points_directory = FeaturizedPointsDirectory::new(embedder_state);

        let ret_type = self.type_id;

        let ret_type_sketcher = &embedder_state.get_space_info(&ret_type).func_sketcher;
        let ret_expansion_matrix = ret_type_sketcher.get_expansion_matrix();

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

                    let out_bound = self.bound.backpropagate_through_transform(&ret_expansion_matrix);

                    //Derive the ellipsoid on the vectorized transform
                    let func_ellipsoid = out_bound.backpropagate_to_vectorized_transform(&arg_feat_vec);

                    //Now, package this information up into a new holed expression
                    let arg_ptr = TermPointer {
                        type_id : *arg_type,
                        index : *arg_index
                    };
                    let arg_ref = TermReference::FuncRef(arg_ptr);
                    let holed_app = HoledApplication::FunctionHoled(arg_ref, ret_type);

                    trace!("Adding function-holed: {}", holed_app.format_string(interpreter_state, "_".to_owned()));

                    let bounded_holed_app = BoundedHoledApplication::new(holed_app, func_ellipsoid);
                    bounded_holes.push(bounded_holed_app);
                }
            }
            
            //Then, handle the argument hole case, which may fail
            for func_index in func_embedding_space.models.keys() {
                let func_ptr = TermPointer {
                    type_id : *func_type,
                    index : *func_index
                };
                trace!("Investigating argument-holed case for {}", func_ptr.display(interpreter_state));

                let func_embedding = func_embedding_space.models.get(func_index).unwrap();
                let func_then_expand = ret_expansion_matrix.dot(func_embedding);
                //Now, get the bound on the featurized argument
                let feat_bound = self.bound.backpropagate_through_transform(&func_then_expand);
                
                //Now, try to propagate the feature-space bound through to the input space
                let mut feat_points = feat_points_directory.get_space(func_type);
                let inverse_model = feat_inverse_directory.get(func_type);
                
                let mut rng = rand::thread_rng();

                //Get a sampling of inputs for an initial featurized point
                let sampled_inputs = inverse_model.sample(&mut rng, &feat_bound, 
                                                          NUM_FUNCTION_SAMPLES, NUM_ELLIPSOID_SAMPLES);

                trace!("Attempting backpropagation through featurization");
                let maybe_contained_input = feat_bound.approx_backpropagate_through_featurization_contained_vec(
                                                        &mut feat_points, sampled_inputs);

                if let Option::Some(contained_input) = maybe_contained_input {
                    let input_bound = 
                        if (is_vector_type(*arg_type)) {
                            Ellipsoid::from_single_point(contained_input)
                        } else {
                            let compressed_input_bound = feat_bound.approx_enclosing_ellipsoid(
                                                                   &mut feat_points, &contained_input);
                            let arg_type_sketcher = &embedder_state.get_space_info(arg_type).func_sketcher;
                            let arg_compress = arg_type_sketcher.get_projection_matrix();
                            compressed_input_bound.backpropagate_through_transform(&arg_compress)
                        };

                    //We have a concrete bound on what the input needs to be.
                    //If the input type is a vector type, then we're completely done
                    //because we can just output the center.
                    //Otherwise, we need to package this information into a new holed expression
                    let holed = HoledApplication::ArgumentHoled(func_ptr.clone());
                    let bounded_holed_app = BoundedHoledApplication::new(holed, input_bound);
                    bounded_holes.push(bounded_holed_app);
                }
            }
        }
        (feat_points_directory, bounded_holes)
    }
}
