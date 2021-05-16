//! **F**unctional **E**mbedding of **T**erms **I**n a **S**patial **H**ierarchy
//!
//! A library for an active research project which implements a small, yet
//! extensible interpreter for a simply-typed combinator-based language,
//! along with a mechanism for deriving _embeddings_ (in the machine learning sense)
//! for terms of each type in the language. 
//!
//! For starting points on this library, see the Rustdoc on [`crate::context::Context`] and 
//! [`crate::interpreter_and_embedder_state::InterpreterAndEmbedderState`]
//! 
//! For a description of what exactly is implemented, see
//! <https://github.com/bubble-07/FETISH-RS/blob/master/FETISH.pdf>
//!
//! Or, alternatively, check out the talk at this link:
//! <https://drive.google.com/file/d/1BrbJivs-VohTdji8Y7C4-O7xTWdugQ78/view?usp=sharing>
//!
//! Please notify ajg137@case.edu if you would like to contribute to this research project, 
//! and we can arrange for an initial conversation.

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]

#[macro_use] extern crate log;
pub mod multiple;
pub mod term_input_output;
pub mod everything;
pub mod schmeared_hole;
pub mod typed_vector;
pub mod newly_evaluated_terms;
pub mod input_to_schmeared_output;
pub mod nonprimitive_term_pointer;
pub mod primitive_term_pointer;
pub mod term_index;
pub mod displayable_with_context;
pub mod primitive_type_space;
pub mod primitive_directory;
pub mod context;
pub mod compressed_inv_schmear;
pub mod prior_specification;
pub mod elaborator;
pub mod kernel;
pub mod space_info;
pub mod interpreter_and_embedder_state;
pub mod feature_space_info;
pub mod function_space_info;
pub mod sampled_model_embedding;
pub mod sqrtm;
pub mod data_points;
pub mod rand_utils;
pub mod term_model;
pub mod sampled_embedder_state;
pub mod sampled_embedding_space;
pub mod sigma_points;
pub mod sherman_morrison;
pub mod data_point;
pub mod normal_inverse_wishart;
pub mod normal_inverse_wishart_sampler;
pub mod wishart;
pub mod test_utils;
pub mod array_utils;
pub mod pseudoinverse;
pub mod linear_sketch;
pub mod linalg_utils;
pub mod feature_collection;
pub mod sketched_linear_feature_collection;
pub mod embedder_state;
pub mod count_sketch;
pub mod quadratic_feature_collection;
pub mod fourier_feature_collection;
pub mod func_scatter_tensor;
pub mod model;
pub mod inverse_schmear;
pub mod func_schmear;
pub mod func_inverse_schmear;
pub mod schmear;
pub mod embedding_space;
pub mod type_id;
pub mod params;
pub mod application_table;
pub mod interpreter_state;
pub mod func_impl;
pub mod term;
pub mod term_pointer;
pub mod term_reference;
pub mod term_application_result;
pub mod term_application;
pub mod type_space;
pub mod displayable_with_state;
