extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::*;
use ndarray_linalg::*;

use crate::holed_application::*;
use crate::holed_linear_expression::*;
use crate::bounded_holed_linear_expression::*;
use crate::term_pointer::*;
use crate::term_reference::*;
use crate::bounded_hole::*;
use crate::ellipsoid::*;
use crate::type_id::*;
use crate::featurized_points_directory::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::params::*;
use crate::linear_expression::*;
use crate::featurization_inverse_directory::*;
use crate::embedder_state::*;
use crate::sampled_embedder_state::*;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;
use crate::sum_of_joint_probabilities_heuristic::*;

struct PritoritizedLinearExpression {
    expr : BoundedHoledLinearExpression,
    dist : f32
}

impl PritoritizedLinearExpression {
    pub fn new(bounded_holed : BoundedHoledLinearExpression, dist : f32) -> PritoritizedLinearExpression {
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


fn get_ellipsoid_cost(type_id : TypeId, embedder_state : &EmbedderState,
                                        sampled_embedder_state : &SampledEmbedderState, 
                                        ellipsoid : &Ellipsoid) -> f32 {
    if (is_vector_type(type_id)) {
        f32::NEG_INFINITY
    } else {
        let model_space = embedder_state.model_spaces.get(&type_id).unwrap();
        let neg_result = sum_of_joint_probabilities_heuristic(model_space, ellipsoid);
        -neg_result
    }
}


impl LinearExpressionQueue {
    pub fn new() -> LinearExpressionQueue {
        let queue = BinaryHeap::new();
        LinearExpressionQueue {
            queue
        }
    }

    pub fn find_within_bound(&mut self, bounded_hole : &BoundedHole,
                                        interpreter_state : &InterpreterState,
                                        embedder_state : &EmbedderState,
                                        sampled_embedder_state : &SampledEmbedderState,
                                        feat_inverse_directory : &FeaturizationInverseDirectory) 

                                        -> (LinearExpression, FeaturizedPointsDirectory) {
        self.queue.clear();

        //First, perform initial population of the queue
        trace!("Finding initial single-holed fillers");
        let (mut feat_points_directory, mut init_bounded_holed_applications) =
            bounded_hole.get_single_holed_fillers(interpreter_state, sampled_embedder_state, feat_inverse_directory);

        trace!("Inserting initial elements into queue");
        for init_bounded_holed_application in init_bounded_holed_applications.drain(..) {
            let bound_expr = init_bounded_holed_application.to_linear_expression();
            let hole_type = bound_expr.expr.get_hole_type();
            let bound = &bound_expr.bound;
            let cost = get_ellipsoid_cost(hole_type, embedder_state, sampled_embedder_state, bound);

            let prioritized = PritoritizedLinearExpression::new(bound_expr, cost);
            self.queue.push(prioritized);
        }

        //Then, the meat of the search
        //TODO: Iteration cap, and yielding best runner-ups?
        while (!self.queue.is_empty()) {
            let prioritized = self.queue.pop().unwrap();
            let bound_expr = prioritized.expr;

            trace!("Popped: {}", bound_expr.expr.display(interpreter_state));

            let bound_hole = bound_expr.get_bounded_hole();

            let term_fillers = bound_hole.get_term_fillers(sampled_embedder_state);

            if (term_fillers.len() > 0) {
                let cap = term_fillers[0].clone();
                trace!("Found a term filler: {}", cap.display(interpreter_state));
                let result = bound_expr.expr.cap(cap);
                return (result, feat_points_directory);
            } else {
                trace!("No term filler found, adding neighbors");
                //No single term fills the hole here, so we need to add neighbors
                let feat_points_delta = self.add_neighbors(&bound_expr, interpreter_state, embedder_state, sampled_embedder_state, feat_inverse_directory);
                feat_points_directory += feat_points_delta;
            }
        }
        error!("Term search failed");
        panic!();
    }

    pub fn add_neighbors(&mut self, bound_expr : &BoundedHoledLinearExpression, 
                         interpreter_state : &InterpreterState,
                         embedder_state : &EmbedderState,
                         sampled_embedder_state : &SampledEmbedderState,
                         feat_inverse_directory : &FeaturizationInverseDirectory) 
                                                                                 -> FeaturizedPointsDirectory {
        let bounded_hole = bound_expr.get_bounded_hole();
        let (feat_points_directory, mut bounded_holed_applications) = 
            bounded_hole.get_single_holed_fillers(interpreter_state, sampled_embedder_state, feat_inverse_directory);

        for bounded_holed_application in bounded_holed_applications.drain(..) {

            let hole_type_id = bounded_holed_application.holed_application.get_hole_type();

            let extended_linear_expr = bound_expr.extend_with_holed(bounded_holed_application);
            let hole_bound = &extended_linear_expr.bound;
            let cost = get_ellipsoid_cost(hole_type_id, embedder_state, sampled_embedder_state, hole_bound);
            
            let prioritized = PritoritizedLinearExpression::new(extended_linear_expr, cost);
            self.queue.push(prioritized);
        }
        feat_points_directory
    }
}
