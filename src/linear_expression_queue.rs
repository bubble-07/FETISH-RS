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
use crate::sampled_embedder_state::*;

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


fn get_ellipsoid_cost(type_id : TypeId, embedder_state : &SampledEmbedderState, 
                                        ellipsoid : &Ellipsoid) -> f32 {
    if (is_vector_type(type_id)) {
        0.0f32
    } else {
        let d = ellipsoid.dims();
        let zeros = Array::zeros([d,]);
        ellipsoid.sq_mahalanobis_dist(&zeros)
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
                                        embedder_state : &SampledEmbedderState,
                                        feat_inverse_directory : &FeaturizationInverseDirectory) 

                                        -> (LinearExpression, FeaturizedPointsDirectory) {
        self.queue.clear();

        //First, perform initial population of the queue
        let (mut feat_points_directory, mut init_bounded_holed_applications) =
            bounded_hole.get_single_holed_fillers(embedder_state, feat_inverse_directory);

        for init_bounded_holed_application in init_bounded_holed_applications.drain(..) {
            let bound_expr = init_bounded_holed_application.to_linear_expression();
            let hole_type = bound_expr.expr.get_hole_type();
            let bound = &bound_expr.bound;
            let cost = get_ellipsoid_cost(hole_type, embedder_state, bound);

            let prioritized = PritoritizedLinearExpression::new(bound_expr, cost);
            self.queue.push(prioritized);
        }

        //Then, the meat of the search
        //TODO: Iteration cap, and yielding best runner-ups?
        while (!self.queue.is_empty()) {
            let prioritized = self.queue.pop().unwrap();
            let bound_expr = prioritized.expr;

            let bound_hole = bound_expr.get_bounded_hole();

            let term_fillers = bound_hole.get_term_fillers(embedder_state);

            if (term_fillers.len() > 0) {
                let cap = term_fillers[0].clone();
                let result = bound_expr.expr.cap(cap);
                return (result, feat_points_directory);
            } else {
                //No single term fills the hole here, so we need to add neighbors
                let feat_points_delta = self.add_neighbors(&bound_expr, embedder_state, feat_inverse_directory);
                feat_points_directory += feat_points_delta;
            }
        }
        error!("Term search failed");
        panic!();
    }

    pub fn add_neighbors(&mut self, bound_expr : &BoundedHoledLinearExpression, 
                         embedder_state : &SampledEmbedderState,
                         feat_inverse_directory : &FeaturizationInverseDirectory) 
                                                                                 -> FeaturizedPointsDirectory {
        let bounded_hole = bound_expr.get_bounded_hole();
        let (feat_points_directory, mut bounded_holed_applications) = 
            bounded_hole.get_single_holed_fillers(embedder_state, feat_inverse_directory);

        for bounded_holed_application in bounded_holed_applications.drain(..) {

            let hole_type_id = bounded_holed_application.holed_application.get_hole_type();

            let extended_linear_expr = bound_expr.extend_with_holed(bounded_holed_application);
            let hole_bound = &extended_linear_expr.bound;
            let cost = get_ellipsoid_cost(hole_type_id, embedder_state, hole_bound);
            
            let prioritized = PritoritizedLinearExpression::new(extended_linear_expr, cost);
            self.queue.push(prioritized);
        }
        feat_points_directory
    }
}
