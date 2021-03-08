use crate::term_pointer::*;
use crate::term_reference::*;
use crate::type_action::*;
use crate::type_id::*;
use std::cmp::*;
use std::fmt::*;
use std::collections::HashSet;
use crate::interpreter_state::*;
use crate::displayable_with_state::*;

pub struct TypeGraph {
    //Indices: from, to, edge index
    edges : Vec<Vec<Vec<TypeAction>>>
}

impl TypeGraph {
    pub fn get_types_which_reach_through_nonvec_path(&self, dest : TypeId) -> HashSet<TypeId> {
        let mut result = HashSet::new();
        let mut up_next = Vec::new();
        up_next.push(dest);
        result.insert(dest);
        while (!up_next.is_empty()) {
            let current = up_next.pop().unwrap();

            for pred_current in 0..total_num_types() {
                if (!is_vector_type(pred_current) &&
                    !result.contains(&pred_current) && 
                    !self.edges[pred_current][current].is_empty()) {

                    result.insert(pred_current);
                    up_next.push(pred_current);
                }
            }
        }
        result
    }
    pub fn build() -> TypeGraph {
        let mut edges = Vec::new();
        for from_type_id in 0..total_num_types() {
            let mut row = Vec::new();
            for to_type_id in 0..total_num_types() {
                let elem = TypeAction::get_actions_for(from_type_id, to_type_id);
                row.push(elem);
            }
            edges.push(row);
        }
        TypeGraph {
            edges
        }
    }
}
