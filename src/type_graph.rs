use crate::type_action::*;
use lazy_static::*;
use crate::type_id::*;
use std::collections::HashSet;

lazy_static! {
    static ref GLOBAL_TYPE_GRAPH : TypeGraph = {
        TypeGraph::build()
    };
}

pub struct TypeGraph {
    //Indices: from, to, edge index
    edges : Vec<Vec<Vec<TypeAction>>>,
    //Reachability predicates
    reachable : Vec<Vec<bool>>,
    //Immediate successors of nodes
    successors : Vec<Vec<TypeId>>
}

pub fn get_type_actions(src : TypeId, dest : TypeId) -> &'static Vec<TypeAction> {
    GLOBAL_TYPE_GRAPH.get_actions(src, dest)
}

pub fn get_type_successors(src : TypeId) -> &'static Vec<TypeId> {
    GLOBAL_TYPE_GRAPH.get_successors(src)
}

pub fn is_type_reachable_from(src : TypeId, dest : TypeId) -> bool {
    GLOBAL_TYPE_GRAPH.is_reachable_from(src, dest)
}

impl TypeGraph {
    pub fn get_actions(&self, src : TypeId, dest : TypeId) -> &Vec<TypeAction> {
        &self.edges[src][dest]
    }
    pub fn get_successors(&self, src : TypeId) -> &Vec<TypeId> {
        &self.successors[src]
    }
    pub fn is_reachable_from(&self, src : TypeId, dest : TypeId) -> bool {
        self.reachable[src][dest]
    }

    fn floyd_warshall(edges : &Vec<Vec<Vec<TypeAction>>>) -> Vec<Vec<bool>> {
        let mut result = Vec::new();
        for i in 0..total_num_types() {
            let mut row = Vec::new();
            for j in 0..total_num_types() {
                let edge_exists = i == j || edges[i][j].len() > 0;
                row.push(edge_exists);
            }
            result.push(Vec::new());
        }

        for k in 0..total_num_types() {
            for i in 0..total_num_types() {
                for j in 0..total_num_types() {
                    result[i][j] |= result[i][k] && result[k][j];
                }
            }
        }

        result
    }

    pub fn successors(edges : &Vec<Vec<Vec<TypeAction>>>) -> Vec<Vec<TypeId>> {
        let mut result = Vec::new();
        for source in 0..total_num_types() {
            let mut successors = Vec::new();
            for dest in 0..total_num_types() {
                if edges[source][dest].len() > 0 {
                    successors.push(dest);
                }
            }
            result.push(successors);
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
        let reachable = TypeGraph::floyd_warshall(&edges);
        let successors = TypeGraph::successors(&edges);
        TypeGraph {
            edges,
            reachable,
            successors
        }
    }
}
