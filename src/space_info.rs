use ndarray::*;

use std::collections::HashMap;
use lazy_static::*;
use crate::type_id::*;
use crate::feature_space_info::*;
use crate::function_space_info::*;
use topological_sort::TopologicalSort;

extern crate pretty_env_logger;

lazy_static! {
    static ref GLOBAL_SPACE_INFO : GlobalSpaceInfo = {
        let mut feature_spaces = HashMap::<TypeId, FeatureSpaceInfo>::new();
        
        let mut topo_sort = TopologicalSort::<TypeId>::new();
        for type_id in 0..total_num_types() {
            match get_type(type_id) {
                Type::FuncType(arg_type_id, ret_type_id) => {
                    topo_sort.add_dependency(arg_type_id, type_id);
                    topo_sort.add_dependency(ret_type_id, type_id);
                },
                Type::VecType(dim) => {
                    let feat_space = FeatureSpaceInfo::build_uncompressed_feature_space(dim);
                    feature_spaces.insert(type_id, feat_space);
                }
            };
        }
        
        while (topo_sort.len() > 0) {
            let mut type_ids : Vec<TypeId> = topo_sort.pop_all();
            for func_type_id in type_ids.drain(..) {
                if let Type::FuncType(arg_type_id, ret_type_id) = get_type(func_type_id) {
                    let in_feat_info = feature_spaces.get(&arg_type_id).unwrap();
                    let out_feat_info = feature_spaces.get(&ret_type_id).unwrap();
                    let func_feat_info = FeatureSpaceInfo::build_function_feature_space(in_feat_info, out_feat_info);

                    feature_spaces.insert(func_type_id, func_feat_info);
                }
            }
        }

        let mut vectorized_feature_spaces = Vec::new();
        for type_id in 0..total_num_types() {
            let feat_space = feature_spaces.remove(&type_id).unwrap();
            vectorized_feature_spaces.push(feat_space);
        }
        GlobalSpaceInfo {
            feature_spaces : vectorized_feature_spaces
        }
    };
}

struct GlobalSpaceInfo {
    feature_spaces : Vec<FeatureSpaceInfo>
}

pub fn get_feature_space_info(type_id : TypeId) -> &'static FeatureSpaceInfo {
    &GLOBAL_SPACE_INFO.feature_spaces[type_id]
}

pub fn build_function_space_info(arg_type_id : TypeId, ret_type_id : TypeId) -> FunctionSpaceInfo<'static> {
    let arg_feat_info = get_feature_space_info(arg_type_id);
    let ret_feat_info = get_feature_space_info(ret_type_id);
    FunctionSpaceInfo {
        in_feat_info : arg_feat_info,
        out_feat_info : ret_feat_info
    }
}

pub fn get_function_space_info(func_type_id : TypeId) -> FunctionSpaceInfo<'static> {
    let func_type = get_type(func_type_id);
    match (func_type) {
        Type::FuncType(arg_type_id, ret_type_id) => {
            let arg_feat_info = get_feature_space_info(arg_type_id);
            let ret_feat_info = get_feature_space_info(ret_type_id);
            FunctionSpaceInfo {
                in_feat_info : arg_feat_info,
                out_feat_info : ret_feat_info
            }
        },
        Type::VecType(_) => {
            panic!();
        }
    }
}
