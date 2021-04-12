use crate::type_graph::*;
use crate::type_id::*;
use crate::space_info::*;
use crate::func_impl::*;
use crate::function_space_info::*;
use crate::feature_space_info::*;
use crate::primitive_directory::*;
use crate::primitive_term_pointer::*;

pub struct Context {
    pub type_info_directory : TypeInfoDirectory,
    pub space_info_directory : SpaceInfoDirectory,
    pub primitive_directory : PrimitiveDirectory
}

pub fn get_default_context() -> Context {
    let type_info_directory = get_default_type_info_directory();
    let space_info_directory = get_default_space_info_directory(&type_info_directory);
    let primitive_directory =  get_default_primitive_directory(&type_info_directory);
    Context {
        type_info_directory,
        space_info_directory,
        primitive_directory
    }
}

impl Context {
    //Primitive information
    pub fn get_primitive(&self, primitive_term_pointer : PrimitiveTermPointer) -> &dyn FuncImpl {
        self.primitive_directory.get_primitive(primitive_term_pointer)
    }

    //Space information
    pub fn get_feature_space_info(&self, type_id : TypeId) -> &FeatureSpaceInfo {
        self.space_info_directory.get_feature_space_info(type_id)
    }
    pub fn build_function_space_info(&self, arg_type_id : TypeId, ret_type_id : TypeId) -> FunctionSpaceInfo {
        let arg_feat_info = self.get_feature_space_info(arg_type_id);
        let ret_feat_info = self.get_feature_space_info(ret_type_id);
        FunctionSpaceInfo {
            in_feat_info : arg_feat_info,
            out_feat_info : ret_feat_info
        }
    }
    pub fn get_function_space_info(&self, func_type_id : TypeId) -> FunctionSpaceInfo {
        let func_type = self.get_type(func_type_id);
        match (func_type) {
            Type::FuncType(arg_type_id, ret_type_id) => {
                let arg_feat_info = self.get_feature_space_info(arg_type_id);
                let ret_feat_info = self.get_feature_space_info(ret_type_id);
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

    //Type Information
    pub fn get_type_id(&self, kind : &Type) -> TypeId {
        self.type_info_directory.get(kind)
    }
    pub fn get_type(&self, id : TypeId) -> Type {
        self.type_info_directory.get_type(id)
    }
    pub fn has_type(&self, kind : &Type) -> bool {
        self.type_info_directory.has_type(kind)
    }
    pub fn get_application_type_ids(&self, id : TypeId) -> Vec::<(TypeId, TypeId)> {
        self.type_info_directory.get_application_type_ids(id)
    }
    pub fn get_total_num_types(&self) -> usize {
        self.type_info_directory.get_total_num_types()
    }
    pub fn get_dimension(&self, vec_type_id : TypeId) -> usize {
        self.type_info_directory.get_dimension(vec_type_id)
    }
    pub fn get_arg_type_id(&self, func_type_id : TypeId) -> TypeId {
        self.type_info_directory.get_arg_type_id(func_type_id)
    }
    pub fn get_ret_type_id(&self, func_type_id : TypeId) -> TypeId {
        self.type_info_directory.get_ret_type_id(func_type_id)
    }
    pub fn is_vector_type(&self, id : TypeId) -> bool {
        self.type_info_directory.is_vector_type(id)
    }
}
